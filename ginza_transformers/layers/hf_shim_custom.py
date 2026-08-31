import srsly

from spacy_transformers.data_classes import HFObjects
from spacy_transformers.layers import hf_shim
from spacy_transformers.layers.hf_shim import HFShim
from thinc.api import get_torch_default_device


def override_hf_shims_to_bytes():
    assert hf_shim.HFShim.to_bytes is not HFShimCustom.to_bytes
    origin = hf_shim.HFShim.to_bytes
    hf_shim.HFShim.to_bytes = HFShimCustom.to_bytes
    return origin

def recover_hf_shims_to_bytes(origin):
    assert hf_shim.HFShim.to_bytes is HFShimCustom.to_bytes
    hf_shim.HFShim.to_bytes = origin


def override_hf_shims_from_bytes():
    assert hf_shim.HFShim.from_bytes is not HFShimCustom.from_bytes
    origin = hf_shim.HFShim.from_bytes
    hf_shim.HFShim.from_bytes = HFShimCustom.from_bytes
    return origin

def recover_hf_shims_from_bytes(origin):
    assert hf_shim.HFShim.from_bytes is HFShimCustom.from_bytes
    hf_shim.HFShim.from_bytes = origin


class HFShimCustom(HFShim):

    def to_bytes(self):
        msg = {
            "config": self._hfmodel.transformer.config.to_dict(),
            "_init_tokenizer_config": self._hfmodel._init_tokenizer_config,
            "_init_transformer_config": self._hfmodel._init_transformer_config,
        }
        return srsly.msgpack_dumps(msg)

    def from_bytes(self, bytes_data):
        msg = srsly.msgpack_loads(bytes_data)
        config_dict = msg["config"]
        _init_tokenizer_config = msg["_init_tokenizer_config"]
        _init_transformer_config = msg["_init_transformer_config"]
        if config_dict:
            # use following lines for loading raw spacy-transformers model which uses ElectraSudachipyTokenizer
            """
            try:
                from sudachitra import ElectraSudachipyTokenizer
                from transformers import AutoTokenizer
                AutoTokenizer.register(ElectraSudachipyTokenizer.__name__, slow_tokenizer_class=ElectraSudachipyTokenizer)
            except Exception:
                pass
            """
            tokenizer = self.tokenizer_cls.from_pretrained(config_dict["_name_or_path"], **_init_tokenizer_config)
            transformer = self.model_cls.from_pretrained(config_dict["_name_or_path"], **_init_transformer_config)
            transformer.to(get_torch_default_device())
            self._hfmodel = HFObjects(
                tokenizer,
                transformer,
                None,
                _init_tokenizer_config,
                _init_transformer_config,
            )
            self._model = transformer
        else:
            self._hfmodel = HFObjects(
                None,
                None,
                None,
                _init_tokenizer_config,
                _init_transformer_config,
            )
        return self
