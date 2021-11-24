import sys
from typing import Any
from io import BytesIO
from pathlib import Path
import srsly
import torch
from spacy.util import SimpleFrozenDict
from spacy.vectors import get_current_ops

from spacy_transformers.layers import hf_shim
from spacy_transformers.layers.hf_shim import HFShim
from spacy_transformers.data_classes import HFObjects
from spacy_transformers.util import make_tempdir

from transformers import AutoModel, AutoConfig, AutoTokenizer


def override_hf_shims_from_bytes():
    assert hf_shim.HFShim.from_bytes is not HFShimCustom.from_bytes
    origin = hf_shim.HFShim.from_bytes
    hf_shim.HFShim.from_bytes = HFShimCustom.from_bytes
    return origin

def recover_hf_shims_from_bytes(origin):
    assert hf_shim.HFShim.from_bytes is HFShimCustom.from_bytes
    hf_shim.HFShim.from_bytes = origin


class HFShimCustom(HFShim):

    def from_bytes(self, bytes_data):
        msg = srsly.msgpack_loads(bytes_data)
        config_dict = msg["config"]
        tok_dict = msg["tokenizer"]
        if config_dict:
            with make_tempdir() as temp_dir:
                config_file = temp_dir / "config.json"
                srsly.write_json(config_file, config_dict)
                config = AutoConfig.from_pretrained(config_file)
                for x, x_bytes in tok_dict.items():
                    Path(temp_dir / x).write_bytes(x_bytes)
                tokenizer = None
                try:
                    tokenizer = AutoTokenizer.from_pretrained(str(temp_dir.absolute()))
                except (ValueError, OSError):
                    pass
                if tokenizer is None:
                    tok_config = srsly.read_json(str((temp_dir / "tokenizer_config.json").absolute()))
                    tokenizer_class_name = tok_config["tokenizer_class"].split(".")
                    if tokenizer_class_name == ["ElectraSudachipyTokenizer"]:
                        from sudachitra.tokenization_electra_sudachipy import ElectraSudachipyTokenizer as tokenizer_class
                        tokenizer = tokenizer_class(vocab_file=str((temp_dir / "vocab.txt").absolute()), **tok_config)
                    else:
                        from importlib import import_module
                        tokenizer_module = import_module(".".join(tokenizer_class_name[:-1]))
                        tokenizer_class = getattr(tokenizer_module, tokenizer_class_name[-1])

                vocab_file_contents = None
                if hasattr(tokenizer, "vocab_file"):
                    vocab_file_name = tokenizer.vocab_files_names["vocab_file"]
                    vocab_file_path = str((temp_dir / vocab_file_name).absolute())
                    with open(vocab_file_path, "rb") as fileh:
                        vocab_file_contents = fileh.read()

            try:
                transformer = AutoModel.from_config(config)
            except OSError as e:
                try:
                    transformer = AutoModel.from_pretrained(config["_name_or_path"], local_files_only=True)
                except OSError as e2:
                    print("trying to download model from huggingface hub:", config["_name_or_path"], "...", file=sys.stderr)
                    transformer = AutoModel.from_pretrained(config["_name_or_path"])
                    print("succeded", file=sys.stderr)

            self._hfmodel = HFObjects(
                tokenizer,
                transformer,
                vocab_file_contents,
                SimpleFrozenDict(),
                SimpleFrozenDict(),
            )
            self._model = transformer
            filelike = BytesIO(msg["state"])
            filelike.seek(0)
            ops = get_current_ops()
            if ops.device_type == "cpu":
                map_location = "cpu"
            else:  # pragma: no cover
                device_id = torch.cuda.current_device()
                map_location = f"cuda:{device_id}"
            self._model.load_state_dict(torch.load(filelike, map_location=map_location))
            self._model.to(map_location)
        else:
            self._hfmodel = HFObjects(
                None,
                None,
                None,
                msg["_init_tokenizer_config"],
                msg["_init_transformer_config"],
            )
        return self
