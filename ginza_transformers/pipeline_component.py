from pathlib import Path
import sys
from typing import List, Callable, Iterable, Union

from spacy.language import Language
from spacy.pipeline.pipe import deserialize_config
from spacy.tokens import Doc
from spacy import util, Errors
from thinc.api import Model, Config

from spacy_transformers.data_classes import FullTransformerBatch
from spacy_transformers.pipeline_component import Transformer, DOC_EXT_ATTR

from .layers.transformer_model import huggingface_from_pretrained_custom


DEFAULT_CONFIG_STR = """
[transformer_custom]
max_batch_items = 4096

[transformer_custom.set_extra_annotations]
@annotation_setters = "spacy-transformers.null_annotation_setter.v1"

[transformer_custom.model]
@architectures = "ginza-transformers.TransformerModel.v1"

[transformer_custom.model.get_spans]
@span_getters = "spacy-transformers.strided_spans.v1"
window = 128
stride = 96
"""

DEFAULT_CONFIG = Config().from_str(DEFAULT_CONFIG_STR)


@Language.factory(
    "transformer_custom",
    assigns=[f"doc._.{DOC_EXT_ATTR}"],
    default_config=DEFAULT_CONFIG["transformer_custom"],
)
def make_transformer_custom(
    nlp: Language,
    name: str,
    model: Model[List[Doc], FullTransformerBatch],
    set_extra_annotations: Callable[[List[Doc], FullTransformerBatch], None],
    max_batch_items: int,
):
    return TransformerCustom(
        nlp.vocab,
        model,
        set_extra_annotations,
        max_batch_items=max_batch_items,
        name=name,
    )


class TransformerCustom(Transformer):

    def from_disk(
        self, path: Union[str, Path], *, exclude: Iterable[str] = tuple()
    ) -> "TransformerCustom":

        def load_model(p):
            try:
                with open(p, "rb") as mfile:
                    self.model.from_bytes(mfile.read())
            except AttributeError:
                raise ValueError(Errors.E149) from None
            except (IsADirectoryError, PermissionError):
                p = Path(p).absolute()
                tokenizer, hf_model = huggingface_from_pretrained_custom(
                    p,
                    self.model._init_tokenizer_config,
                    self.model._init_transformer_config,
                    self.model.attrs["name"],
                )
                self.model.attrs["tokenizer"] = tokenizer
                self.model.attrs["set_transformer"](self.model, hf_model)

        deserialize = {
            "vocab": self.vocab.from_disk,
            "cfg": lambda p: self.cfg.update(deserialize_config(p)),
            "model": load_model,
        }
        util.from_disk(path, deserialize, exclude)
        return self
