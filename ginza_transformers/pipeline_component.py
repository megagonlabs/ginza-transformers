from pathlib import Path
import sys
from typing import List, Callable, Iterable, Union

from spacy.language import Language
from spacy.tokens import Doc
from thinc.api import Model, Config

from spacy_transformers.data_classes import FullTransformerBatch
from spacy_transformers.pipeline_component import Transformer, DOC_EXT_ATTR

from .layers.hf_shim_custom import override_hf_shims_from_bytes, recover_hf_shims_from_bytes


DEFAULT_CONFIG_STR = """
[transformer_custom]
max_batch_items = 4096

[transformer_custom.set_extra_annotations]
@annotation_setters = "spacy-transformers.null_annotation_setter.v1"

[transformer_custom.model]
@architectures = "ginza-transformers.TransformerModel.v3"

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

        origin = override_hf_shims_from_bytes()
        try:
            super().from_disk(path, exclude=exclude)
        finally:
            recover_hf_shims_from_bytes(origin)
        return self
