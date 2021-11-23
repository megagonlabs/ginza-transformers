from .layers import TransformerModelCustom
from spacy_transformers.util import registry


registry.architectures.register(
    "ginza-transformers.TransformerModel.v3", func=TransformerModelCustom
)
