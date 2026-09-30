import os

from transformers import AutoModel

from spacy_transformers.layers import transformer_model


def override_huggingface_from_pretrained():
    assert transformer_model.huggingface_from_pretrained is not huggingface_from_pretrained_custom
    origin = transformer_model.huggingface_from_pretrained
    transformer_model.huggingface_from_pretrained = huggingface_from_pretrained_custom
    return origin

def recover_huggingface_from_pretrained(origin):
    assert transformer_model.huggingface_from_pretrained is huggingface_from_pretrained_custom
    transformer_model.huggingface_from_pretrained = origin


def huggingface_from_pretrained_custom(
    huggingface_from_pretrained_origin = None,
    **kwargs,
):
    kwargs["model_cls"] = AutoModelCustom
    return huggingface_from_pretrained_origin(**kwargs)


class AutoModelCustom(AutoModel):
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | os.PathLike | None,
        *model_args,
        config = None,
        **kwargs,
    ):
        return super().from_pretrained(config["_name_or_path"], *model_args, config=config, **kwargs)
