from pathlib import Path
import sys
from typing import Union, Dict

from transformers import AutoConfig, AutoModel, AutoTokenizer

from spacy_transformers.layers import transformer_model
from spacy_transformers.data_classes import HFObjects

from thinc.api import get_current_ops, CupyOps


def override_huggingface_from_pretrained():
    assert transformer_model.huggingface_from_pretrained is not huggingface_from_pretrained_custom
    origin = transformer_model.huggingface_from_pretrained
    transformer_model.huggingface_from_pretrained = huggingface_from_pretrained_custom
    return origin

def recover_huggingface_from_pretrained(origin):
    assert transformer_model.huggingface_from_pretrained is huggingface_from_pretrained_custom
    transformer_model.huggingface_from_pretrained = origin


def huggingface_from_pretrained_custom(
    source: Union[Path, str], tok_config: Dict, trf_config: Dict
) -> HFObjects:
    """Create a Huggingface transformer model from pretrained weights. Will
    download the model if it is not already downloaded.

    source (Union[str, Path]): The name of the model or a path to it, such as
        'bert-base-cased'.
    tok_config (dict): Settings to pass to the tokenizer.
    trf_config (dict): Settings to pass to the transformer.
    """
    if hasattr(source, "absolute"):
        str_path = str(source.absolute())
    else:
        str_path = source

    try:
        tokenizer = AutoTokenizer.from_pretrained(str_path, **tok_config)
    except ValueError as e:
        if "tokenizer_class" not in tok_config:
            raise e
        tokenizer_class_name = tok_config["tokenizer_class"].split(".")
        from importlib import import_module
        tokenizer_module = import_module(".".join(tokenizer_class_name[:-1]))
        tokenizer_class = getattr(tokenizer_module, tokenizer_class_name[-1])
        tokenizer = tokenizer_class(vocab_file=str_path + "/vocab.txt", **tok_config)
    vocab_file_contents = None
    if hasattr(tokenizer, "vocab_file"):
        with open(tokenizer.vocab_file, "rb") as fileh:
            vocab_file_contents = fileh.read()

    try:
        trf_config["return_dict"] = True
        config = AutoConfig.from_pretrained(str_path, **trf_config)
        transformer = AutoModel.from_pretrained(model_name, local_files_only=True)
    except OSError as e1:
        try:
            transformer = AutoModel.from_pretrained(str_path, config=config)
        except OSError as e2:
            model_name = str(source)
            print("trying to download model from huggingface hub:", model_name, "...", file=sys.stderr)
            transformer = AutoModel.from_pretrained(model_name)
            print("succeded", file=sys.stderr)
    ops = get_current_ops()
    if isinstance(ops, CupyOps):
        transformer.cuda()
    return HFObjects(tokenizer, transformer, vocab_file_contents)
