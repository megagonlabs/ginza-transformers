from typing import Dict, Union, Optional
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from thinc.api import get_current_ops, CupyOps


def huggingface_from_pretrained_custom(source: Union[Path, str], tokenizer_config: Dict, model_name: Optional[str] = None):
    """Create a Huggingface transformer model from pretrained weights. Will
    download the model if it is not already downloaded.

    source (Union[str, Path]): The name of the model or a path to it, such as
        'bert-base-cased'.
    config (dict): Settings to pass to the tokenizer.
    """
    if hasattr(source, "absolute"):
        str_path = str(source.absolute())
    else:
        str_path = source

    try:
        tokenizer = AutoTokenizer.from_pretrained(str_path, **tokenizer_config)
    except ValueError as e:
        if "tokenizer_class" not in tokenizer_config:
            raise e
        tokenizer_class_name = tokenizer_config["tokenizer_class"].split(".")
        from importlib import import_module
        tokenizer_module = import_module(".".join(tokenizer_class_name[:-1]))
        tokenizer_class = getattr(tokenizer_module, tokenizer_class_name[-1])
        tokenizer = tokenizer_class(vocab_file=str_path + "/vocab.txt", **tokenizer_config)

    try:
        transformer = AutoModel.from_pretrained(str_path)
    except OSError as e:
        try:
            transformer = AutoModel.from_pretrained(model_name, local_files_only=True)
        except OSError as e2:
            transformer = AutoModel.from_pretrained(model_name)
    ops = get_current_ops()
    if isinstance(ops, CupyOps):
        transformer.cuda()
    return tokenizer, transformer
