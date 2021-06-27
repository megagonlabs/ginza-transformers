from typing import Dict, Union
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from thinc.api import get_current_ops, CupyOps


def huggingface_from_pretrained_custom(source: Union[Path, str], config: Dict):
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
        tokenizer = AutoTokenizer.from_pretrained(str_path, **config)
    except ValueError as e:
        if "tokenizer_class" not in config:
            raise e
        tokenizer_class_name = config["tokenizer_class"].split(".")
        from importlib import import_module
        tokenizer_module = import_module(".".join(tokenizer_class_name[:-1]))
        tokenizer_class = getattr(tokenizer_module, tokenizer_class_name[-1])
        tokenizer = tokenizer_class(vocab_file=str_path + "/vocab.txt", **config)

    transformer = AutoModel.from_pretrained(str_path)
    ops = get_current_ops()
    if isinstance(ops, CupyOps):
        transformer.cuda()
    return tokenizer, transformer
