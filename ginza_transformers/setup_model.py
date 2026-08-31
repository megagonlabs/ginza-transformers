import json
import sys

import spacy

from .layers.hf_shim_custom import override_hf_shims_to_bytes, recover_hf_shims_to_bytes


def main():
    org_spacy_model_path = sys.argv[1]
    dst_spacy_model_path = sys.argv[2]
    transformers_model_name = sys.argv[3]
    revision = sys.argv[4] if len(sys.argv) > 4 else None
    # load pipeline and save a transformers model independently
    try:
        from sudachitra import ElectraSudachipyTokenizer
        from transformers import AutoTokenizer
        AutoTokenizer.register(ElectraSudachipyTokenizer.__name__, slow_tokenizer_class=ElectraSudachipyTokenizer)
    except Exception:
        pass
    nlp = spacy.load(org_spacy_model_path)
    transformer = nlp.get_pipe("transformer")
    for node in transformer.model.walk():
        if node.shims:
            break
    else:
        assert False
    node.shims[0]._hfmodel.transformer.config._name_or_path = transformers_model_name
    node.shims[0]._hfmodel.transformer.save_pretrained(transformers_model_name)
    # save tokenizer with AutoTokenizer feature
    node.shims[0]._hfmodel.tokenizer.save_pretrained(transformers_model_name)
    tokenizer_class_name = node.shims[0]._hfmodel.tokenizer.__class__.__name__
    init_tokenizer_config = node.shims[0]._hfmodel._init_tokenizer_config or {}
    if tokenizer_class_name == "ElectraSudachipyTokenizer":
        with open(f"{transformers_model_name}/tokenizer_config.json", "r", encoding="utf8") as fin:
            tokenizer_config = json.load(fin)
        tokenizer_config["tokenizer_class"] = tokenizer_class_name
        tokenizer_config["auto_map"] = {
            "AutoTokenizer": [
                f"modeling.{tokenizer_class_name}",
                None,
            ]
        }
        with open(f"{transformers_model_name}/tokenizer_config.json", "w", encoding="utf8") as fout:
            json.dump(tokenizer_config, fout, indent=1)
            print(file=fout)
        with open(f"{transformers_model_name}/modeling.py", "w", encoding="utf8") as fout:
            print(f"from sudachitra import {tokenizer_class_name}", file=fout)
        # set trust_remote_code = True for spacy-transformers
        init_tokenizer_config["trust_remote_code"] = True
        if revision:
            init_tokenizer_config["revision"] = revision
    node.shims[0]._hfmodel._init_tokenizer_config = init_tokenizer_config
    # save a light-weight pipeline which can be loaded with ginza-transformers
    origin = override_hf_shims_to_bytes()
    try:
        nlp.to_disk(dst_spacy_model_path)
    finally:
        recover_hf_shims_to_bytes(origin)


if __name__ == "__main__":
    main()
