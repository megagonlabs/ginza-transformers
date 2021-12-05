import sys

import spacy

from .layers.hf_shim_custom import override_hf_shims_to_bytes, recover_hf_shims_to_bytes


def main():
    org_spacy_model_path = sys.argv[1]
    dst_spacy_model_path = sys.argv[2]
    transformers_model_name = sys.argv[3]

    nlp = spacy.load(org_spacy_model_path)
    transformer = nlp.get_pipe("transformer")
    for i, node in enumerate(transformer.model.walk()):
        if node.shims:
            break
    else:
        assert False
    node.shims[0]._hfmodel.transformer.config._name_or_path = transformers_model_name
    node.shims[0]._hfmodel.tokenizer.save_pretrained(transformers_model_name)
    node.shims[0]._hfmodel.transformer.save_pretrained(transformers_model_name)
    override_hf_shims_to_bytes()
    try:
        origin = nlp.to_disk(dst_spacy_model_path)
    finally:
        recover_hf_shims_to_bytes(origin)


if __name__ == "__main__":
    main()
