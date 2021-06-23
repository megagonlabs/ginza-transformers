from spacy_transformers.layers.transformer_model import *

from ginza_transformers.util import huggingface_from_pretrained_custom


def TransformerModelCustom(
    name: str, get_spans: Callable, tokenizer_config: dict
) -> Model[List[Doc], FullTransformerBatch]:
    return Model(
        "transformer",
        forward,
        init=init_custom,
        layers=[],
        dims={"nO": None},
        attrs={
            "tokenizer": None,
            "get_spans": get_spans,
            "name": name,
            "tokenizer_config": tokenizer_config,
            "set_transformer": set_pytorch_transformer,
            "has_transformer": False,
            "flush_cache_chance": 0.0,
        },
    )


def init_custom(model: Model, X=None, Y=None):
    if model.attrs["has_transformer"]:
        return
    name = model.attrs["name"]
    tok_cfg = model.attrs["tokenizer_config"]
    tokenizer, transformer = huggingface_from_pretrained_custom(name, tok_cfg)
    model.attrs["tokenizer"] = tokenizer
    model.attrs["set_transformer"](model, transformer)
    # Call the model with a batch of inputs to infer the width
    if X:
        # If we're dealing with actual texts, do the work to setup the wordpieces
        # batch properly
        docs = X
        get_spans = model.attrs["get_spans"]
        nested_spans = get_spans(docs)
        flat_spans = []
        for doc_spans in nested_spans:
            flat_spans.extend(doc_spans)
        token_data = huggingface_tokenize(
            model.attrs["tokenizer"],
            [span.text for span in flat_spans]
        )
        wordpieces = WordpieceBatch.from_batch_encoding(token_data)
        align = get_alignment(
            flat_spans,
            wordpieces.strings, model.attrs["tokenizer"].all_special_tokens
        )
        wordpieces, align = truncate_oversize_splits(
            wordpieces, align, tokenizer.model_max_length
        )
    else:
        texts = ["hello world", "foo bar"]
        token_data = huggingface_tokenize(
            model.attrs["tokenizer"],
            texts
        )
        wordpieces = WordpieceBatch.from_batch_encoding(token_data)
    model.layers[0].initialize(X=wordpieces)
    tensors = model.layers[0].predict(wordpieces)
    t_i = find_last_hidden(tensors)
    model.set_dim("nO", tensors[t_i].shape[-1])
