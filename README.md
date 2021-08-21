# ginza-transformers: Use custom tokenizers in spacy-transformers

The `ginza-transformers` is a simple extension of the [spacy-transformers](https://github.com/explosion/spacy-transformers) to use the custom tokenizers (defined outside of [huggingface/transformers](https://huggingface.co/transformers/)) in `transformer` pipeline component of [spaCy v3](https://spacy.io/usage/v3). The `ginza-transformers` also provides the ability to download the models from [Hugging Face Hub](https://huggingface.co/models) automatically at run time.

## Fall-back mechanisms
There are two fall-back tricks in `ginza-transformers`.
- Loading a custom tokenizer specified in `components.transformer.model.tokenizer_config.tokenizer_class` attribute of `config.cfg` of a spaCy language model package
  - `ginza-transformers` initially tries to import a tokenizer class with the standard manner of `huggingface/transformers` (via `AutoTokenizer.from_pretrained()`)
  - If a `ValueError` raised from `AutoTokenizer.from_pretrained()`, the fall-back logic of `ginza-transformers` tries to import the class via `importlib.import_module` with the `tokenizer_class` value
- Downloading the model files published in Hugging Face Hub at run time
  - `ginza-transformers` initially tries to load local model directory (i.e. `/${local_spacy_model_dir}/transformer/model/`)
  - If `OSError` raised, the fall-back logic passes a model name specified in `components.transformer.model.name` attribute of `config.cfg` to `AutoModel.from_pretrained()`
