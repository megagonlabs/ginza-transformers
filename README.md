# ginza-transformers: Use custom tokenizers in spacy-transformers

The `ginza-transformers` is a simple extension of the [spacy-transformers](https://github.com/explosion/spacy-transformers) to use the custom tokenizers (defined outside of [huggingface/transformers](https://huggingface.co/transformers/)) in `transformer` pipeline component of [spaCy v3](https://spacy.io/usage/v3). The `ginza-transformers` also provides the ability to download the models from [Hugging Face Hub](https://huggingface.co/models) automatically at run time.

## Fallback mechanisms
There are two fallback tricks in `ginza-transformers`.

### Cutom tokenizer fallbacking
Loading a custom tokenizer specified in `components.transformer.model.tokenizer_config.tokenizer_class` attribute of `config.cfg` of a spaCy language model package, as follows.
- `ginza-transformers` initially tries to import a tokenizer class with the standard manner of `huggingface/transformers` (via `AutoTokenizer.from_pretrained()`)
- If a `ValueError` raised from `AutoTokenizer.from_pretrained()`, the fallback logic of `ginza-transformers` tries to import the class via `importlib.import_module` with the `tokenizer_class` value

### Model loading at run time
Downloading the model files published in Hugging Face Hub at run time, as follows.
- `ginza-transformers` initially tries to load local model directory (i.e. `/${local_spacy_model_dir}/transformer/model/`)
- If `OSError` raised, the first fallback logic passes a model name specified in `components.transformer.model.name` attribute of `config.cfg` to `AutoModel.from_pretrained()` with `local_files_only=True` option, which means the first fallback logic will immediately look in the local cache and will not reference the Hugging Face Hub at this point
- If `OSError` raised from the first fallback logic, the second fallback logic executes `AutoModel.from_pretrained()` without `local_files_only` option, which means the second fallback logic will search specified model name in the Hugging Face Hub

## How to use
Before executing `spacy train` command, make sure that [spaCy is working with cuda suppot](https://spacy.io/usage#gpu), and then install this package like:
```cosole
pip install -U ginza-transformers
```

You need to use `config.cfg` with a different setting when performing the analysis than the `spacy train`.

### Setting for training phase
[Here is an example](https://github.com/megagonlabs/ginza/blob/develop/config/ja_ginza_electra.cfg) of spaCy's `config.cfg` for training phase.
With this config, `ginza-transformers` employs [`SudachiTra`](https://github.com/WorksApplications/SudachiTra) as a transformer tokenizer and use [`megagonlabs/tansformers-ud-japanese-electra-base-discriminator`](https://huggingface.co/models/megagonlabs/tansformers-ud-japanese-electra-base-discriminator) as a pretrained transformer model.
The attributes of the training phase that differ from the defaults of spacy-transformers model are as follows:
```
[components.transformer.model]
@architectures = "ginza-transformers.TransformerModel.v1"
name = "megagonlabs/transformers-ud-japanese-electra-base-discriminator"

[components.transformer.model.tokenizer_config]
use_fast = false
tokenizer_class = "sudachitra.tokenization_electra_sudachipy.ElectraSudachipyTokenizer"
do_lower_case = false
do_word_tokenize = true
do_subword_tokenize = true
word_tokenizer_type = "sudachipy"
subword_tokenizer_type = "wordpiece"
word_form_type = "dictionary_and_surface"

[components.transformer.model.tokenizer_config.sudachipy_kwargs]
split_mode = "A"
dict_type = "core"
```

### Setting for analysis phases
[Here is an example](https://github.com/megagonlabs/ginza/blob/develop/config/ja_ginza_electra.analysis.cfg) of `config.cfg` for analysis phase.
This config refers [`megagonlabs/tansformers-ud-japanese-electra-base-ginza`](https://huggingface.co/models/megagonlabs/tansformers-ud-japanese-electra-base-ginza). The transformer model specified at `components.transformer.model.name` would be downloaded from the Hugging Face Hub at run time.
The attributes of the analysis phase that differ from the training phase are as follows:
```
[components.transformer]
factory = "transformer_custom"

[components.transformer.model]
name = "megagonlabs/transformers-ud-japanese-electra-base-ginza"
```
