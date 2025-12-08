# Prompt Generation & Tokenization Pipeline

This note summarizes how prompts are built and tokenized before training, based on `medalpaca/handler.py` and `medalpaca/train.py`.

## Components
- **Tokenizer**: Hugging Face `AutoTokenizer`/`LlamaTokenizer` matching the base model. Padding is set to left; `pad_token_id` falls back to `eos_token_id` if missing.
- **Prompt template**: JSON file (e.g., `medalpaca/prompt_templates/medalpaca.json`) with keys:
  - `primer`, `instruction`, `input`, `output`, `response_split`.
- **DataHandler** (`medalpaca/handler.py`): wraps prompt construction and tokenization.

## Flow (per example)
1. **Generate prompt**  
   `generate_prompt(instruction, input, output)` concatenates template fields:  
   `primer + instruction + input + output`.
2. **Tokenize**  
   `tokenize(prompt, add_eos_token=True)` runs the tokenizer with:
   - `truncation=True`, `max_length=model_max_length`
   - no padding, no special tokens added by the tokenizer
   - If the last token is not EOS and length < `model_max_length` and `add_eos_token` is True, append `eos_token_id` and update `attention_mask`.
   - `labels` is a copy of `input_ids`.
3. **Optional masking (`train_on_inputs=False`)**  
   `generate_and_tokenize_prompt` rebuilds a user-only prompt (instruction + input), tokenizes it without EOS, measures its length, and sets the first `user_prompt_len` entries of `labels` to `-100`. This makes the loss ignore the instruction/input tokens (model trains only on the response tokens).

## How training uses it
- In `train.py`, a `DataHandler` is created once, then `generate_and_tokenize_prompt` is mapped over the JSON dataset (via `datasets.load_dataset`). The resulting DatasetDict feeds directly into Hugging Face `Trainer`.

## Key knobs
- `model_max_length` (default 256): truncation limit. Set it above your longest prompt (instruction + input + response + EOS). If it’s too small, the response can be truncated; if `train_on_inputs=False`, ensure it’s large enough that some response tokens remain (otherwise all labels become `-100` and the sample contributes no loss).
- `train_on_inputs`: 
  - `True` — loss covers instruction + input + response (labels copy all tokens).
  - `False` — instruction/input tokens are masked to `-100`; they condition the model but don’t contribute to loss.
- `add_eos_token` (default True in `generate_and_tokenize_prompt`): appends EOS if not present and there’s room.
- Template selection: swap `prompt_template` path to change surface form.
-    **We need to train the model to mask the instruction and inputs by setting `train_on_inputs=False` for the classification task. Meanwhile, keep the `model_max_length` greater than the instruction+input length to retain the unmasked response.**

## Practical snippets
```python
from medalpaca.handler import DataHandler
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m", use_fast=True, trust_remote_code=True)
if tokenizer.pad_token_id in (None, -1):
    tokenizer.pad_token_id = tokenizer.eos_token_id or 0
tokenizer.padding_side = "left"

dh = DataHandler(
    tokenizer=tokenizer,
    prompt_template="medalpaca/prompt_templates/medalpaca.json",
    model_max_length=256,
    train_on_inputs=True,
)

sample = {"instruction": "Predict stress level", "input": "Steps: 8000; HR: 65", "output": "3"}
tokenized = dh.generate_and_tokenize_prompt(sample)
```
