# Training Pipeline Notes (Gemma/QLoRA focus)

This note captures how we set up fine‑tuning with Hugging Face Transformers + PEFT. It covers the core APIs and what their inputs do, with links to the current docs.

## AutoModelForCausalLM.from_pretrained
- Purpose: load a causal LM checkpoint.
- Key args:
  - `pretrained_model_name_or_path`: HF model id or local path.
  - `dtype`: preferred load dtype (`"auto"`, `torch.float16`, `torch.bfloat16`, …). `dtype="auto"` keeps the saved precision. [Transformers v4.57](https://huggingface.co/docs/transformers/v4.57.2/models)
  - `device_map`: e.g., `"auto"` to shard across available devices/CPU. [Transformers v4.57](https://huggingface.co/docs/transformers/v4.57.2/models)
  - `quantization_config`: pass a `BitsAndBytesConfig` for 8‑bit/4‑bit loading (new pattern replacing legacy `load_in_8bit` kwarg). [Bitsandbytes](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)
  - `trust_remote_code`: required for models with custom code (e.g., Gemma). [Transformers v4.57](https://huggingface.co/docs/transformers/v4.57.2/models)

## BitsAndBytesConfig (Transformers quantization helper)
- Purpose: configure bitsandbytes int8/fp4/nf4 loading.
- Common fields (exclusive):
  - `load_in_8bit=True` *or* `load_in_4bit=True`.
  - `bnb_4bit_quant_type`: `"fp4"` or `"nf4"` (default `"fp4"`).
  - `bnb_4bit_compute_dtype`: compute dtype (bf16 recommended when available).
  - `llm_int8_threshold`, `llm_int8_enable_fp32_cpu_offload`: outlier handling/offload knobs.
- Passed as `quantization_config=BitsAndBytesConfig(...)` to `from_pretrained`. [Bitsandbytes](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes)

## prepare_model_for_kbit_training (PEFT)
- Purpose: ready an 8‑bit/4‑bit model for adapter training (QLoRA).
- Effects:
  - Casts LayerNorms to fp32 for stability.
  - Freezes base weights (`requires_grad=False`).
  - Keeps/upsamples `lm_head` in fp32 and enables its grads by default.
  - Optionally enables gradient checkpointing (via `use_gradient_checkpointing=True` and kwargs).
- Call immediately after loading a quantized model, before adding LoRA. [PeftModel](https://huggingface.co/docs/peft/en/package_reference/peft_model) and [Quantization](https://huggingface.co/docs/peft/en/developer_guides/quantization).

## LoraConfig (PEFT)
- Purpose: specify LoRA adapter hyperparameters.
- Common fields:
  - `r`: rank (adapter bottleneck).
  - `lora_alpha`: scaling factor (effective scale = `lora_alpha / r`).
  - `target_modules`: module names/patterns to wrap (e.g., `("q_proj", "v_proj")`, `"all-linear"`).
  - `lora_dropout`: dropout on the LoRA branch.
  - `bias`: `"none"`, `"lora_only"`, or `"all"` (bias training scope).
  - `task_type`: e.g., `"CAUSAL_LM"`, `"SEQ_2_SEQ_LM"`, `"SEQ_CLS"`. [LoRA](https://huggingface.co/docs/peft/en/package_reference/lora)

## get_peft_model
- Purpose: attach PEFT adapters (e.g., LoRA) to a base model.
- Usage: `model = get_peft_model(model, lora_config)`
- Result: base weights remain frozen; only adapter params (and any `modules_to_save` you specify) are trainable. [PeftModel](https://huggingface.co/docs/peft/en/package_reference/peft_model) and [LoRA](https://huggingface.co/docs/peft/en/package_reference/lora). 

## Typical QLoRA recipe (Gemma 3 example)
```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

quant_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-270m",
    quantization_config=quant_config,
    dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",  # Gemma 3 recommendation
)
model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=("q_proj", "v_proj"),
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
```

## DataCollatorForSeq2Seq
- Purpose: batch a list of tokenized examples, applying dynamic padding to inputs and labels.
- Key args we use:  
  - `tokenizer`: ensures pad token/side match the model.  
  - `padding=True`: pad to the longest in the batch.  
  - `pad_to_multiple_of=8`: pad lengths up to a multiple of 8 for Tensor Core efficiency.  
  - `return_tensors="pt"`: return PyTorch tensors.  
- Labels are padded with `label_pad_token_id=-100` by default so padded positions are ignored in the loss. [Data Collator](https://huggingface.co/docs/transformers/main_classes/data_collator)


## TrainingArguments (common fields we set)
[TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer)
- `output_dir`: where checkpoints are saved.
- `per_device_train_batch_size`, `per_device_eval_batch_size`: batch sizes per device.
- `gradient_accumulation_steps`: number of mini-batches to accumulate before **one optimizer step**; effective batch size scales by this. It controls how many mini‑batches you process before applying one optimizer step (i.e., one parameter update). “Accumulate” means: for each mini‑batch you run forward + backward, but you do not call `optimizer.step()`/`zero_grad()` yet—the gradients are summed (accumulated) in memory. After `gradient_accumulation_steps` such mini‑batches, you perform a single optimizer step, then zero the grads. Why it matters:
  - Lets you simulate a larger effective batch size without fitting that many samples at once in GPU memory.
  - Effective batch size = `per_device_train_batch_size × gradient_accumulation_steps × num_devices`.
  - Learning rate schedules and step‑based logging/eval count optimizer steps, not mini‑batches.
  - Example: `per_device_train_batch_size=8`, `gradient_accumulation_steps=4`, single GPU → 32 samples contribute to each optimizer step, even though you only load 8 at a time.
  - When using gradient accumulation, one step is counted as one step with backward pass. It means the “step counters” (for `logging_steps`, `eval_steps`, `save_steps`) tick on optimizer steps, not on individual mini‑batches. With accumulation, an optimizer step happens every `gradient_accumulation_steps` mini‑batches, so any interval you set is effectively multiplied by that factor in terms of raw batches/examples. Example:`gradient_accumulation_steps=4`, `logging_steps=10`, batch size 8. You run 4 mini‑batches → 1 optimizer step. Logs fire every 10 optimizer steps → every 10×4 = 40 mini‑batches, i.e., 40×8 = 320 examples.
- `num_train_epochs`: training passes over the dataset.
- `learning_rate`: base LR.
- `fp16`/`bf16`: mixed precision for trainable parts (works with int8 base + LoRA). Net effect: backbone weights sit in int8, forward/backward math runs in fp16 for the trainable LoRA layers, LayerNorms stay stable in fp32, and grad scaling keeps training numerically safe. This combo is the standard QLoRA recipe.
  - `BitsAndBytesConfig(load_in_8bit=True)`: base weights are stored in 8‑bit (int8); activations and gradients still use floating point (fp16/bf16/fp32). This drastically cuts VRAM for the frozen backbone.
  - `fp16=True` in `TrainingArguments`: enables mixed‑precision autocast + grad scaling for the trainable parts. LoRA weights (and any trainable heads) run/grad in fp16, which saves memory and is standard for QLoRA. It doesn’t “turn the 8‑bit weights into fp16”; it just sets the compute dtype for non‑int8 paths.
  - `prepare_model_for_kbit_training(model)`: massages the 8‑bit model for training—casts LayerNorms to fp32 for stability, freezes base params, can enable grad checkpointing, and keeps the int8 modules wired correctly.
  - LoRA: adapters are added in fp16 (or bf16 if you choose); only these small matrices get gradients and optimizer updates. The frozen int8 backbone stays frozen.
- `logging_steps`: log every N optimizer steps (note: with accumulation, one optimizer step spans multiple mini-batches).
- `evaluation_strategy`: `"steps"` to run eval periodically, `"no"` to skip.
- `eval_steps`: interval (in optimizer steps) for evaluation.
- `save_strategy`, `save_steps`, `save_total_limit`: checkpoint schedule and retention.
- `report_to`: e.g., `"none"` to disable external loggers. 


## Interactions / notes
- With 8‑bit base + LoRA, set `fp16=True` (or `bf16`) in `TrainingArguments`; base stays int8, adapters train in fp16/bf16.
- `use_cache` should be `False` during training, especially with gradient checkpointing.
- To train more layers, expand `target_modules` or use `"all-linear"`; base weights stay frozen unless you manually unfreeze.

## Managing OOM with long inputs
- Memory grows with sequence length. If you raise `model_max_length` (e.g., to ~1280) and hit OOM:
  - Lower `per_device_train_batch_size`; keep effective batch via higher `gradient_accumulation_steps`.
  - Enable `gradient_checkpointing=True` and often `gradient_checkpointing_kwargs={"use_reentrant": False}` for stability; keep `model.config.use_cache=False`.
  - Trim overly long prompts/inputs or cap outlier examples so responses still fit.
  - Keep 8‑bit base + LoRA with `fp16`/`bf16` to save memory.
