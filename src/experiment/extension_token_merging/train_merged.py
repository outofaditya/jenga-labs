"""
Token Merging training driver. Fine tune the LoRA adapter on top of Llama 2 7B with
merge_eliminated enabled from step 0 so the merged token participates in
training. Mirrors the end to end time experiment driver but writes the
adapter to checkpoints/peft_model_merged so the existing baseline adapter
is preserved.
"""

import math
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import torch
import transformers
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, Trainer

from jenga.models.modeling_llama import LlamaForCausalLM
from jenga.utils.config_utils import get_llama_qk
from jenga.utils.others import seed_everything, smart_tokenizer_and_embedding_resize

BEGIN_TOKEN, END_TOKEN = "<<BEGIN>>", "<<END>>"
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="checkpoints/llama2")
    pack_loss: bool = field(default=False)
    predictor_path: str = field(default="checkpoints/predictor")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    flash_attention: bool = field(default=True)
    pool_size: int = field(default=64)
    thresh: float = field(default=0.4)
    model_max_length: int = field(default=8192)
    gradient_checkpoint: bool = field(default=True)


def set_RoPE(config, model_max_length):
    orig_rope_scaling = getattr(config, "rope_scaling", None) or {"factor": 1}
    f = orig_rope_scaling.get("factor", 1)
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len:
        effective = orig_ctx_len * f
        if model_max_length > effective:
            scaling = float(math.ceil(model_max_length / effective))
            config.rope_scaling = {"type": "linear", "factor": scaling}
    return config


def tokenize_fn(tokenizer, example):
    context_length = tokenizer.model_max_length
    outputs = tokenizer(
        tokenizer.eos_token.join(example["text"]),
        truncation=False,
        return_tensors="pt",
        pad_to_multiple_of=context_length,
        padding=True,
    )
    return {"input_ids": outputs["input_ids"].view(-1, context_length)}


def train():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    seed_everything(42)

    config = get_llama_qk(
        model_name=model_args.model_name_or_path,
        flash_attention=training_args.flash_attention,
        pool_size=training_args.pool_size,
        thresh=training_args.thresh,
    )
    config = set_RoPE(config, training_args.model_max_length)
    # soft elimination merge must be active from step 0 so the adapter trains against it
    config.merge_eliminated = True
    config.time = False

    pruned_cfg = torch.load(
        os.path.join(model_args.predictor_path, "pruned_config.pth")
    )
    config.predictor_layers = pruned_cfg["layers"]

    model_dtype = torch.bfloat16 if training_args.bf16 else torch.float32
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=model_dtype,
        config=config,
    )
    attn_state_dict = torch.load(
        os.path.join(model_args.predictor_path, "predictor.pth"), map_location="cpu"
    )
    msd = model.state_dict()
    for k, v in attn_state_dict.items():
        if k in msd:
            msd[k].copy_(v)
        else:
            print("skip", k, flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        padding_size="right",
        use_fast=True,
    )
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    if tokenizer.cls_token is None:
        special_tokens_dict["cls_token"] = BEGIN_TOKEN
    if tokenizer.sep_token is None:
        special_tokens_dict["sep_token"] = END_TOKEN
    special_tokens_dict["additional_special_tokens"] = ["[INST]", "[/INST]"]
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    dataset = load_dataset(
        "./dataset/RedPajama-Data-1T-Sample",
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    train_dataset = dataset["train"].select(range(1000))
    small_dataset = DatasetDict({"train": train_dataset})
    small_dataset = small_dataset.map(
        partial(tokenize_fn, tokenizer),
        batched=True,
        num_proc=4,
        remove_columns=["text", "meta"],
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    if training_args.gradient_checkpoint:
        model.config.use_cache = False
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=small_dataset["train"],
        eval_dataset=None,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model()
    trainer.save_state()


if __name__ == "__main__":
    train()
