#!/usr/bin/env python
"""
Extension A (I1) measurement: sweep dynamic_threshold_lambda over Jenga
forwards on RedPajama documents and capture per layer (entropy_norm,
retention_ratio) tuples. Output a CSV that the plot script renders as the
scatter figure for REPORT 6.1a.

Runs inference only (no fine tuning). PPL comparison (Figure 6.1b in the
original plan) is dropped because the artifact's PPL harness uses the dense
baseline model and would need a new Jenga aware eval driver to surface the
adaptive threshold effect on perplexity.
"""
import argparse
import csv
import math
import os
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoTokenizer

import jenga.models.modeling_llama as jmm
from jenga.models.modeling_llama import LlamaForCausalLM
from jenga.utils.config_utils import get_llama_qk
from jenga.utils.others import smart_tokenizer_and_embedding_resize


BEGIN_TOKEN, END_TOKEN = "<<BEGIN>>", "<<END>>"
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="checkpoints/llama2")
    parser.add_argument("--peft_model", default="checkpoints/peft_model/rp/8k/jenga")
    parser.add_argument("--predictor_path", default="checkpoints/predictor")
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--n_docs", type=int, default=4)
    parser.add_argument("--pool_size", type=int, default=64)
    parser.add_argument("--thresh", type=float, default=0.4)
    parser.add_argument("--lams", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2])
    parser.add_argument("--out_csv", default="logs/extensions/adaptive_thresholds/retention.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    config = get_llama_qk(
        model_name=args.base_model,
        flash_attention=True,
        pool_size=args.pool_size,
        thresh=args.thresh,
    )
    config = set_RoPE(config, args.seq_len)
    pruned_cfg = torch.load(os.path.join(args.predictor_path, "pruned_config.pth"))
    config.predictor_layers = pruned_cfg["layers"]
    config.log_adaptive = True
    config.dynamic_threshold_lambda = 0.0

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, model_max_length=args.seq_len, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("loading base model...", flush=True)
    model = LlamaForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, config=config
    )
    attn_state_dict = torch.load(os.path.join(args.predictor_path, "predictor.pth"), map_location="cpu")
    msd = model.state_dict()
    for k, v in attn_state_dict.items():
        if k in msd:
            msd[k].copy_(v)

    # The Jenga LoRA adapters were saved with vocab 32005. Force the embedding
    # matrix to the same size directly so PEFT load matches regardless of which
    # special tokens the tokenizer already had.
    model.resize_token_embeddings(32005)

    if args.peft_model and args.peft_model != "none":
        print(f"loading peft adapter {args.peft_model}...", flush=True)
        model = PeftModel.from_pretrained(model, args.peft_model)
    model = model.to(device).eval()

    print("loading RedPajama...", flush=True)
    ds = load_dataset("./dataset/RedPajama-Data-1T-Sample", trust_remote_code=True)["train"]
    texts = []
    for row in ds.select(range(min(args.n_docs * 8, len(ds)))):
        if len(texts) >= args.n_docs:
            break
        if len(row["text"]) > args.seq_len * 4:
            texts.append(row["text"])
    print(f"using {len(texts)} long docs", flush=True)

    # Aim for the underlying jenga model object whose attention modules read
    # config.dynamic_threshold_lambda. With PEFT the real model is at
    # model.base_model.model; without PEFT it is just model.
    inner = getattr(getattr(model, "base_model", model), "model", model)
    inner_cfg = getattr(inner, "config", model.config)

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lam", "doc_idx", "layer", "entropy_norm", "retention", "q_len_orig"])
        for lam in args.lams:
            print(f"sweeping lam={lam}", flush=True)
            inner_cfg.dynamic_threshold_lambda = float(lam)
            for doc_idx, txt in enumerate(texts):
                ids = tokenizer(txt, return_tensors="pt", truncation=True, max_length=args.seq_len).input_ids.to(device)
                if ids.size(1) < args.seq_len:
                    continue
                jmm._ADAPTIVE_STATS = []
                with torch.no_grad():
                    _ = model(ids)
                stats = list(jmm._ADAPTIVE_STATS)
                for s in stats:
                    w.writerow([lam, doc_idx, s["layer"], s["entropy_norm"], s["retention"], s["q_len_orig"]])
                f.flush()
                print(f"  lam={lam} doc {doc_idx + 1}/{len(texts)} captured {len(stats)} layer rows", flush=True)

    print(f"wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
