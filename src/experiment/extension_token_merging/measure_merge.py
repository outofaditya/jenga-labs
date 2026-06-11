#!/usr/bin/env python
"""
Extension C (I3) measurement: compare baseline Jenga (hard drop) against
Jenga + Token Merging (soft elimination) on the same Llama 2 7B + Jenga
adapter setup. Run a brief forward pass over RedPajama documents in each
mode, capture peak memory and mean next token loss, write to CSV.

Inference only (no fine tuning). Single seed, batch size 1, sequence
length 8192. Token merging is gated by config.merge_eliminated.
"""
import argparse
import csv
import math
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoTokenizer

from jenga.models.modeling_llama import LlamaForCausalLM
from jenga.utils.config_utils import get_llama_qk


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
    parser.add_argument("--out_csv", default="logs/extensions/token_merging/comparison.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    config = get_llama_qk(
        model_name=args.base_model,
        flash_attention=True,
        pool_size=64,
        thresh=0.4,
    )
    config = set_RoPE(config, args.seq_len)
    pruned_cfg = torch.load(os.path.join(args.predictor_path, "pruned_config.pth"))
    config.predictor_layers = pruned_cfg["layers"]
    config.merge_eliminated = False

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
    model.resize_token_embeddings(32005)

    if args.peft_model and args.peft_model != "none":
        print(f"loading peft adapter {args.peft_model}...", flush=True)
        model = PeftModel.from_pretrained(model, args.peft_model)
    model = model.to(device).eval()

    inner = getattr(getattr(model, "base_model", model), "model", model)
    inner_cfg = getattr(inner, "config", model.config)

    print("loading RedPajama...", flush=True)
    ds = load_dataset("./dataset/RedPajama-Data-1T-Sample", trust_remote_code=True)["train"]
    texts = []
    for row in ds.select(range(min(args.n_docs * 8, len(ds)))):
        if len(texts) >= args.n_docs:
            break
        if len(row["text"]) > args.seq_len * 4:
            texts.append(row["text"])
    print(f"using {len(texts)} long docs", flush=True)

    rows = []
    for mode in ["baseline", "merged"]:
        inner_cfg.merge_eliminated = (mode == "merged")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        losses = []
        durations = []
        for doc_idx, txt in enumerate(texts):
            ids = tokenizer(txt, return_tensors="pt", truncation=True, max_length=args.seq_len).input_ids.to(device)
            if ids.size(1) < args.seq_len:
                continue
            t0 = time.time()
            with torch.no_grad():
                out = model(input_ids=ids, labels=ids)
            dt = time.time() - t0
            loss = float(out.loss.detach().cpu())
            durations.append(dt)
            losses.append(loss)
            print(f"[{mode}] doc {doc_idx + 1}/{len(texts)} loss={loss:.4f} dt={dt:.2f}s", flush=True)
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        rows.append({
            "mode": mode,
            "n_docs": len(losses),
            "mean_loss": sum(losses) / max(len(losses), 1),
            "ppl_approx": math.exp(sum(losses) / max(len(losses), 1)) if losses else float("nan"),
            "peak_memory_mb": float(peak_mb),
            "mean_forward_s": sum(durations) / max(len(durations), 1),
        })
        print(f"[{mode}] mean_loss={rows[-1]['mean_loss']:.4f} ppl_approx={rows[-1]['ppl_approx']:.3f} peak_mb={rows[-1]['peak_memory_mb']:.1f} mean_s={rows[-1]['mean_forward_s']:.2f}", flush=True)

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out_csv}", flush=True)


if __name__ == "__main__":
    main()
