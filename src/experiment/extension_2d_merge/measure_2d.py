"""Atom I5 evaluation harness.

Loads the 2D sparsity model (Jenga token sparsity + LongLoRA shifted
attention) with the retrained I5 LoRA adapter and measures forward loss
on a held out RedPajama slice under two modes:

  baseline_2d  config.merge_eliminated = False  (2D sparsity only)
  merged_2d    config.merge_eliminated = True   (2D + token merging)

Held out documents are drawn from indices start_index..start_index+N*8
in the dataset, matching the convention of measure_three_way.py.
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

from jenga.models.modeling_llama_2D import LlamaForCausalLM
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


def build_model(base_model, predictor_path, peft_model, seq_len, device):
    config = get_llama_qk(model_name=base_model, flash_attention=True, pool_size=64, thresh=0.4)
    config = set_RoPE(config, seq_len)
    pruned_cfg = torch.load(os.path.join(predictor_path, "pruned_config.pth"))
    config.predictor_layers = pruned_cfg["layers"]
    config.dynamic_threshold_lambda = 0.0
    config.log_adaptive = False
    config.merge_eliminated = False  # will toggle per mode

    tokenizer = AutoTokenizer.from_pretrained(base_model, model_max_length=seq_len, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = LlamaForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, config=config)
    attn_state_dict = torch.load(os.path.join(predictor_path, "predictor.pth"), map_location="cpu")
    msd = model.state_dict()
    for k, v in attn_state_dict.items():
        if k in msd:
            msd[k].copy_(v)
    model.resize_token_embeddings(32005)
    model = PeftModel.from_pretrained(model, peft_model)
    model = model.to(device).eval()
    inner = getattr(getattr(model, "base_model", model), "model", model)
    inner_cfg = getattr(inner, "config", model.config)
    return model, tokenizer, inner_cfg


def load_texts(seq_len, n_docs, start_index=1000):
    ds = load_dataset("./dataset/RedPajama-Data-1T-Sample", trust_remote_code=True)["train"]
    texts = []
    end = min(start_index + n_docs * 8, len(ds))
    for row in ds.select(range(start_index, end)):
        if len(texts) >= n_docs:
            break
        if len(row["text"]) > seq_len * 4:
            texts.append(row["text"])
    return texts


def run_mode(model, tokenizer, inner_cfg, texts, seq_len, mode, device):
    inner_cfg.merge_eliminated = (mode == "merged_2d")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    losses, durations = [], []
    for i, txt in enumerate(texts):
        ids = tokenizer(txt, return_tensors="pt", truncation=True, max_length=seq_len).input_ids.to(device)
        if ids.size(1) < seq_len:
            continue
        t0 = time.time()
        with torch.no_grad():
            out = model(input_ids=ids, labels=ids)
        dt = time.time() - t0
        loss = float(out.loss.detach().cpu())
        durations.append(dt)
        losses.append(loss)
        print(f"[{mode}] doc {i + 1}/{len(texts)} loss={loss:.4f} dt={dt:.2f}s", flush=True)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return {
        "mode": mode,
        "n_docs": len(losses),
        "mean_loss": sum(losses) / max(len(losses), 1),
        "ppl_approx": math.exp(sum(losses) / max(len(losses), 1)) if losses else float("nan"),
        "peak_memory_mb": float(peak_mb),
        "mean_forward_s": sum(durations) / max(len(durations), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="checkpoints/llama2")
    parser.add_argument("--peft_model", default="checkpoints/peft_model_2d_merge")
    parser.add_argument("--predictor_path", default="checkpoints/predictor")
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--n_docs", type=int, default=500)
    parser.add_argument("--out_csv", default="logs/extensions/token_merging_2d/comparison_500.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    texts = load_texts(args.seq_len, args.n_docs)
    print(f"using {len(texts)} long docs", flush=True)

    rows = []
    print("=== 2d adapter ===", flush=True)
    model, tokenizer, inner_cfg = build_model(args.base_model, args.predictor_path, args.peft_model, args.seq_len, device)
    r = run_mode(model, tokenizer, inner_cfg, texts, args.seq_len, "baseline_2d", device)
    r["label"] = "2D sparsity, no merging"
    rows.append(r)
    r = run_mode(model, tokenizer, inner_cfg, texts, args.seq_len, "merged_2d", device)
    r["label"] = "2D sparsity, token merging"
    rows.append(r)

    fieldnames = ["label", "mode", "n_docs", "mean_loss", "ppl_approx", "peak_memory_mb", "mean_forward_s"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    print(f"wrote {out_csv}", flush=True)
    for r in rows:
        print(f"{r['label']:42s} loss={r['mean_loss']:.4f} ppl={r['ppl_approx']:.3f} peak={r['peak_memory_mb']:.1f} dt={r['mean_forward_s']:.2f}", flush=True)


if __name__ == "__main__":
    main()
