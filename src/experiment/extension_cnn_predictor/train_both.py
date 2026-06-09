#!/usr/bin/env python
"""
Extension B (I2): head to head MLP vs CNN attention predictor.

The script does two phases:

1. **Cache phase** (one time per pod): run a frozen Llama 2 7B forward over a
   small RedPajama subset with attention outputs enabled. For every transformer
   layer, capture (hidden_state_input, pooled_attention_score) tuples. Save to
   a single torch tensor file in logs/extensions/cnn_predictor/cache/.

2. **Train phase** (runs twice, once per predictor type): instantiate the
   predictor (`AttnPredictor1` for mlp, `CNNAttnPredictor` for cnn), train with
   AdamW against the cached attention scores for the requested number of
   epochs, log per epoch MSE to a CSV.

Output CSV schema: `epoch,seed,predictor_type,train_loss`. Plot script reads
this and emits the dual line loss curve PDF.
"""
import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

from jenga.models.predictor import AttnPredictor1, CNNAttnPredictor

POOL_SIZE = 64
HIDDEN_DIM = 128


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cache_path(out_dir: str, seq_len: int, n_docs: int) -> Path:
    return Path(out_dir) / f"cache_seqlen{seq_len}_n{n_docs}.pt"


def build_cache(model_path: str, seq_len: int, n_docs: int, out_dir: str, device: str):
    """Run Llama 2 forward with attention output, collect (hidden_in, attn_score) per layer."""
    from transformers import AutoModelForCausalLM, AutoConfig

    cache_file = cache_path(out_dir, seq_len, n_docs)
    if cache_file.exists():
        print(f"[cache] reusing existing cache at {cache_file}", flush=True)
        return cache_file

    print(f"[cache] building cache: model={model_path} seq_len={seq_len} n_docs={n_docs}", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=seq_len, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(model_path)
    config.output_attentions = True
    config.attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, config=config
    ).to(device).eval()
    num_layers = config.num_hidden_layers
    target_layers = list(range(0, num_layers, 2))
    print(f"[cache] target layers (every other of {num_layers}): {target_layers}", flush=True)

    ds = load_dataset("./dataset/RedPajama-Data-1T-Sample", trust_remote_code=True)["train"]
    head = ds.select(range(min(n_docs * 4, len(ds))))
    texts = []
    for row in head:
        if len(texts) >= n_docs:
            break
        t = row["text"]
        if len(t) > seq_len * 4:
            texts.append(t)
    print(f"[cache] collected {len(texts)} long documents", flush=True)

    samples_h = []
    samples_a = []
    for i, txt in enumerate(texts):
        ids = tokenizer(txt, return_tensors="pt", truncation=True, max_length=seq_len).input_ids.to(device)
        if ids.size(1) < seq_len:
            continue
        with torch.no_grad():
            out = model(ids, output_hidden_states=True, output_attentions=True, return_dict=True)
        hs = out.hidden_states
        attns = out.attentions
        for layer_idx in target_layers:
            h_in = hs[layer_idx]
            attn = attns[layer_idx]
            num_blocks = seq_len // POOL_SIZE
            attn_blocked = attn.view(
                attn.size(0), attn.size(1), num_blocks, POOL_SIZE, num_blocks, POOL_SIZE
            ).sum(dim=(3, 5))
            attn_target = attn_blocked.sum(dim=1)
            samples_h.append(h_in[:, : num_blocks * POOL_SIZE, :].detach().cpu().float())
            samples_a.append(attn_target.detach().cpu().float())
        print(f"[cache] doc {i + 1}/{len(texts)} processed", flush=True)
    H = torch.cat(samples_h, dim=0)
    A = torch.cat(samples_a, dim=0)
    print(f"[cache] saving H={tuple(H.shape)} A={tuple(A.shape)} -> {cache_file}", flush=True)
    torch.save({"hidden": H, "attn": A}, cache_file)
    del model
    torch.cuda.empty_cache()
    return cache_file


def train_predictor(predictor_type: str, cache_file: Path, epochs: int, lr: float, seed: int,
                    n_head: int, head_dim: int, device: str):
    seed_everything(seed)
    data = torch.load(cache_file)
    H = data["hidden"].to(device)
    A = data["attn"].to(device)

    if predictor_type == "mlp":
        model = AttnPredictor1(dim=head_dim, hidden_dim=HIDDEN_DIM, n_head=n_head).to(device)
    elif predictor_type == "cnn":
        model = CNNAttnPredictor(dim=head_dim, hidden_dim=HIDDEN_DIM, n_head=n_head).to(device)
    else:
        raise ValueError(predictor_type)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses = []
    bs = 1
    n = H.size(0)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        steps = 0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            x = H[idx]
            y = A[idx]
            pred = model(x)
            loss = F.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
            steps += 1
        avg = total / max(steps, 1)
        losses.append(avg)
        if ep == 0 or (ep + 1) % 10 == 0:
            print(f"[train {predictor_type} seed={seed}] epoch {ep + 1}/{epochs} mse={avg:.6f}", flush=True)
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="checkpoints/opt-1.3b")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--n_docs", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--out_dir", default="logs/extensions/cnn_predictor")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_file = build_cache(args.model_path, args.seq_len, args.n_docs, str(out_dir / "cache"), device)
    data = torch.load(cache_file, map_location="cpu")
    from transformers import AutoConfig
    model_cfg = AutoConfig.from_pretrained(args.model_path)
    n_head = getattr(model_cfg, "num_attention_heads", 32)
    head_dim = data["hidden"].size(-1) // n_head

    csv_path = out_dir / "loss.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["epoch", "seed", "predictor_type", "train_loss"])
        for predictor_type in ["mlp", "cnn"]:
            for seed in args.seeds:
                t0 = time.time()
                losses = train_predictor(
                    predictor_type, cache_file, args.epochs, args.lr, seed,
                    n_head, head_dim, device,
                )
                elapsed = time.time() - t0
                print(f"[done] {predictor_type} seed={seed} epochs={args.epochs} elapsed={elapsed:.1f}s",
                      flush=True)
                for ep, l in enumerate(losses, start=1):
                    w.writerow([ep, seed, predictor_type, l])
                f.flush()
    print(f"[done] wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
