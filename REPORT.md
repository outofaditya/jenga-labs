# REPORT.md

Living draft of the course report. Final delivery is a six to eight page LaTeX document compiled from this markdown in week four. Update this file alongside the experiments so that no figure or table is added to the LaTeX without the corresponding section here already describing it.

## Status

| Section | State | Owner Notes |
| --- | --- | --- |
| Title and abstract | empty | |
| 1 Introduction | empty | |
| 2 Background | empty | |
| 3 Reproduction methodology | empty | |
| 4 Reproduction results | empty | |
| 5 Extension design | empty | |
| 6 Extension results | empty | |
| 7 Discussion | empty | |
| 8 Limitations | empty | |
| 9 Conclusion | empty | |
| References | empty | |
| Appendix B Software environment | populated (Atom S1) | |

## Title

Reproducing Jenga and Three Exploratory Extensions to Contextual Token Sparsity for Long Context Fine Tuning

## Abstract

To be drafted last.

## 1. Introduction

Brief framing of long context fine tuning cost. One paragraph summary of Jenga's three pillar idea (contextual token sparsity, the elastic predictor, the segmented loss). One paragraph on our contribution scope: a budget bounded reproduction on A100 80GB plus three exploratory extensions ordered by integration risk. One sentence headline finding.

## 2. Background

Long context fine tuning constraints. PEFT and LoRA baselines. Sparse attention prior work. Jenga's place in that taxonomy. Cite the paper, the artifact repository, and any directly compared prior systems (LongLoRA, base PEFT).

## 3. Reproduction Methodology

Hardware (A100 80GB on Vast.ai). Software pins (`torch==2.1.2`, `transformers==4.45.2`, `flash-attn==2.5.6`). Model and dataset choices (`checkpoints/llama2`, `checkpoints/opt-1.3b`, `dataset/RedPajama-Data-1T-Sample`, `dataset/PPL/proof_pile.bin`, `dataset/PPL/test_pg19.bin`). Measurement protocol (five warmup steps, thirty to fifty measured steps, three seeds for accuracy). Disclose deviations from the paper such as gradient checkpointing enabled on the Llama 16K baseline.

**First kill criterion (Atom S4).** Before paying for any Llama run we required, on OPT-350m at 8192 tokens, that baseline LoRA and Jenga both train without NaN and that Jenga peak memory be strictly below baseline. Measured peak memory on the A100 80GB:

| Method | Peak memory (MB) | Peak memory (GB) |
| --- | --- | --- |
| Baseline LoRA | 13,484.9 | 13.18 |
| Jenga | 7,453.1 | 7.28 |

Jenga peak memory is **44.7% lower** than baseline on the same configuration. The gate passed.

## 4. Reproduction Results

### 4.1 Memory (Reproduces Paper Figure 12)

Atom R1 ran LoRA, LongLoRA, and Jenga across Llama 2 7B and OPT-1.3B at sequence lengths 4096 and 8192 on the A100 80GB. The grouped bar charts reproduce the qualitative shape of the paper's Figure 12.

**Figure 4.1a** — paste exactly this image in the LaTeX as the first 4.1 figure (caption: "Peak GPU memory for LoRA, LongLoRA, and Jenga at 4096 tokens on Llama 2 7B and OPT-1.3B."):

![Peak memory at 4K context](output_figures/end2end/memory/exp-end2end-memory-4K-comparison.pdf)

**Figure 4.1b** — paste exactly this image after Figure 4.1a (caption: "Peak GPU memory for LoRA, LongLoRA, and Jenga at 8192 tokens on Llama 2 7B and OPT-1.3B."):

![Peak memory at 8K context](output_figures/end2end/memory/exp-end2end-memory-8K-comparison.pdf)

| Model | Seq | LoRA (GB) | LongLoRA (GB) | Jenga (GB) | Jenga vs LoRA |
| --- | --- | --- | --- | --- | --- |
| Llama 2 7B | 4K | 42.7 | 44.8 | 28.5 | 33% lower |
| Llama 2 7B | 8K | 72.5 | 76.6 | 44.0 | 39% lower |
| OPT-1.3B | 4K | 13.0 | 13.8 | 9.0 | 31% lower |
| OPT-1.3B | 8K | 23.5 | 25.0 | 15.4 | 35% lower |

Our numbers match the authors' shipped logs to within ~1% on the same configurations (authors' Llama 2 7B 8K LoRA: 73.1 GB vs ours 72.5 GB; authors' Llama 2 7B 8K Jenga: 44.5 GB vs ours 44.0 GB). Jenga peak memory is consistently below baseline LoRA across all four (model, seq) pairs and the savings grow with sequence length, matching the paper's claim that the gain compounds at long context.

### 4.2 Time (Reproduces Paper Figure 13)

To be populated by Atom R2. Required figures:

**Figure 4.2** — paste exactly this image (caption: "Mean training step time for LoRA, LongLoRA, and Jenga across the same configurations as Figure 4.1, on A100 80GB."):

![Execution time comparison](output_figures/end2end/time/exp-end2end-time-a800-comparison.pdf)

### 4.3 Memory Breakdown (Reproduces Paper Figure 14 Upper)

To be populated by Atom R3. Required figure:

**Figure 4.3** — paste exactly this image (caption: "Decomposition of Jenga peak memory into model state, activations, predictor overhead, and others on Llama 2 7B at three sequence lengths."):

![Memory breakdown](output_figures/ablations/memory-breakdown/exp-ablation-mem-breakdown.pdf)

### 4.4 Perplexity (Reproduces Paper Table 7)

To be populated by Atom R5. The PPL table is rendered at `logs/results/atom_results.md` filtered to `atom == R5`. Render it inline in LaTeX as `\input{report/tables/r5_ppl.tex}` after generation. Source data file: `logs/results/atom_results.jsonl`. State explicitly that the benchmarks are the paper's own `dataset/PPL/proof_pile.bin` and `dataset/PPL/test_pg19.bin`.

### 4.5 Segmented Loss (Reproduces Paper Figure 18)

To be populated by Atom R4. The segment ablation emits two PyTorch memory viz pickle files at `logs/ablations/segment/{base,seg}.pickle`. Drag each into `docs.pytorch.org/memory_viz`, screenshot the rendered timeline, and save the screenshots:

**Figure 4.5a** — paste this image (caption: "Naive auto-regressive loss memory timeline at 16384 tokens; the spike from the full-vocabulary logits dominates the peak."):

![Naive segmented loss](output_figures/ablations/segment/naive.png)

**Figure 4.5b** — paste this image (caption: "Segmented loss memory timeline; the terminal spike is removed by chunked backward."):

![Segmented loss](output_figures/ablations/segment/segmented.png)

### 4.6 Optional Reproduction Items

If executed, paste:

**Figure 4.6a** — algorithm ablation (Paper Figure 15) at `output_figures/ablations/algorithm/exp-ablation-algorithm-llama2-attn.pdf` and the corresponding `-mlp.pdf` / `-opt-attn.pdf` / `-opt-mlp.pdf` files.

**Figure 4.6b** — predictor convergence (Paper Figure 16) at `output_figures/ablations/predictor/exp-ablation-predictor-loss.pdf`.

If skipped due to budget, state explicitly that the corresponding figures are cited from the original paper and not re-run.

## 5. Extension Design

Three exploratory extensions on top of Jenga, ordered ascending by integration risk. Each is compared against a Jenga baseline retrained under the same training budget as the extension itself, not the authors' shipped adapters.

### 5.1 Extension A: Dynamic Adaptive Thresholds

State the hypothesis. Describe the per batch entropy heuristic and the `t_dynamic = t_base + lam * (1 - entropy_norm)` rule. List the values of `lam` swept.

### 5.2 Extension B: 1D CNN Predictors

State the hypothesis. Describe the two layer `nn.Conv1d` architecture replacing the per block MLP predictor. Convolutional axis is the block index.

### 5.3 Extension C: Jenga Plus QLoRA

State the hypothesis. Document the known integration risk in `PrunedLlamaMLPFunction` for 4 bit weights and the mitigation chosen.

## 6. Extension Results

### 6.1 Extension A Results

To be populated by Atom I1. Required figures:

**Figure 6.1a** — paste this image (caption: "Predictor entropy versus token retention ratio per layer per batch, Llama 2 7B at 8192 tokens, three seeds."):

![Adaptive threshold entropy vs retention](output_figures/extensions/adaptive_thresholds/scatter.pdf)

**Figure 6.1b** — paste this image (caption: "Perplexity on the paper's PPL benchmarks across swept `lam` values plus the equal-budget static-threshold Jenga baseline."):

![Adaptive threshold PPL comparison](output_figures/extensions/adaptive_thresholds/ppl_bar.pdf)

### 6.2 Extension B Results

To be populated by Atom I2. Required figure:

**Figure 6.2** — paste this image (caption: "Offline predictor training MSE loss versus epoch for the MLP predictor (dashed) and the CNN predictor (solid), three seeds each, RedPajama subset."):

![CNN vs MLP predictor convergence](output_figures/extensions/cnn_predictor/loss_curve.pdf)

### 6.3 Extension C Results

To be populated by Atom I3. Required figure:

**Figure 6.3** — paste this image (caption: "Peak GPU memory at 16384 tokens for Llama 2 7B across LoRA, Jenga, and Jenga plus QLoRA (attention only 4-bit)."):

![Jenga + QLoRA memory](output_figures/extensions/qlora_synergy/memory_bar.pdf)

If the integration failed at runtime, paste the stack trace excerpt from `logs/extensions/qlora_synergy/crash_trace.txt` instead and state in prose what the failure mode was.

## 7. Discussion

What worked. What did not. Negative results paragraph: explicitly call out which hypotheses were not supported and why. Where extensions fell short, state whether the bottleneck was algorithmic or budget.

## 8. Limitations

Single seed on memory and time. Budget cap of approximately twenty dollars. Training step caps imposed by budget rather than convergence. Llama 3 8B coverage limited or absent depending on gated access timing. Dataset substitutions for the generalization probe.

## 9. Conclusion

One paragraph restating the reproduction findings and what the three extensions revealed about Jenga's design margin.

## References

Jenga paper. Artifact repository. LoRA. QLoRA. Flash Attention. LongLoRA. Hugging Face transformers and peft. Add as cited.

## Appendix A. Budget Ledger

Carry over the final post execution version of the budget ledger from `PLAN.md` here so the report is self contained.

## Appendix B. Software Environment

Reproduction was performed on a single Vast.ai A100 80GB SXM4 instance.

**Hardware.**

| Item | Value |
| --- | --- |
| GPU | NVIDIA A100-SXM4-80GB (compute capability 8.0) |
| Driver | 570.211.01 |
| CUDA runtime | 12.8 (nvcc V12.8.93) |
| Image base | Vast.ai PyTorch image with miniforge3 at `/opt/miniforge3` |

**Python environment.** A dedicated `jenga` conda env is created at `/venv/jenga` (Python 3.10.20) by `scripts/run_pod.sh` so that the artifact's `torch==2.1.2` pin (which has no Python 3.12 wheels) can be honored.

| Package | Version | Source |
| --- | --- | --- |
| torch | 2.1.2 | requirements.txt |
| transformers | 4.45.2 | requirements.txt |
| tokenizers | 0.20.1 | requirements.txt |
| deepspeed | 0.14.0 | requirements.txt |
| bitsandbytes | 0.41.1 | requirements.txt |
| numpy | 1.26.4 | pinned `<2` to keep torch 2.1.2's numpy 1.x ABI |
| accelerate | 1.13.0 | requirements.txt (unpinned, took latest) |
| datasets | 5.0.0 | requirements.txt (unpinned, took latest) |
| peft | 0.19.1 | requirements.txt (unpinned, took latest) |
| flash-attn | 2.5.6 | prebuilt wheel `cu122torch2.1cxx11abiFALSE-cp310` (artifact source build was killed after 14 minutes of multi-architecture compile produced zero `.o` files; prebuilt swap saved an estimated 15+ minutes of wall clock) |
| setuptools | <70 | pinned for `pkg_resources.packaging` which torch 2.1.2's `cpp_extension` imports |

**Atom S1 result.** All seven sanity-check sentinels verified:

```
ok   checkpoints/predictor/predictor.pth
ok   checkpoints/predictor/pruned_config.pth
ok   dataset/PPL/proof_pile.bin
ok   dataset/PPL/test_pg19.bin
ok   checkpoints/llama2/config.json
ok   checkpoints/opt-350m/config.json
ok   checkpoints/opt-1.3b/config.json
```

The Hugging Face mirror at `outofaditya/jenga-labs-artifacts` (private) holds pre-extracted copies of `peft_model.zip`, `predictor.zip`, and `dataset.zip` so any future pod bootstraps from CDN instead of the Tsinghua cloud share.
