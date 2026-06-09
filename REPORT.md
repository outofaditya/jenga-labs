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

Atom R1 ran LoRA, LongLoRA, and Jenga across Llama 2 7B and OPT-1.3B at sequence lengths 4096 and 8192 on the A100 80GB. The grouped bar chart at `output_figures/end2end/memory/exp-end2end-memory-{4K,8K}-comparison.pdf` reproduces the qualitative shape of the paper's Figure 12.

| Model | Seq | LoRA (GB) | LongLoRA (GB) | Jenga (GB) | Jenga vs LoRA |
| --- | --- | --- | --- | --- | --- |
| Llama 2 7B | 4K | 42.7 | 44.8 | 28.5 | 33% lower |
| Llama 2 7B | 8K | 72.5 | 76.6 | 44.0 | 39% lower |
| OPT-1.3B | 4K | 13.0 | 13.8 | 9.0 | 31% lower |
| OPT-1.3B | 8K | 23.5 | 25.0 | 15.4 | 35% lower |

Our numbers match the authors' shipped logs to within ~1% on the same configurations (authors' Llama 2 7B 8K LoRA: 73.1 GB vs ours 72.5 GB; authors' Llama 2 7B 8K Jenga: 44.5 GB vs ours 44.0 GB). Jenga peak memory is consistently below baseline LoRA across all four (model, seq) pairs and the savings grow with sequence length, matching the paper's claim that the gain compounds at long context.

### 4.2 Time (Reproduces Paper Figure 13)

Insert grouped bar chart of mean step time with standard deviation as error bars. Annotate the relative speedup.

### 4.3 Memory Breakdown (Reproduces Paper Figure 14 Upper)

Insert stacked horizontal bar chart of model state, activations, predictor, others.

### 4.4 Perplexity (Reproduces Paper Table 7)

Insert table cross referencing sequence length against baseline LoRA and Jenga perplexity on `proof_pile.bin` and `pg.bin`. State explicitly that these are the paper's benchmark files.

### 4.5 Segmented Loss (Reproduces Paper Figure 18)

Insert the memory viz visualization or its summary table for naive versus segmented loss at sequence length 16384.

### 4.6 Optional Reproduction Items

If executed, predictor convergence (Paper Figure 16) and algorithm ablation (Paper Figure 15). If skipped, state explicitly that we cite the paper directly here.

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

Insert scatter of predictor entropy versus token retention ratio. Insert bar chart of perplexity across the swept `lam` values plus the equal budget Jenga baseline. State whether the trend matches the hypothesis.

### 6.2 Extension B Results

Insert the dual line plot of epoch versus MSE loss for the MLP and CNN predictors. State convergence rate and final loss comparison.

### 6.3 Extension C Results

Insert grouped bar chart of peak memory at sequence length 16384 across LoRA, Jenga, and Jenga plus QLoRA. If the integration failed, document the failure mode and the dequantization mitigation attempt.

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
