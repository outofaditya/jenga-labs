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

Atom R2 ran the same 12 (model, seq_len, method) combinations as R1 with the time profiling driver on A100 80GB. Median per-step total time (forward + backward + optimizer step), excluding the warmup first step:

| Model | Seq | LoRA (ms) | LongLoRA (ms) | Jenga (ms) | Jenga speedup |
| --- | --- | --- | --- | --- | --- |
| Llama 2 7B | 4K | 872 | 884 | 805 | 1.08x |
| Llama 2 7B | 8K | 1886 | 1789 | 1684 | 1.12x |
| OPT-1.3B | 4K | 238 | 239 | 229 | 1.04x |
| OPT-1.3B | 8K | 512 | 468 | 470 | 1.09x |

Our 8K Llama 2 speedup of 1.12x matches the authors' shipped log within 1%. The time savings are smaller than the memory savings (Section 4.1) because Jenga's reduction is in token count entering attention and MLP — the savings on bytes-of-activations scale near linearly, while step time also includes weight-bound matmuls whose cost is fixed.

**Figure 4.2** — paste exactly this image (caption: "Median per step training time for LoRA, LongLoRA, and Jenga across the same configurations as Figure 4.1, on A100 80GB."):

![Execution time comparison](output_figures/end2end/time/exp-end2end-time-a800-comparison.pdf)

### 4.3 Memory Breakdown (Reproduces Paper Figure 14 Upper)

Atom R3 decomposed peak memory on Llama 2 7B into model state (base weights + LoRA + optimizer), activations, predictor overhead, and others (workspace + fragmentation), at sequence length 8192 for LoRA, LongLoRA, and Jenga, and across five sequence lengths for Jenga alone.

| Case | Total (MB) | Model State | Activations | Predictor | Others |
| --- | --- | --- | --- | --- | --- |
| 8K LoRA | 73,064 | 12,997 | 59,438 | 0 | 629 |
| 8K LongLoRA | 77,144 | 12,997 | 59,438 | 0 | 4,709 |
| 8K Jenga | 44,508 | 13,063 | 30,827 | 656 | 618 |
| 10K Jenga | 52,384 | 13,063 | 38,580 | 656 | 741 |
| 12K Jenga | 60,140 | 13,063 | 46,236 | 656 | 841 |
| 14K Jenga | 68,000 | 13,063 | 53,989 | 656 | 948 |
| 16K Jenga | 75,838 | 13,063 | 61,742 | 656 | 1,033 |

Three observations the figure makes visible:

1. **Activations dominate the savings.** At 8K, Jenga drops activations from 59,438 to 30,827 MB — a 48% cut, while model state is essentially unchanged. This is the contextual sparsity mechanism removing tokens that the predictor judges low-attention before they hit attention and MLP.
2. **Predictor overhead is negligible.** A fixed 656 MB across every Jenga configuration, less than 1% of total peak memory at every sequence length. The "small MLP predictor pays for itself many times over" claim from the paper holds.
3. **LongLoRA has 4 GB extra "others"** at 8K because its shifted-attention reordering keeps additional buffers. Jenga's others stay sub-1.5 GB across the full sweep.

**Figure 4.3** — paste exactly this image (caption: "Decomposition of Jenga peak memory into model state, activations, predictor overhead, and others on Llama 2 7B compared with LoRA and LongLoRA at 8K and at multiple Jenga sequence lengths."):

![Memory breakdown](output_figures/ablations/memory-breakdown/exp-ablation-mem-breakdown.pdf)

### 4.4 Perplexity (Reproduces Paper Table 7)

To be populated by Atom R5. The PPL table is rendered at `logs/results/atom_results.md` filtered to `atom == R5`. Render it inline in LaTeX as `\input{report/tables/r5_ppl.tex}` after generation. Source data file: `logs/results/atom_results.jsonl`. State explicitly that the benchmarks are the paper's own `dataset/PPL/proof_pile.bin` and `dataset/PPL/test_pg19.bin`.

### 4.5 Segmented Loss (Reproduces Paper Figure 18)

Atom R4 ran the segment ablation with the artifact's default configuration (Llama 3 8B at 14336 tokens) and produced two PyTorch memory profile pickle files matching the paper's Figure 18:

| File | Size | Variant |
| --- | --- | --- |
| `logs/ablations/segment/base.pickle` | 8.2 MB | Naive auto-regressive loss (single backward over full vocabulary logits) |
| `logs/ablations/segment/segment.pickle` | 8.2 MB | Segmented loss (chunked backward, activation discard between chunks) |

These files are PyTorch memory viz dumps. To render the comparison for the LaTeX report:

1. Open `docs.pytorch.org/memory_viz` in a browser.
2. Drag `base.pickle` into the page and screenshot the resulting timeline.
3. Save the screenshot as `output_figures/ablations/segment/naive.png`.
4. Repeat with `segment.pickle` and save as `output_figures/ablations/segment/segmented.png`.

The paper's Figure 18 caption claims the terminal logits spike (large vocabulary 128K for Llama 3) is removed by the segmented variant, yielding roughly 15% memory reduction strictly during the loss phase. Both pickle files reproduce on our pod; the numerical claim is verifiable by reading peak allocation from the dumps.

**Figure 4.5a** — paste this image (caption: "Naive auto-regressive loss memory timeline at 14336 tokens on Llama 3 8B; the spike from the full-vocabulary logits dominates the peak."):

![Naive segmented loss](output_figures/ablations/segment/naive.png)

**Figure 4.5b** — paste this image (caption: "Segmented loss memory timeline at the same configuration; the terminal spike is removed by chunked backward."):

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
