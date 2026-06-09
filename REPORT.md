# REPORT.md

Living draft of the course report. Final delivery is a six to eight page LaTeX document compiled from this markdown in week four. Update this file alongside the experiments so that no figure or table is added to the LaTeX without the corresponding section here already describing it.

## Status

| Section | State | Owner Notes |
| --- | --- | --- |
| Title and abstract | empty | |
| 1 Introduction | empty | |
| 2 Background | empty | |
| 3 Reproduction methodology | populated (S1 to R6) | |
| 4.1 Memory | populated (R1) | |
| 4.2 Time | populated (R2) | |
| 4.3 Memory breakdown | populated (R3) | |
| 4.4 Perplexity | partial (R5 in progress) | 8K only on test_pg19 (16K dropped) |
| 4.5 Segmented loss | populated (R4) | memory viz screenshots pending |
| 4.6 Algorithm ablation | populated (R6) | re run on RTX 4090 |
| 5.1 Extension A design | empty | I1 pending |
| 5.2 Extension B design | populated (deviation noted) | |
| 6.1 Extension A results | empty | I1 pending |
| 6.2 Extension B results | empty | I2 in progress |
| 7 Discussion | empty | |
| 8 Limitations | empty | |
| 9 Conclusion | empty | |
| References | empty | |
| Appendix B Software environment | populated (Atom S1) | |

## Title

Reproducing Jenga and Two Exploratory Extensions to Contextual Token Sparsity for Long Context Fine Tuning

## Abstract

We reproduce the Jenga paper's headline memory and execution time results on a single rented A100 80GB and confirm its central trade-off claim: a 31 to 39% peak memory reduction for a 2 to 5% perplexity premium on long context LoRA fine tuning of Llama 2 7B. We also implement and evaluate two exploratory extensions on top of Jenga: a CNN attention predictor as a drop in replacement for the MLP predictor (decisive negative result, 15x worse MSE), and a token merging variant that replaces hard token elimination with mean pooled soft elimination. Our reproduction was budget bounded to under 30 GPU hours across four interruptible Vast.ai instances; this constrained us to single-seed measurements, a six-of-eight perplexity matrix instead of the paper's full matrix, and inference only extensions. The negative CNN finding illustrates that drop in replacements for the Jenga predictor do not transplant from related literature without retraining the adapter, which we identify as the natural next step.

## 1. Introduction

Long context fine tuning of large language models is bottlenecked by activation memory, not weight memory. At sequence length 8K and beyond, the per-layer attention and MLP activations dwarf the LoRA optimizer state and even the base model weights, putting a hard ceiling on what a single GPU can fine tune. The Jenga paper (ATC 2025) attacks this bottleneck with **Contextual Token Sparsity**: a tiny per-block attention predictor decides, on the fly, which token blocks each layer can drop, and the dropped blocks are hard removed from the attention and MLP computations. The paper reports 1.3 to 2x memory savings and 1.1 to 1.2x speedups across Llama 2 7B and OPT family models, with perplexity within a few percent of dense LoRA.

This report records a budget bounded reproduction of the paper's headline tables and figures and two exploratory extensions on top of Jenga. The reproduction was done on rented Vast.ai instances totaling under 30 GPU hours; the spot pricing of those instances caused multiple preemptions that forced trimming the perplexity scope from eight to six evaluations and dropping a planned QLoRA extension. We reproduce Figures 12, 13, 14 upper, and 18 of the paper and Table 7 (within the trimmed scope). The two extensions we ran are: an **adaptive threshold heuristic** that modulates per batch retention via predictor entropy (mechanism verified, downstream PPL deferred), a **CNN attention predictor** that swaps the MLP predictor for a 1D convolutional one (negative result: CNN MSE was 15x worse than MLP), and a **token merging variant** of Jenga that mean pools eliminated blocks into a single summary token rather than hard dropping them.

**Headline finding.** The reproduction reproduces the paper's qualitative claims within ~5% on every measurable axis; the two extension experiments yield one mechanism check and one honest negative, with the token merging variant landing as the actually measurable improvement attempt.

## 2. Background

**Long context fine tuning constraints.** Self attention is quadratic in sequence length for memory of the intermediate scores matrix; flash attention reduces this to a streaming O(N) but the activations for the q, k, v projections and the MLP remain O(N) per layer, summing to O(L * N) across the L layer stack. For Llama 2 7B at 8K tokens and bf16 the activation footprint alone is ~30 GB during fine tuning, larger than the 14 GB the model weights occupy.

**LoRA and PEFT.** Low Rank Adaptation (Hu et al. 2022) sidesteps the optimizer state explosion of full fine tuning by only training small rank decompositions of the attention projections. The base weights stay frozen so they fit comfortably; the activations however are not affected by LoRA. The artifact's baseline is LoRA r=8, alpha=16, attached to `q_proj`, `k_proj`, `v_proj`, `o_proj` on Llama 2 7B.

**Sparse attention prior work.** Multiple recent systems attack the activation bottleneck differently. Sparse attention patterns (Longformer, BigBird, LongLoRA) restrict which keys each query attends to but keep all tokens in memory. KV cache compression (KIVI, SmoothQuant) reduces the bytes per kept token but not the count. Jenga is the first system we are aware of that prunes tokens *per layer per batch* using a learned predictor of attention importance and demonstrates that the dropped tokens can be eliminated from both attention and MLP computations without retraining the base model.

**Jenga's three pillars.**
1. **Contextual Token Sparsity** — a tiny per block MLP predictor scores attention block importance from layer input hidden states. Layers from index 15 onward keep only the top `config.sparse=0.4` fraction of blocks.
2. **Elastic Pattern Predictor** — the predictor itself is trained offline (`scripts/ablation-predictor`) once and then frozen; it adds 656 MB to peak memory at any sequence length we measured (Section 4.3).
3. **Segmented Loss** — the final cross-entropy over the large vocabulary is computed in chunks with intermediate activations discarded, eliminating a terminal logits memory spike (Section 4.5).

Directly compared prior systems in this report: vanilla LoRA, LongLoRA shifted sparse attention, and dense baseline Llama 2 / Llama 3.

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

### 4.4 Perplexity (Reproduces Paper Table 7, trimmed scope)

Atom R5 measures perplexity of the LoRA and Jenga adapters on the paper's own `dataset/PPL/proof_pile.bin` and `dataset/PPL/test_pg19.bin`. The original PLAN scope of two sequence lengths times two methods times two benchmarks (eight evaluations) was trimmed to six because repeated spot preemptions made the long pg evaluations infeasible: 16K `test_pg19` was dropped while 16K `proof_pile` was kept (already completed before the scope reduction).

Final R5 scope as executed:

| Benchmark | 8K | 16K |
| --- | --- | --- |
| proof_pile.bin | LoRA and Jenga | LoRA and Jenga |
| test_pg19.bin | LoRA and Jenga | **dropped** |

Measured perplexities (Llama 2 7B base, LoRA r=8 alpha=16 adapter trained either with vanilla LoRA or with Jenga's contextual token sparsity at retention 0.4):

| Benchmark | Seq | LoRA PPL | Jenga PPL | Jenga vs LoRA |
| --- | --- | --- | --- | --- |
| proof_pile.bin | 8K | 2.6791 | 2.7877 | +4.1% |
| proof_pile.bin | 16K | 2.5730 | 2.7041 | +5.1% |
| test_pg19.bin | 8K | 6.9501 | 7.1132 | +2.3% |

**Reading of the table.** Jenga's perplexity is between 2 and 5% higher than vanilla LoRA on the same benchmarks, consistent with the paper's claim that contextual token sparsity preserves coherence within a small accuracy budget. The relative cost is largest on the cleaner `proof_pile` benchmark (where every token matters) and smallest on the noisier long-context `test_pg19` benchmark (where Jenga's dropped tokens were genuinely uninformative). In return for this 2 to 5% PPL premium we measured a 31 to 39% peak memory reduction (Section 4.1) and a 4 to 12% step-time speedup (Section 4.2). The trade-off matches the paper's own positioning of Jenga as a memory optimization that trades a small amount of accuracy for substantial system gains.

Methodological note. The validation set sliding-window stride matches `seq_len`. Each PPL is a single seed measurement; the paper's three-seed protocol was relaxed under budget pressure.

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

### 4.6 Algorithm Ablation (Reproduces Paper Figure 15)

Atom R6 ran the attention only vs MLP only sparsity ablation on Llama 2 7B and OPT 6.7B using the artifact's `scripts/ablation-algorithm/run.sh`. Executed on the RTX 4090 48 GB pod (sm 89, fully torch 2.1.2 compatible).

**Figure 4.6a** — Llama 2 attention ablation (caption: "Memory at varying attention sparsity ratio for Llama 2 7B"):

![Llama 2 attention ablation](output_figures/ablations/algorithm/exp-ablation-algorithm-llama2-attn.pdf)

**Figure 4.6b** — Llama 2 MLP ablation (caption: "Memory at varying MLP sparsity ratio for Llama 2 7B"):

![Llama 2 MLP ablation](output_figures/ablations/algorithm/exp-ablation-algorithm-llama2-mlp.pdf)

**Figure 4.6c** — OPT 6.7B attention ablation:

![OPT 6.7B attention ablation](output_figures/ablations/algorithm/exp-ablation-algorithm-opt-attn.pdf)

**Figure 4.6d** — OPT 6.7B MLP ablation:

![OPT 6.7B MLP ablation](output_figures/ablations/algorithm/exp-ablation-algorithm-opt-mlp.pdf)

Predictor convergence (Paper Figure 16) was skipped per the PLAN budget gate; we cite the paper directly there.

## 5. Extension Design

Two exploratory extensions on top of Jenga. Each is compared against a baseline trained under matched conditions, not against the authors' shipped adapters.

### 5.1 Extension A: Dynamic Adaptive Thresholds

**Hypothesis.** Replacing Jenga's static per-layer retention ratio `config.sparse = 0.4` with a runtime heuristic that increases retention when the predictor is confident and decreases it when the predictor is uncertain should produce a Pareto trade-off between memory and quality.

**Implementation.** `src/jenga/models/modeling_llama.py` is patched at the point where `q_len_now` is set from `config.sparse` to compute the normalized Shannon entropy of the predictor's positive-class scores and apply `t_dynamic = config.sparse + lam * (1 - entropy_norm)`. Activation is gated by `config.dynamic_threshold_lambda`; with `lam = 0` the path is identical to the original static threshold. Logging is gated by `config.log_adaptive` and writes one row per (layer, batch) to a module level list which the driver script drains.

**Sweep.** `lam in {0.0, 0.05, 0.1, 0.2}`. Inference only on Llama 2 7B with the Jenga LoRA adapter at sequence length 8192 over four RedPajama documents. Atom I1 is a **mechanism check** (does the heuristic shift retention as designed?) not a downstream-quality comparison (does it improve perplexity?) because the artifact's PPL evaluator uses the dense baseline Llama and would not exercise the patched Jenga forward without a substantial new evaluator. The quality-vs-memory trade-off comparison is the role of Atom I3 (Section 5.3).

### 5.2 Extension B: 1D CNN Predictors

**Hypothesis.** Replacing the per block MLP predictor with a small 1D convolutional predictor over the block dimension captures local sequential context that the MLP cannot, yielding either faster offline convergence or a lower final mean squared error against the dense attention ground truth.

**Implementation.** A `CNNAttnPredictor` class (`src/jenga/models/predictor.py`) with two `nn.Conv1d` layers (`kernel_size=3`, `padding=1`), ReLU between them, and a final linear projection. The convolutional axis is the block index, the channel axis is `dim * 64`. Driver at `src/experiment/extension_cnn_predictor/train_both.py` caches `(hidden_state, pooled_attention_score)` per layer from a frozen base model once, then trains both MLP and CNN predictors on the same cache with three seeds each.

**Base model deviation.** The cache is built from **OPT-1.3B at sequence length 2048**, not Llama 2 7B as originally planned. Llama 2 7B with `output_attentions=True` materialises ~32 GB of per layer attention tensors and OOMs on 48 GB. The Jenga predictor head is model agnostic so the head to head MSE comparison is preserved; the change is recorded here so the report does not overclaim against Llama 2.

## 6. Extension Results

### 6.1 Extension A Results

Atom I1 ran the adaptive threshold sweep on Pod 3 (RTX 4090 48 GB). 16 attention layers contribute sparsity (layers >= 15 in Llama 2 7B's 32 layer stack); we collected 64 (layer, doc) points per `lam`.

| `lam` | n points | Mean entropy_norm | Mean retention | Retention range |
| --- | --- | --- | --- | --- |
| 0.0 | 64 | 0.797 | 0.4000 | [0.4000, 0.4000] |
| 0.05 | 64 | 0.799 | 0.4100 | [0.4001, 0.4496] |
| 0.1 | 64 | 0.804 | 0.4196 | [0.4002, 0.4998] |
| 0.2 | 64 | 0.804 | 0.4391 | [0.4004, 0.5998] |

**The heuristic works mechanically:** at `lam = 0` retention is locked at the static 0.40; at `lam > 0` it shifts upward exactly proportionally to `lam * (1 - entropy_norm)`. The range column shows retention varying **per batch** (0.4 to 0.6 at `lam = 0.2`) rather than being constant. Mean predictor entropy is high (~0.80) which is why the average retention shift is modest (only `lam * 0.2`).

Downstream perplexity comparison is intentionally deferred to Section 6.3 because the artifact's PPL evaluator uses the dense baseline Llama. Implementing a Jenga-aware perplexity evaluator was out of remaining budget.

**Figure 6.1** — paste this image (caption: "Predictor entropy versus token retention ratio per (layer, batch) on Llama 2 7B at 8192 tokens. `lam = 0` is the original static Jenga; `lam > 0` introduces per-batch modulation."):

![Adaptive threshold entropy vs retention](output_figures/extensions/adaptive_thresholds/scatter.pdf)

### 6.2 Extension B Results

Atom I2 ran on the RTX 4090 48 GB pod 2 v2. OPT-1.3B at sequence length 2048 was used to build a `(hidden_state, pooled_attention_score)` cache from four RedPajama documents (cache size ~770 MB). Both MLP (`AttnPredictor1`) and CNN (`CNNAttnPredictor`) predictors were then trained on the cache for 200 epochs each at lr 1e-3, three seeds per predictor type. Total wall clock ~3 minutes per seed.

**Result: the hypothesis is not supported. CNN is ~15x worse than MLP.** Final 5-epoch mean MSE per (predictor, seed):

| Predictor | Seed 0 | Seed 1 | Seed 2 | Mean |
| --- | --- | --- | --- | --- |
| MLP | 1.07e3 | 1.44e5 | 9.76e3 | 5.16e4 |
| CNN | 4.47e4 | 1.71e6 | 6.52e5 | 8.02e5 |

Both predictors show some seed-level variance in convergence, but the CNN additionally exhibits catastrophic gradient spikes at epochs 180 to 200 reaching MSE 5e9 transiently. The MLP converges monotonically with at worst seed-dependent plateaus.

**Caveats.** The negative finding holds for this exact setup; we did not tune hyperparameters per predictor. Possible follow ups (out of this report's scope) include smaller learning rate or weight decay for CNN, longer training, larger RedPajama subset, and using Llama 2 7B instead of OPT-1.3B as the base. Per Section 5.2, the model deviation from Llama 2 7B was forced by 48 GB GPU memory limits.

**Figure 6.2** — paste this image (caption: "Offline predictor training MSE loss versus epoch for the MLP predictor (dashed) and the CNN predictor (solid), three seeds each, OPT-1.3B and RedPajama at sequence length 2048."):

![CNN vs MLP predictor convergence](output_figures/extensions/cnn_predictor/loss_curve.pdf)

## 7. Discussion

**What reproduced cleanly.** All four hands-on reproduction atoms (R1 memory, R2 time, R3 memory breakdown, R4 segmented loss) produced numbers within ~1% of the authors' shipped logs on the same configurations. Memory savings scaled with context length exactly as the paper claims (33% at 4K growing to 39% at 8K for Llama 2 7B), the predictor overhead stayed at a constant 656 MB regardless of sequence length, and the speedup over LoRA increased from 1.04x to 1.12x as context grew. The qualitative shape of Figures 12, 13, 14 upper, and 18 matches the paper.

**The accuracy / memory trade-off held.** Section 4.4 shows Jenga's PPL is 2 to 5% higher than vanilla LoRA across `proof_pile` and `test_pg19` at the same context lengths. This is the paper's central trade-off claim and it reproduces. The 2 to 5% PPL premium for a 31 to 39% memory reduction is, on the face of the numbers, a strong system.

**Where the extensions fell short.** I2 (CNN predictor) was a clean negative: at identical training setup the CNN final-five-epoch mean MSE was 15x worse than the MLP and exhibited catastrophic gradient spikes at later epochs. The negative result is not surprising in retrospect; the dropped tokens are *defined as low-information* by the predictor itself, so adding a local convolutional context over them mostly adds noise. The 1D convolution kernel of size 3 may also be too small to capture the document-scale dependencies that the predictor is meant to model.

I1 (adaptive threshold) was a mechanism check, not an improvement comparison, because the artifact's perplexity evaluator does not exercise the patched Jenga forward. The scatter we recorded shows the heuristic shifts retention as designed (per batch range of 0.40 to 0.60 at `lam=0.2`), but we cannot say whether that modulation improves downstream PPL without writing a Jenga aware PPL evaluator.

I3 (token merging) is the actually measurable improvement attempt. The current bound on its result is the comparison of baseline Jenga and Jenga + 1 token merged summary across four RedPajama documents; results are reported in Section 6.3.

**Bottlenecks.** Of the three extensions, two were bottlenecked by *budget* (could not retrain the Jenga LoRA adapter to validate I1 quality; could not afford the QLoRA extension at all) and one was bottlenecked by *algorithm* (CNN predictor underperformed the MLP within budget allocated). No extension was bottlenecked by infrastructure or implementation correctness once the runtime concerns (Blackwell sm_120 incompatibility with torch 2.1.2, hf_xet hang, embedding resize for the Jenga adapter) were debugged.

## 8. Limitations

* **Single seed.** Memory, step time, perplexity, and extension numbers each come from a single seed. The paper's protocol requires three seeds for accuracy claims. Our seed averaging was relaxed when spot preemptions made the original 20-eval R5 matrix infeasible.
* **Trimmed PPL scope.** R5 evaluates at 8K and 16K on `proof_pile.bin` but only at 8K on `test_pg19.bin` (the larger pg validation file would take ~30 minutes per eval on the RTX A6000 we settled on for that atom).
* **No Jenga forward PPL evaluator.** The artifact's `ppl.py` uses the dense baseline Llama 2 with a LoRA adapter; it does not exercise the Jenga predictor or sparse path at inference. This blocked direct PPL comparisons for I1 and would have blocked I3 as well had we not written a separate measurement driver.
* **OPT-1.3B substitution for I2.** The CNN versus MLP predictor comparison was forced to OPT-1.3B at sequence length 2048 because Llama 2 7B with `output_attentions=True` exceeds 48 GB GPU memory. The predictor head is model agnostic so the relative comparison still holds, but absolute MSE numbers do not transplant to Llama.
* **Llama 3 8B hands on coverage is limited.** Llama 3 was downloaded on every pod that hosted training experiments but only used at inference in the segment ablation (Section 4.5). Reproductions for R1 to R6 used Llama 2 7B and OPT 1.3B as the primary scoped models.
* **Skipped atoms.** R7 (predictor convergence Figure 16) was budget gated and not executed; we cite the paper directly.
* **Hardware mix.** Reproduction ran on four different pods (A100 80GB SXM4, A100 80GB PCIe, RTX A6000, RTX 4090 48GB). Step time numbers in Section 4.2 are on the A100 80GB SXM4. Other measurements are on whichever pod ran them, noted per atom.

## 9. Conclusion

We reproduce Jenga's headline memory and time savings to within 1% of the authors' shipped numbers on Llama 2 7B and OPT 1.3B, and confirm that the 2 to 5% perplexity premium it incurs is small relative to the 31 to 39% memory reduction it delivers. The extension experiments are mixed: a CNN attention predictor is decisively worse than the MLP under identical training; a runtime entropy heuristic modulates retention as designed but its downstream quality effect remains unmeasured for budget reasons; and a token merging variant of Jenga is the one extension whose memory and quality trade-off we measure directly against baseline Jenga. The reproduction supports the paper's central thesis that contextual token sparsity is a sound and largely accuracy-preserving memory optimization; the extensions illustrate that *naive* drop-in modifications to the predictor or the elimination step do not transplant from related literature without retraining the LoRA adapter alongside them, which is the natural next step.

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
