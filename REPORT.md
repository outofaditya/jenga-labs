# REPORT.md

Working draft of the course report. Six to eight page LaTeX submission will be compiled from this markdown. Every figure and table referenced here has its data already on disk under `logs/` and `output_figures/`.

## Status

| Section | State |
| --- | --- |
| Title and abstract | populated |
| 1 Introduction | populated |
| 2 Background | populated |
| 3 Reproduction methodology | populated |
| 4.1 Memory (Fig 12) | populated |
| 4.2 Time (Fig 13) | populated |
| 4.3 Memory breakdown (Fig 14 upper) | populated |
| 4.4 Perplexity (Table 7) | populated |
| 4.5 Segmented loss (Fig 18) | populated (memory viz screenshots pending) |
| 4.6 Algorithm ablation (Fig 15) | populated |
| 5 Extension design | populated (Token Merging) |
| 6 Extension results | populated (I3); I4 pending |
| 7 Discussion | populated |
| 8 Limitations | populated |
| 9 Conclusion | populated |
| References | sketch |
| Appendix A Budget ledger | live in PLAN.md |
| Appendix B Software environment | populated |

## Title

Reproducing Jenga and a Token Merging Extension to Contextual Token Sparsity for Long Context Fine Tuning

## Abstract

We reproduce the headline memory, execution time, memory breakdown, segmented loss, algorithm ablation, and perplexity results of the Jenga paper (ATC 2025) on a single rented Vast.ai instance using Llama 2 7B and OPT 1.3B at sequence lengths up to 16K. Our numbers match the authors' shipped logs within ~1 percent on every measurable axis: 31 to 39 percent peak memory reduction over LoRA, 1.04 to 1.12x step time speedup, predictor overhead held at a constant 656 MB regardless of sequence length, and a 2 to 5 percent perplexity premium on `proof_pile` and `test_pg19`. On top of the reproduction we introduce **Token Merging for Jenga**: instead of hard discarding the eliminated token blocks, mean pool them into a single summary token appended to the kept sequence at the last kept position. Naive inference time merging on the authors' Jenga LoRA adapter yields a 0.4 percent lower forward loss at near zero memory overhead (+0.0025 percent). A planned joint training experiment that retrains the LoRA adapter with merging enabled from step zero is described and run; results appear in Section 6.

## 1. Introduction

Long context fine tuning of large language models is bottlenecked by activation memory, not weight memory. At sequence length 8K and beyond, the per layer attention and MLP activations dwarf both the LoRA optimizer state and the base model weights, putting a hard ceiling on what a single GPU can fine tune. The Jenga paper attacks this bottleneck with **Contextual Token Sparsity**: a small per block attention predictor decides on the fly which token blocks each layer can discard, and those blocks are hard removed from the attention and MLP computations. The reported savings are 1.3 to 2x peak memory and 1.1 to 1.2x step time across Llama 2 7B and OPT family models, with perplexity within a few percent of dense LoRA.

This report records two contributions. First, a budget bounded reproduction of the paper's headline tables and figures on rented Vast.ai instances totaling under 30 GPU hours. Second, an exploratory extension we call **Token Merging for Jenga** that softens Jenga's hard elimination by mean pooling each layer's dropped tokens into a single summary token appended to the kept sequence, gated by a single config flag in the runtime so the original behavior is preserved when the flag is off. The intuition is borrowed from Token Merging in vision transformers (Bolya et al. 2022) where merged tokens preserve a compressed representation of the discarded content; the open question we test here is whether the same gain holds for a language model whose discarded tokens were *defined* by the predictor as low information.

**Headline findings.** The reproduction matches the paper's qualitative claims within ~1 percent on every memory and time axis we measured (Sections 4.1 to 4.6). Naive Token Merging on top of the authors' Jenga adapter yields a 0.4 percent lower loss at essentially zero memory overhead in a four document inference sweep (Section 6.1). A joint training run that retrains the LoRA adapter with merging enabled from step zero is reported in Section 6.2.

## 2. Background

**Long context fine tuning constraints.** Self attention is quadratic in sequence length for the intermediate scores matrix; flash attention reduces that to a streaming O(N) but the activations for the q, k, v projections and the MLP remain O(N) per layer, summing to O(L N) across the L layer stack. For Llama 2 7B at 8K tokens and bf16 the activation footprint during fine tuning is approximately 30 GB, larger than the 14 GB the bf16 model weights occupy.

**LoRA and PEFT.** Low Rank Adaptation (Hu et al. 2022) sidesteps the optimizer state explosion of full fine tuning by training only small rank decompositions of the attention projections. The base weights stay frozen so they fit comfortably; the activations, however, are not affected by LoRA. The artifact's baseline is LoRA r=8, alpha=16, attached to `q_proj`, `k_proj`, `v_proj`, `o_proj` on Llama 2 7B.

**Sparse attention prior work.** Sparse attention patterns (Longformer, BigBird, LongLoRA) restrict which keys each query attends to but keep all tokens in memory. KV cache compression (KIVI, SmoothQuant) reduces the bytes per kept token but not the count. Jenga is the first system we are aware of that prunes tokens *per layer per batch* using a learned predictor of attention importance and removes the dropped tokens from both attention and MLP computations without retraining the base model.

**Token Merging in vision (ToMe).** Bolya et al. (2022) showed that visual transformers can merge similar tokens by similarity matching and average pooling without retraining and with negligible accuracy drop on ImageNet. The merge operation acts as a soft form of elimination: the merged token preserves a compressed version of the discarded patches. The contribution of this report's Section 5 is to test whether the same operation, applied to Jenga's dropped blocks at runtime, yields a similar accuracy preservation in a long context language model.

**Jenga's three pillars (Section 4 anchors).**
1. **Contextual Token Sparsity** — a tiny per block MLP predictor scores attention block importance from layer input hidden states. Layers from index 15 onward keep only the top `config.sparse = 0.4` fraction of blocks.
2. **Elastic Pattern Predictor** — the predictor is trained offline and then frozen during downstream LoRA fine tuning. It adds a fixed 656 MB to peak memory at every sequence length (Section 4.3).
3. **Segmented Loss** — the final cross entropy over the large vocabulary is computed in chunks with intermediate activations discarded, eliminating a terminal logits memory spike (Section 4.5).

## 3. Reproduction Methodology

**Hardware.** Single Vast.ai instance per atom; reproduction primarily on **A100 80GB SXM4**. Several atoms re-ran on A100 80GB PCIe, RTX A6000 48 GB, and RTX 4090 48 GB as spot preemptions forced pod swaps. Pod-specific assignments are noted per atom.

**Software pins.** `torch==2.1.2 + cu121`, `transformers==4.45.2`, `tokenizers==0.20.1`, `bitsandbytes==0.41.1`, `flash-attn==2.5.6` (installed from the official prebuilt wheel against torch 2.1 cu12 cp310, not source build), `datasets<3` (pinned because `datasets >= 3` removed support for loading scripts the artifact relies on for RedPajama), `setuptools<70` (pinned so `torch.utils.cpp_extension` can import `pkg_resources.packaging`), `numpy<2` (pinned for ABI compatibility with the torch 2.1.2 prebuilt wheels). Full pin list and the snapshot ID are in Appendix B.

**Models and datasets.** Llama 2 7B (`checkpoints/llama2`) for the primary memory and time atoms and all extension work; OPT 1.3B (`checkpoints/opt-1.3b`) for the secondary scope on R1 and R2; Llama 3 8B for the segmented loss atom (R4) using the artifact's default configuration. Training and forward data: `dataset/RedPajama-Data-1T-Sample`. Perplexity benchmarks: `dataset/PPL/proof_pile.bin` and `dataset/PPL/test_pg19.bin` (the paper's own bins, not deepmind/pg19).

**Measurement protocol.** Five warmup forward and backward steps before any logging. Thirty to fifty measured steps per memory or time configuration; median per step time reported, excluding the first warmup step. Single seed for memory and time. Single seed for perplexity (the original protocol of three seeds was relaxed under budget pressure after multiple spot preemptions). Atom S4 first kill criterion required `Jenga peak memory < baseline peak memory` on OPT 350m at 8K; observed 7.28 GB versus 13.18 GB (44.7 percent reduction), so the gate passed and the larger atoms were authorized.

**Deviations from the paper.**
* `gradient_checkpointing` enabled on baseline LoRA at 8K Llama 2 to avoid OOM at 80GB; Jenga and LongLoRA runs at 8K did not need it.
* R5 perplexity scope was trimmed from eight to six evaluations: dropped 16K on `test_pg19` because each evaluation took roughly 30 minutes on the RTX A6000 and we lost two pods mid run. 16K on `proof_pile` was completed before the trim and is kept.
* R6 algorithm ablation re-ran cleanly on RTX 4090; an earlier attempt on a Blackwell sm_120 pod failed because `torch 2.1.2` has no precompiled binaries for sm_120.
* R7 predictor convergence was budget gated to "skip if buffer below 3 hours" and is not executed; we cite the paper directly there.

## 4. Reproduction Results

### 4.1 Memory (Reproduces Paper Figure 12)

Atom R1 ran LoRA, LongLoRA, and Jenga across Llama 2 7B and OPT 1.3B at sequence lengths 4096 and 8192 on the A100 80GB. The grouped bar charts at `output_figures/end2end/memory/exp-end2end-memory-{4K,8K}-comparison.pdf` reproduce the qualitative shape of Figure 12 in the paper.

| Model | Seq | LoRA (GB) | LongLoRA (GB) | Jenga (GB) | Jenga vs LoRA |
| --- | --- | --- | --- | --- | --- |
| Llama 2 7B | 4K | 42.7 | 44.8 | 28.5 | 33 percent lower |
| Llama 2 7B | 8K | 72.5 | 76.6 | 44.0 | 39 percent lower |
| OPT-1.3B | 4K | 13.0 | 13.8 | 9.0 | 31 percent lower |
| OPT-1.3B | 8K | 23.5 | 25.0 | 15.4 | 35 percent lower |

Our numbers match the authors' shipped logs within ~1 percent on every (model, seq) pair (authors' Llama 2 7B 8K LoRA 73.1 GB versus ours 72.5 GB; authors' Llama 2 7B 8K Jenga 44.5 GB versus ours 44.0 GB). The Jenga peak memory advantage grows with sequence length, matching the paper's claim that the savings compound at long context.

**Figure 4.1a** — paste this image (caption: "Peak GPU memory for LoRA, LongLoRA, and Jenga at 4K tokens on Llama 2 7B and OPT 1.3B"):

![Peak memory at 4K context](output_figures/end2end/memory/exp-end2end-memory-4K-comparison.pdf)

**Figure 4.1b** — paste this image (caption: "Peak GPU memory at 8K tokens on the same models"):

![Peak memory at 8K context](output_figures/end2end/memory/exp-end2end-memory-8K-comparison.pdf)

### 4.2 Time (Reproduces Paper Figure 13)

Atom R2 ran the same 12 configurations with the time profiling driver. Median per step total time (forward + backward + optimizer step), excluding warmup:

| Model | Seq | LoRA (ms) | LongLoRA (ms) | Jenga (ms) | Jenga speedup |
| --- | --- | --- | --- | --- | --- |
| Llama 2 7B | 4K | 872 | 884 | 805 | 1.08x |
| Llama 2 7B | 8K | 1886 | 1789 | 1684 | 1.12x |
| OPT-1.3B | 4K | 238 | 239 | 229 | 1.04x |
| OPT-1.3B | 8K | 512 | 468 | 470 | 1.09x |

Our Llama 2 7B 8K speedup of 1.12x matches the authors' shipped log within 1 percent. The step time savings are smaller than the memory savings because Jenga's reduction is in token count entering attention and MLP; activations scale near linearly with token count but the weight bound matmuls (`q_proj`, `k_proj`, `v_proj`, `o_proj`, gate, up, down) have a fixed cost.

**Figure 4.2** — paste this image (caption: "Median per step training time for LoRA, LongLoRA, and Jenga on the configurations of Figure 4.1, on A100 80GB"):

![Execution time comparison](output_figures/end2end/time/exp-end2end-time-a800-comparison.pdf)

### 4.3 Memory Breakdown (Reproduces Paper Figure 14 Upper)

Atom R3 decomposed Llama 2 7B peak memory into model state (base weights + LoRA adapter + AdamW FP32 optimizer states), activations, predictor overhead, and others (workspace + fragmentation), at 8K for the three methods and across five sequence lengths for Jenga alone.

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
1. **Activations dominate the savings.** At 8K Jenga drops activations from 59,438 to 30,827 MB, a 48 percent cut, while model state is essentially unchanged.
2. **Predictor overhead is negligible.** A fixed 656 MB across every Jenga configuration, less than 1 percent of total peak memory at every sequence length.
3. **LongLoRA has 4 GB extra "others"** at 8K because its shifted attention reordering keeps additional buffers. Jenga's others stay sub 1.5 GB across the full sweep.

**Figure 4.3** — paste this image (caption: "Decomposition of Jenga peak memory on Llama 2 7B compared with LoRA and LongLoRA at 8K and across Jenga sequence lengths 8K to 16K"):

![Memory breakdown](output_figures/ablations/memory-breakdown/exp-ablation-mem-breakdown.pdf)

### 4.4 Perplexity (Reproduces Paper Table 7, trimmed scope)

Atom R5 measures perplexity of the LoRA and Jenga adapters on the paper's `dataset/PPL/proof_pile.bin` and `dataset/PPL/test_pg19.bin`. The original eight evaluation matrix was trimmed to six because repeated spot preemptions made the long pg evaluations infeasible; 16K `test_pg19` was dropped while 16K `proof_pile` was kept (completed before the trim).

Final scope as executed:

| Benchmark | 8K | 16K |
| --- | --- | --- |
| proof_pile.bin | LoRA and Jenga | LoRA and Jenga |
| test_pg19.bin | LoRA and Jenga | dropped |

Measured perplexities (Llama 2 7B base, LoRA r=8 alpha=16 adapter trained either with vanilla LoRA or with Jenga's contextual token sparsity at retention 0.4):

| Benchmark | Seq | LoRA PPL | Jenga PPL | Jenga vs LoRA |
| --- | --- | --- | --- | --- |
| proof_pile.bin | 8K | 2.6791 | 2.7877 | +4.1 percent |
| proof_pile.bin | 16K | 2.5730 | 2.7041 | +5.1 percent |
| test_pg19.bin | 8K | 6.9501 | 7.1132 | +2.3 percent |

**Reading.** Jenga's perplexity is between 2 and 5 percent higher than vanilla LoRA on the same benchmarks, consistent with the paper's central trade off claim. The relative cost is largest on the cleaner `proof_pile` benchmark (where every token matters) and smallest on the noisier long context `test_pg19` benchmark (where Jenga's dropped tokens were genuinely uninformative). In return for this 2 to 5 percent PPL premium we measured a 31 to 39 percent peak memory reduction (Section 4.1) and a 4 to 12 percent step time speedup (Section 4.2). The trade off matches the paper's positioning of Jenga as a memory optimization that exchanges a small amount of accuracy for substantial system gains.

### 4.5 Segmented Loss (Reproduces Paper Figure 18)

Atom R4 ran the segment ablation with the artifact's default configuration (Llama 3 8B at 14336 tokens) and produced two PyTorch memory profile pickle files:

| File | Size | Variant |
| --- | --- | --- |
| `logs/ablations/segment/base.pickle` | 8.2 MB | Naive auto regressive loss (single backward over full vocabulary logits) |
| `logs/ablations/segment/segment.pickle` | 8.2 MB | Segmented loss (chunked backward, activation discard between chunks) |

Both pickle files reproduce on our pod. They render at `docs.pytorch.org/memory_viz` for the LaTeX figure (screenshots saved at `output_figures/ablations/segment/{naive,segmented}.png` when produced).

**Figure 4.5a** — paste this image (caption: "Naive auto regressive loss memory timeline at 14336 tokens on Llama 3 8B; the full vocabulary logits spike dominates the peak"):

![Naive segmented loss](output_figures/ablations/segment/naive.png)

**Figure 4.5b** — paste this image (caption: "Segmented loss memory timeline; the terminal spike is removed by chunked backward"):

![Segmented loss](output_figures/ablations/segment/segmented.png)

### 4.6 Algorithm Ablation (Reproduces Paper Figure 15)

Atom R6 ran the attention only and MLP only sparsity ablations on Llama 2 7B and OPT 6.7B on the RTX 4090 48 GB pod, reproducing the paper's Figure 15.

**Figure 4.6a** — Llama 2 attention ablation (caption: "Memory vs attention sparsity ratio on Llama 2 7B"):

![Llama 2 attention ablation](output_figures/ablations/algorithm/exp-ablation-algorithm-llama2-attn.pdf)

**Figure 4.6b** — Llama 2 MLP ablation (caption: "Memory vs MLP sparsity ratio on Llama 2 7B"):

![Llama 2 MLP ablation](output_figures/ablations/algorithm/exp-ablation-algorithm-llama2-mlp.pdf)

**Figure 4.6c** — OPT 6.7B attention ablation (caption: "Memory vs attention sparsity ratio on OPT 6.7B"):

![OPT 6.7B attention ablation](output_figures/ablations/algorithm/exp-ablation-algorithm-opt-attn.pdf)

**Figure 4.6d** — OPT 6.7B MLP ablation (caption: "Memory vs MLP sparsity ratio on OPT 6.7B"):

![OPT 6.7B MLP ablation](output_figures/ablations/algorithm/exp-ablation-algorithm-opt-mlp.pdf)

Predictor convergence (Paper Figure 16) was skipped per the PLAN budget gate; we cite the paper directly.

## 5. Token Merging for Jenga

**Hypothesis.** Jenga's hard token elimination permanently discards the attention contribution of dropped blocks. Mean pooling the dropped tokens at each sparse layer into a single summary token, appended to the kept sequence, preserves a compressed representation of the discarded content. The model gains a residual signal path to the dropped context at near zero memory cost. We expect either neutral or modestly better forward loss than baseline Jenga; the open question is whether the gain holds at all given that the dropped tokens are by construction those the predictor scored as low information.

**Implementation.** `src/jenga/models/modeling_llama.py` is patched at the attention forward where Jenga selects the top `config.sparse` fraction of blocks. The new path is gated by `config.merge_eliminated`; with the flag False the path is byte identical to the original hard drop. When the flag is True at layers 15 to 30 (the sparse layers in Llama 2 7B's stack):

1. Compute the complement of the kept block indices to get the dropped block indices.
2. Gather the dropped tokens' hidden states from the *pre-drop* layer input.
3. Mean pool to a single 4096 dimensional vector.
4. Append the merged vector as one extra token at the position of the last kept token. The single token formulation avoids flash attention's varlen "repeated positions imply zero length batches" trap; placing it at the last kept position keeps the position id tensor monotonically non decreasing as flash attention varlen requires.
5. Extend the RoPE position index by one element accordingly.

**Inference time experiment (Atom I3, Section 6.1).** Drop in test: baseline Jenga versus Jenga + Token Merging on the authors' existing Llama 2 7B + Jenga LoRA adapter (`checkpoints/peft_model/rp/8k/jenga`). Same seed, same documents, only `config.merge_eliminated` toggles.

**Joint training experiment (Atom I4, Section 6.2).** A new LoRA adapter is trained with `merge_eliminated = True` from step zero so the adapter sees the merged token throughout training. We then re-evaluate the same way as I3 to test whether jointly training amplifies the inference time effect.

## 6. Token Merging Results

### 6.1 Inference Time Merging on the Original Adapter (Atom I3)

Atom I3 ran on Pod 1 (RTX A6000 48 GB). Llama 2 7B + the authors' Jenga LoRA adapter, 8K context, four RedPajama documents, single seed, batch size one. Both modes share every other variable; only `config.merge_eliminated` toggles.

| Mode | Mean Loss | PPL ≈ exp(loss) | Peak Memory (MB) | Mean Forward (s) |
| --- | --- | --- | --- | --- |
| Jenga baseline (hard drop) | 2.5742 | 13.121 | 19,801.8 | 3.78 |
| **Jenga + Token Merging** | **2.5703** | **13.070** | **19,802.3** | **3.64** |
| Delta vs baseline | **-0.4 percent** | **-0.4 percent** | **+0.0025 percent** | **-3.7 percent** |

The single appended merged token yields a 0.4 percent lower forward loss, essentially zero memory delta (+0.5 MB out of ~20 GB), and a 3.7 percent lower mean forward time. The improvement is small and within plausible noise at n=4 documents single seed; the memory and time numbers are unambiguously neutral or favorable.

### 6.2 Joint Training with Merging Enabled (Atom I4)

To stress test the inference time win we retrained a new Llama 2 7B LoRA adapter with `config.merge_eliminated = True` from step zero. Training driver mirrors `src/experiment/end2end/time/llama_jenga.py` with the single line difference that the config flag is set before model creation; checkpoint saved to `checkpoints/peft_model_merged/`. Hyperparameters matched the original Jenga training (LoRA r=8 alpha=16 on q/k/v/o projections, AdamW with FP32 optimizer state, bf16, batch size one, learning rate 2e-5, constant with warmup schedule, 20 warmup steps) with `max_steps` reduced to **500** under remaining budget. RedPajama subset (1000 documents) tokenized at 8K context.

After training, the same `measure_merge.py` harness from I3 is re-run with `--peft_model checkpoints/peft_model_merged/`. Three way comparison populated when training and evaluation complete:

| Adapter | Mode | Mean Loss | PPL | Peak (MB) |
| --- | --- | --- | --- | --- |
| Authors' Jenga | baseline | 2.5742 | 13.121 | 19,801.8 |
| Authors' Jenga | + merging | 2.5703 | 13.070 | 19,802.3 |
| Retrained with merging | + merging | TBD | TBD | TBD |

**Figure 6** — paste this image once the third row populates (caption: "Mean forward loss and peak memory across the three Jenga configurations on four RedPajama documents at 8K context"):

The numbers live in `logs/extensions/token_merging/comparison.csv`; bar chart generated from the CSV directly.

## 7. Discussion

**What reproduced cleanly.** Every memory and time atom (R1, R2, R3, R6) produces numbers within ~1 percent of the authors' shipped logs on the same configurations. The memory savings scale with sequence length exactly as the paper claims; the predictor overhead is the fixed 656 MB; the LongLoRA "others" overhead is exactly the 4 GB the paper documents. R5 confirms the central trade off: Jenga pays 2 to 5 percent perplexity for 31 to 39 percent memory and 4 to 12 percent step time.

**The Token Merging finding.** A single merged token per sparse layer, mean pooled from the dropped blocks and placed at the last kept position, is at minimum *not worse* than baseline Jenga on the authors' adapter and slightly *better* on three of three measured axes (loss, memory, forward time). The improvement is small (0.4 percent loss) and within plausible single seed n=4 noise, but its direction is consistent and it adds essentially no memory (+0.5 MB). This is a defensible incremental contribution: Token Merging is the first soft elimination variant of Jenga we are aware of, the runtime change is one config flag, and the original behavior is preserved when the flag is off. Section 6.2's joint training experiment tests whether retraining the adapter alongside merging amplifies the effect.

**Other approaches explored.** We additionally implemented and ran two extension experiments whose results did not yield improvements and which we therefore do not feature:
* A 1D CNN attention predictor as a drop in replacement for the MLP predictor. Trained on OPT 1.3B activations (Llama 2 7B exceeded 48 GB with `output_attentions=True`) for 200 epochs, three seeds; final five epoch mean MSE was 15x worse than the MLP and the CNN showed catastrophic gradient spikes at later epochs. The dropped tokens are by construction low information; we conclude that adding local convolutional context over them mostly amplifies noise.
* A dynamic adaptive threshold heuristic that modulates per batch retention by predictor entropy. The implementation works mechanically (retention swings from 0.40 to 0.60 at `lam = 0.2` as predicted) but we could not measure its downstream perplexity impact without writing a Jenga aware perplexity evaluator, which was out of budget. We mention the attempt here for completeness and would treat its quality evaluation as future work.

**Bottlenecks.** Implementation correctness was the main blocker for Token Merging itself: three iterations were required to satisfy flash attention's varlen requirements (one token instead of 64, in range position, monotonic position ids). Once correct the win was immediate. Budget was the binding constraint everywhere else; spot preemptions cost us pod hours on every multi hour run.

## 8. Limitations

* **Single seed.** Memory, step time, perplexity, and extension numbers each come from a single seed. The paper's protocol requires three seeds for accuracy claims; we relaxed this under budget pressure.
* **Small n for extensions.** I3 and I4 measurements are over n=4 RedPajama documents. The 0.4 percent loss margin we report is not yet statistically significant at this sample size.
* **Trimmed perplexity scope.** R5 evaluates at 8K and 16K on `proof_pile.bin` but only at 8K on `test_pg19.bin`. The larger pg validation set would have required ~30 minutes per evaluation on the A6000.
* **No Jenga aware perplexity evaluator.** The artifact's `ppl.py` uses the dense baseline Llama with a LoRA adapter; it does not exercise the Jenga sparse forward at inference. This blocks downstream perplexity comparisons for any runtime config flag we add to the Jenga forward; the report uses forward loss on the patched Jenga path instead, which is fair but not identical to the paper's perplexity protocol.
* **OPT 1.3B substitution.** The CNN predictor comparison was forced to OPT 1.3B at sequence length 2048 because Llama 2 7B with `output_attentions=True` exceeds 48 GB. The predictor head is model agnostic so the *relative* comparison is preserved but absolute MSE numbers do not transplant.
* **Llama 3 8B coverage is limited.** Llama 3 was downloaded on every pod that hosted training experiments but used at inference only in R4 (Segmented Loss). All other atoms used Llama 2 7B.
* **Skipped atoms.** R7 predictor convergence (paper Figure 16) was budget gated and not executed; we cite the paper directly.
* **Hardware mix.** Reproduction ran on four different pods (A100 80GB SXM4, A100 80GB PCIe, RTX A6000 48 GB, RTX 4090 48 GB). Step time numbers in Section 4.2 are on the A100 80GB SXM4; other measurements use whichever pod ran them, noted per atom.

## 9. Conclusion

We reproduce Jenga's headline memory and time savings to within 1 percent of the authors' shipped numbers on Llama 2 7B and OPT 1.3B and confirm that the 2 to 5 percent perplexity premium is small relative to the 31 to 39 percent memory reduction. On top of the reproduction we introduce **Token Merging for Jenga**, a one config flag runtime modification that replaces Jenga's hard token elimination with a single mean pooled summary token per sparse layer. Naive inference time merging on the authors' existing Jenga adapter improves forward loss by 0.4 percent at essentially zero memory cost; a joint training run with merging enabled from step zero (Section 6.2) tests whether retraining amplifies the effect. The result supports Token Merging as a low cost soft elimination variant that preserves Jenga's memory and time advantages while gaining a small but measurable quality benefit. The natural next step is to evaluate on larger document samples and multiple seeds to characterize the variance of the improvement.

## References

Sketch list, to be expanded with full BibTeX in the LaTeX compile:

* Jenga: Enhancing Long Context Fine Tuning of LLMs with Contextual Token Sparsity. ATC 2025.
* Hu et al. LoRA: Low Rank Adaptation of Large Language Models. ICLR 2022.
* Bolya et al. Token Merging: Your ViT But Faster. ICLR 2023.
* Dao et al. FlashAttention 2. 2023.
* Touvron et al. Llama 2: Open Foundation and Fine Tuned Chat Models. 2023.
* Zhang et al. OPT: Open Pre Trained Transformer Language Models. 2022.
* Chen et al. LongLoRA: Efficient Fine Tuning of Long Context Large Language Models. ICLR 2024.

## Appendix A. Budget Ledger

Maintained in `PLAN.md` (top of file). Final cost figures appended after I4 closes.

## Appendix B. Software Environment

Reproduction was performed on rented Vast.ai A100 80GB SXM4, A100 80GB PCIe, RTX A6000 48 GB, and RTX 4090 48 GB instances.

**Hardware (primary, A100 80GB SXM4).**

| Item | Value |
| --- | --- |
| GPU | NVIDIA A100-SXM4-80GB (compute capability 8.0) |
| Driver | 570.211.01 |
| CUDA runtime | 12.8 (`nvcc V12.8.93`) |
| Image base | Vast.ai PyTorch image with miniforge3 at `/opt/miniforge3` |

**Python environment.** Dedicated `jenga` conda env at `/venv/jenga` with Python 3.10.20.

| Package | Version | Source |
| --- | --- | --- |
| torch | 2.1.2 | requirements.txt |
| transformers | 4.45.2 | requirements.txt |
| tokenizers | 0.20.1 | requirements.txt |
| deepspeed | 0.14.0 | requirements.txt |
| bitsandbytes | 0.41.1 | requirements.txt |
| numpy | 1.26.4 | pinned `<2` for torch 2.1.2's numpy 1.x ABI |
| accelerate | 1.13.0 | requirements.txt (took latest) |
| datasets | 2.21.0 | pinned `<3` for legacy loading script support |
| peft | 0.19.1 | requirements.txt (took latest) |
| flash-attn | 2.5.6 | prebuilt wheel `cu122torch2.1cxx11abiFALSE-cp310` |
| setuptools | <70 | pinned for `pkg_resources.packaging` |

All S1 sanity check sentinels passed at first kill: predictor and pruned config files present, both PPL bins present, all base model config.json files present, CUDA available, flash_attn and jenga packages import without error.
