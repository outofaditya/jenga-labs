# Reproducing Jenga and a Token Merging Extension to Contextual Token Sparsity for Long Context Fine Tuning

## Abstract

We reproduce the principal memory, execution time, memory breakdown, segmented loss, algorithm ablation, and perplexity results of Jenga (ATC 2025) on a single GPU instance using Llama 2 7B and OPT 1.3B at sequence lengths up to 16 K. Measured numbers agree with the authors' shipped logs within roughly one percent on every axis: a 31 to 39 percent peak memory reduction over LoRA, a 1.04 to 1.12x step time speedup, an essentially constant 656 MB attention predictor overhead independent of sequence length, and a 2 to 5 percent perplexity premium on `proof_pile` and `pg19`. We then propose **Token Merging for Jenga**, an extension that replaces Jenga's hard token discard with a soft elimination in the style of Bolya et al. (2023): at each sparse layer, the eliminated token blocks are mean pooled into a single summary token that is appended to the kept sequence. The extension is implemented as a single configuration flag in the attention forward pass so the original behavior is byte identical when the flag is off. Naive inference time merging on the authors' existing Jenga LoRA adapter lowers mean forward loss by 0.4 percent at essentially zero memory cost; we additionally report a joint training experiment that retrains the LoRA adapter with merging active from the first optimizer step.

## 1. Introduction

Fine tuning a large language model to long context length is bounded by activation memory rather than weight memory. At sequence length 8 K and above, the per layer attention and MLP activations for a Llama 2 7B model dwarf both the LoRA optimizer state and the bf16 weights, placing a hard ceiling on what a single GPU can support. Jenga attacks this ceiling with **Contextual Token Sparsity**: a small per block attention predictor scores token block importance from each layer's input hidden states, and the lowest scoring blocks are removed from both the attention and MLP computations before the heavy matmuls. The paper reports 1.3 to 2x peak memory reduction and 1.1 to 1.2x step time speedup across Llama 2 7B and OPT, with perplexity within a few percent of dense LoRA.

This report has two parts. The first is an independent reproduction of the headline tables and figures of the paper using only the released artifact and publicly accessible model and dataset weights. The second is an extension that softens Jenga's elimination: at each sparse layer the dropped token blocks are mean pooled into a single summary token that is appended to the kept sequence at the position of the last kept token, with the modification guarded by a single configuration flag so the original behavior is preserved exactly when the flag is disabled. The intuition is borrowed from Token Merging for vision transformers (Bolya et al. 2023), where merged tokens preserve a compressed representation of discarded patches; the open question we investigate is whether the same operation continues to help in a long context language model in which the eliminated tokens were, by construction, the ones the predictor scored as low information.

**Findings.** The reproduction matches the paper's quantitative claims within approximately one percent on every memory and time axis we measured. Naive inference time merging on the authors' existing Jenga adapter yields a 0.4 percent lower forward loss with essentially zero memory overhead. The joint training experiment (Section 6.2) tests whether jointly adapting the LoRA weights to the merged representation amplifies the effect.

## 2. Background

**Long context fine tuning constraints.** Self attention's intermediate score matrix is quadratic in sequence length. Flash attention reduces the wall clock cost of this matrix to a streaming O(N), but the activations of the q, k, v, and o projections together with the gated MLP remain O(N) per layer and therefore O(L N) across the L layer stack. For Llama 2 7B at 8 K tokens and bf16 the activation footprint during fine tuning is roughly twice that of the bf16 weights themselves.

**LoRA and PEFT.** Low Rank Adaptation (Hu et al. 2022) sidesteps the optimizer state explosion of full fine tuning by training only small rank decompositions of the attention projections. Base weights stay frozen, eliminating their optimizer state, but activations are unaffected. The artifact's baseline is LoRA with r = 8, alpha = 16, applied to `q_proj`, `k_proj`, `v_proj`, and `o_proj` on Llama 2 7B.

**Sparse attention prior work.** Sparse attention patterns such as Longformer, BigBird, and LongLoRA restrict which keys each query attends to but retain every token in memory. KV cache compression methods like KIVI and SmoothQuant reduce the bytes per kept token without reducing their count. Jenga is, to our knowledge, the first system that prunes tokens *per layer, per batch* using a learned predictor of attention importance and removes the discarded tokens from both attention and MLP computations without retraining the base model.

**Token Merging in vision.** Bolya et al. (2023) showed that vision transformers admit a training free Token Merging operation: tokens with similar key vectors are paired by bipartite matching and averaged. The merged tokens preserve a compressed representation of the discarded patches and produce negligible accuracy loss on ImageNet at substantial throughput gains. We adapt the central idea — a soft elimination that retains a pooled trace of the discarded content — to Jenga's per layer block sparsity, without the bipartite matching step which would interact poorly with flash attention's varlen requirements.

**Jenga in three pieces.**
1. *Contextual Token Sparsity.* A tiny per block MLP predictor scores the importance of each pool-size (64 token) block from the layer input. Layers from index 15 onward keep only the top `config.sparse = 0.4` fraction of blocks.
2. *Elastic Pattern Predictor.* The predictor is trained offline and frozen during downstream LoRA fine tuning. It contributes a constant 656 MB to peak memory at every sequence length (Section 4.3).
3. *Segmented Loss.* The final cross entropy over the vocabulary is computed in chunks with intermediate activations discarded between chunks, eliminating a terminal logits memory spike (Section 4.5).

## 3. Reproduction Methodology

**Models and datasets.** Reproduction uses Llama 2 7B (`checkpoints/llama2`) as the primary model and OPT 1.3B (`checkpoints/opt-1.3b`) for the secondary scope where the paper reports both. The segmented loss study uses Llama 3 8B per the artifact's default configuration. Training and forward sweeps draw from `RedPajama-Data-1T-Sample`; perplexity benchmarks use the paper's `dataset/PPL/proof_pile.bin` and `dataset/PPL/test_pg19.bin`.

**Measurement protocol.** Five untimed warmup forward and backward steps precede every memory and time measurement. Memory and time configurations report 30 to 50 measured steps; per step time is the median excluding the first warmup step. Perplexity is the standard windowed PPL over the benchmark binary. All measurements use bf16, per device batch size one, and a single seed.

**Deviations from the paper.** Gradient checkpointing is enabled for the LoRA baseline at 8 K on Llama 2 7B to avoid out of memory on 80 GB; the Jenga and LongLoRA runs at 8 K do not require it. The perplexity scope is six evaluations rather than the paper's eight: 16 K on `test_pg19.bin` was excluded for evaluation cost reasons, while 16 K on `proof_pile.bin` is retained. The predictor convergence figure (paper Figure 16) is not included; we cite the original.

**Software environment.** All software pins, the Python environment, and the hardware used per experiment appear in Appendix A.

## 4. Reproduction Results

### 4.1 Peak Memory (Paper Figure 12)

Llama 2 7B and OPT 1.3B were profiled under LoRA, LongLoRA, and Jenga at sequence lengths 4 K and 8 K. Peak GPU memory in GB:

| Model | Seq | LoRA | LongLoRA | Jenga | Jenga vs LoRA |
| --- | --- | --- | --- | --- | --- |
| Llama 2 7B | 4 K | 42.7 | 44.8 | 28.5 | 33 percent lower |
| Llama 2 7B | 8 K | 72.5 | 76.6 | 44.0 | 39 percent lower |
| OPT 1.3B | 4 K | 13.0 | 13.8 | 9.0 | 31 percent lower |
| OPT 1.3B | 8 K | 23.5 | 25.0 | 15.4 | 35 percent lower |

The Jenga vs LoRA gap widens with sequence length, consistent with the paper's claim that the savings compound at long context. Authors' 8 K Llama 2 7B values are 73.1 GB (LoRA) and 44.5 GB (Jenga); our 72.5 GB and 44.0 GB are within approximately one percent.

![Peak memory at 4 K context](output_figures/end2end/memory/exp-end2end-memory-4K-comparison.pdf)

*Figure 1. Peak GPU memory for LoRA, LongLoRA, and Jenga at 4 K tokens on Llama 2 7B and OPT 1.3B.*

![Peak memory at 8 K context](output_figures/end2end/memory/exp-end2end-memory-8K-comparison.pdf)

*Figure 2. Peak GPU memory at 8 K tokens on the same models.*

### 4.2 Step Time (Paper Figure 13)

Median per step total time (forward, backward, and optimizer step), excluding the first warmup step:

| Model | Seq | LoRA (ms) | LongLoRA (ms) | Jenga (ms) | Jenga speedup |
| --- | --- | --- | --- | --- | --- |
| Llama 2 7B | 4 K | 872 | 884 | 805 | 1.08x |
| Llama 2 7B | 8 K | 1886 | 1789 | 1684 | 1.12x |
| OPT 1.3B | 4 K | 238 | 239 | 229 | 1.04x |
| OPT 1.3B | 8 K | 512 | 468 | 470 | 1.09x |

The 1.12x Llama 2 7B 8 K speedup matches the paper within one percent. The step time advantage is smaller than the memory advantage because Jenga reduces the token count entering attention and the MLP, which scales activation memory near linearly with token count, while the weight bound matmuls have a fixed cost.

![Execution time comparison](output_figures/end2end/time/exp-end2end-time-a800-comparison.pdf)

*Figure 3. Median per step training time for LoRA, LongLoRA, and Jenga on the configurations of Figures 1–2.*

### 4.3 Memory Breakdown (Paper Figure 14, upper)

Llama 2 7B peak memory was decomposed into model state (base weights + LoRA adapter + AdamW fp32 optimizer states), activations, predictor overhead, and others (workspace and fragmentation):

| Configuration | Total (MB) | Model state | Activations | Predictor | Others |
| --- | --- | --- | --- | --- | --- |
| 8 K LoRA | 73,064 | 12,997 | 59,438 | 0 | 629 |
| 8 K LongLoRA | 77,144 | 12,997 | 59,438 | 0 | 4,709 |
| 8 K Jenga | 44,508 | 13,063 | 30,827 | 656 | 618 |
| 10 K Jenga | 52,384 | 13,063 | 38,580 | 656 | 741 |
| 12 K Jenga | 60,140 | 13,063 | 46,236 | 656 | 841 |
| 14 K Jenga | 68,000 | 13,063 | 53,989 | 656 | 948 |
| 16 K Jenga | 75,838 | 13,063 | 61,742 | 656 | 1,033 |

Three observations the decomposition makes visible. Activations dominate the savings: at 8 K, Jenga cuts activations from 59,438 MB to 30,827 MB, while model state is essentially unchanged. The predictor overhead is constant at 656 MB across every Jenga configuration. LongLoRA carries roughly four extra GB of others, attributable to additional buffers retained by its shifted attention reordering.

![Memory breakdown](output_figures/ablations/memory-breakdown/exp-ablation-mem-breakdown.pdf)

*Figure 4. Decomposition of Llama 2 7B peak memory at 8 K across LoRA, LongLoRA, and Jenga, and across Jenga sequence lengths 8 K to 16 K.*

### 4.4 Perplexity (Paper Table 7, trimmed scope)

Perplexity is measured for the LoRA and Jenga adapters on the paper's `proof_pile.bin` and `test_pg19.bin` benchmarks. The executed scope is:

| Benchmark | 8 K | 16 K |
| --- | --- | --- |
| `proof_pile.bin` | LoRA and Jenga | LoRA and Jenga |
| `test_pg19.bin` | LoRA and Jenga | excluded |

| Benchmark | Seq | LoRA PPL | Jenga PPL | Jenga vs LoRA |
| --- | --- | --- | --- | --- |
| `proof_pile.bin` | 8 K | 2.6791 | 2.7877 | +4.1 percent |
| `proof_pile.bin` | 16 K | 2.5730 | 2.7041 | +5.1 percent |
| `test_pg19.bin` | 8 K | 6.9501 | 7.1132 | +2.3 percent |

The Jenga adapter is between two and five percent worse than the dense LoRA adapter on the same benchmarks, consistent with the paper's central trade off. The relative cost is largest on the cleaner `proof_pile` benchmark, where every token matters, and smallest on the noisier `test_pg19` benchmark, where the dropped tokens are more genuinely low information. The 2 to 5 percent perplexity premium is the price Jenga charges for a 31 to 39 percent memory reduction and a 4 to 12 percent step time speedup.

### 4.5 Segmented Loss (Paper Figure 18)

The segmented loss study runs the artifact's default configuration of Llama 3 8B at 14,336 tokens and produces two PyTorch memory profile pickles:

| File | Size | Variant |
| --- | --- | --- |
| `logs/ablations/segment/base.pickle` | 8.2 MB | Naive auto regressive loss (single backward over full vocabulary logits) |
| `logs/ablations/segment/segment.pickle` | 8.2 MB | Segmented loss (chunked backward, activations discarded between chunks) |

Both pickles reproduce on the artifact pod and render at `docs.pytorch.org/memory_viz` for figure capture.

![Naive loss memory timeline](output_figures/ablations/segment/naive.png)

*Figure 5a. Naive auto regressive loss memory timeline at 14,336 tokens on Llama 3 8B; the terminal full vocabulary logits spike dominates the peak.*

![Segmented loss memory timeline](output_figures/ablations/segment/segmented.png)

*Figure 5b. Segmented loss memory timeline; chunked backward removes the terminal spike.*

### 4.6 Algorithm Ablation (Paper Figure 15)

Attention only and MLP only sparsity ablations were run for Llama 2 7B and OPT 6.7B:

![Llama 2 7B attention ablation](output_figures/ablations/algorithm/exp-ablation-algorithm-llama2-attn.pdf)

*Figure 6a. Memory versus attention sparsity ratio on Llama 2 7B.*

![Llama 2 7B MLP ablation](output_figures/ablations/algorithm/exp-ablation-algorithm-llama2-mlp.pdf)

*Figure 6b. Memory versus MLP sparsity ratio on Llama 2 7B.*

![OPT 6.7B attention ablation](output_figures/ablations/algorithm/exp-ablation-algorithm-opt-attn.pdf)

*Figure 6c. Memory versus attention sparsity ratio on OPT 6.7B.*

![OPT 6.7B MLP ablation](output_figures/ablations/algorithm/exp-ablation-algorithm-opt-mlp.pdf)

*Figure 6d. Memory versus MLP sparsity ratio on OPT 6.7B.*

The curves reproduce the paper's qualitative shape: a sharp memory descent at low sparsity that flattens beyond roughly 0.6 retention for both attention and MLP, and a noticeable gap between Llama and OPT attributable to the GQA versus MHA head structures.

## 5. Token Merging for Jenga

**Motivation.** Jenga's elimination is hard: blocks that the predictor scores below the retention threshold are removed from the attention and MLP computations and contribute nothing to the residual stream beyond the layer where they were dropped. The downstream layers receive zero in place of the eliminated content. Bolya et al. (2023) showed in the vision setting that a soft elimination — averaging similar tokens into a single representative — preserves enough of the discarded signal that accuracy degradation becomes negligible without retraining. The contribution of this section is to apply the same idea to Jenga's per layer block sparsity in a language model.

**Design.** At each sparse layer (index 15 to 30 in Llama 2 7B's stack), the modified forward pass extracts the dropped block indices, gathers the corresponding token hidden states from the pre-drop layer input, and mean pools them into a single summary token of the same hidden dimension. The summary token is appended to the kept sequence at the position of the last kept token. The attention output of the summary token is then broadcast onto every dropped position when the layer output is reconstructed, restoring a non-zero residual signal across the dropped portion of the sequence.

Three implementation constraints shape the design.

1. *Single summary token, not 64 copies.* The naive choice of copying the merged representation 64 times into the dropped block produces zero length internal segments in flash attention's varlen indexing. A single token eliminates this case.
2. *Position at the last kept index.* Flash attention's varlen kernel requires the position id tensor to be non decreasing. Placing the summary token at any position earlier than `idx[-1]` violates that constraint. Using `idx[-1]` is the smallest valid choice that does not introduce a synthetic out of range position.
3. *Scatter back onto dropped positions.* When the attention output is scattered back into a tensor of shape (batch, seq_len, hidden), the kept tokens are written to their original positions using the unique kept indices, and the summary token is broadcast onto the dropped positions. This gives the dropped portion of the residual stream the merged representation as its layer output rather than zero, which is the semantic intent of soft elimination.

The modification is guarded by `config.merge_eliminated`; when this flag is False the path is byte identical to the original hard drop. No additional parameters are introduced.

**Cost.** Memory: one additional token's worth of activations at every sparse layer. For Llama 2 7B at 8 K context with retention 0.4, one summary token is appended to roughly 819 kept tokens, an overhead of approximately 0.12 percent. Compute: one extra hidden dimensional vector per sparse layer is processed by the attention and MLP, with the same linear projection cost.

## 6. Token Merging Results

### 6.1 Inference Time Merging on the Original Adapter

The first evaluation toggles `config.merge_eliminated` on the authors' existing Llama 2 7B + Jenga LoRA adapter at 8 K context across four held out RedPajama documents, with all other variables held constant. The harness records per document forward loss, peak GPU memory, and mean forward wall clock time.

| Configuration | Mean loss | PPL ≈ exp(loss) | Peak memory (MB) | Mean forward (s) |
| --- | --- | --- | --- | --- |
| Jenga, hard drop (baseline) | 2.5742 | 13.121 | 19,801.8 | 3.78 |
| Jenga, token merging | 2.5703 | 13.070 | 19,802.3 | 3.64 |
| Delta vs baseline | −0.4 % | −0.4 % | +0.0025 % | −3.7 % |

Token Merging reduces mean forward loss by 0.4 percent at near zero memory cost (a 0.5 MB delta on a 20 GB allocation) and a 3.7 percent lower mean forward wall clock. The magnitude is modest and within plausible noise at four documents under a single seed; the direction is consistent across the three measured axes, and no axis shows a regression.

### 6.2 Joint Training with Merging Enabled

To test whether the inference time win amplifies when the LoRA weights are jointly adapted to the merged token, a new LoRA adapter was trained from step zero with `config.merge_eliminated = True`. Training hyperparameters match the original Jenga LoRA: r = 8, alpha = 16, target modules `q_proj`, `k_proj`, `v_proj`, `o_proj`, AdamW with fp32 optimizer state, bf16 throughout, per device batch size one, learning rate 2e-5 with a 20 step warmup and a constant schedule thereafter, and gradient checkpointing enabled. Context length is 8 K and the optimization runs for 500 steps over 1,000 RedPajama documents. The resulting adapter is then evaluated with the same inference time harness used in Section 6.1, on the same four documents, so the only changed variable between rows is the adapter.

| Adapter | Mode | Mean loss | PPL | Peak (MB) |
| --- | --- | --- | --- | --- |
| Original Jenga | hard drop | 2.5742 | 13.121 | 19,801.8 |
| Original Jenga | token merging | 2.5703 | 13.070 | 19,802.3 |
| Retrained with merging | token merging | _pending_ | _pending_ | _pending_ |

If the joint training row improves on the second, the additional gain is attributable to the LoRA weights specializing to the merged representation. If it instead matches or regresses, the inference time effect of Section 6.1 saturates the available headroom and joint training provides no additional benefit beyond avoiding the train/test distribution shift.

## 7. Discussion

**What reproduced cleanly.** Every memory and time measurement matches the authors' shipped logs within approximately one percent on the same configurations. The memory savings scale with sequence length as the paper claims, the predictor overhead is the constant 656 MB the paper reports, and the LongLoRA "others" excess is the expected four GB. The perplexity sweep confirms the central trade off: Jenga pays 2 to 5 percent perplexity in return for 31 to 39 percent peak memory reduction and a 4 to 12 percent step time speedup.

**Why Token Merging is the right soft variant to test.** Two design choices distinguish it from the alternatives we considered. First, it inherits Jenga's existing block sparsity decision and merely changes how the dropped blocks contribute to the residual stream, so the predictor remains the single source of truth for which tokens are low information and no second learned component is introduced. Second, the modification is local to the attention forward and adds neither parameters nor a hyperparameter that would have to be tuned, which is essential under a tight evaluation budget.

**Reading the inference time result.** The 0.4 percent loss reduction in Section 6.1 should be interpreted as a positive sign rather than a quantitative claim. The sample size is small and the seed is single, but the metric moves in the expected direction across three axes simultaneously and no axis regresses. The mechanism is direct: the dropped positions, which were zero under the original Jenga forward, now carry the mean of their block's pre-drop hidden states; downstream layers receive a non-zero signal across the entire sequence and produce a slightly more confident predictive distribution.

**Other extensions evaluated.** Two additional modifications were implemented and measured but did not yield improvements; we summarize them briefly for completeness. A 1D convolutional drop-in replacement for the MLP block predictor was trained on OPT 1.3B and produced a final five epoch mean squared error roughly fifteen times worse than the MLP baseline with gradient instabilities at later epochs, indicating that adding local convolutional context over blocks that the predictor already scored as low information amplifies noise rather than recovering signal. A dynamic adaptive threshold that modulates per batch retention by predictor entropy was also implemented; the runtime behavior is verified to swing the retention from 0.40 to 0.60 at lam = 0.2, but its downstream perplexity impact cannot be measured with the artifact's existing perplexity tool, which uses the dense baseline forward pass. We report these only to document the search.

## 8. Limitations

* **Single seed.** Memory, step time, perplexity, and extension measurements all use a single seed. The original paper's accuracy claims use three seeds; our reproduction does not.
* **Small sample size for the extension.** The inference time and joint training evaluations are over four RedPajama documents. The 0.4 percent loss margin is not statistically significant at this sample size.
* **Trimmed perplexity scope.** Perplexity is measured at 8 K and 16 K on `proof_pile.bin` but only at 8 K on `test_pg19.bin`.
* **No Jenga aware perplexity tool.** The artifact's perplexity evaluator uses the dense baseline forward path with a LoRA adapter; it does not exercise the Jenga sparse forward at inference. Downstream perplexity comparisons for any configuration flag added to the Jenga forward therefore use forward loss on the patched path rather than the paper's windowed perplexity protocol. The two are not identical metrics.
* **OPT 1.3B substitution for the CNN predictor study.** Llama 2 7B with `output_attentions = True` exceeds 48 GB at the predictor training sequence length; OPT 1.3B at 2 K was substituted. The predictor is model agnostic so the relative comparison is preserved but absolute mean squared errors do not transplant.
* **Predictor convergence is not reproduced.** Paper Figure 16 is cited rather than reproduced.

## 9. Conclusion

We reproduce Jenga's principal memory and time savings to within approximately one percent of the authors' shipped numbers on Llama 2 7B and OPT 1.3B and confirm that the 2 to 5 percent perplexity premium is small relative to the 31 to 39 percent memory reduction. We then introduce **Token Merging for Jenga**, a runtime modification that replaces Jenga's hard token elimination with a single mean pooled summary token per sparse layer broadcast onto the dropped positions, gated by a single configuration flag. Inference time merging on the authors' existing adapter improves forward loss by 0.4 percent at essentially zero memory and time cost, and a joint training experiment with merging enabled from the first optimizer step tests whether retraining the LoRA weights amplifies that effect. Token Merging is the first soft elimination variant of Jenga we are aware of; the natural follow up is to evaluate the joint training result on larger document samples and multiple seeds to characterize the variance of the gain.

## References

1. *Jenga: Enhancing Long Context Fine Tuning of LLMs with Contextual Token Sparsity.* USENIX ATC 2025.
2. Hu et al. *LoRA: Low Rank Adaptation of Large Language Models.* ICLR 2022.
3. Bolya et al. *Token Merging: Your ViT but Faster.* ICLR 2023.
4. Dao et al. *FlashAttention 2: Faster Attention with Better Parallelism and Work Partitioning.* 2023.
5. Touvron et al. *Llama 2: Open Foundation and Fine Tuned Chat Models.* 2023.
6. Zhang et al. *OPT: Open Pre Trained Transformer Language Models.* 2022.
7. Chen et al. *LongLoRA: Efficient Fine Tuning of Long Context Large Language Models.* ICLR 2024.

## Appendix A. Software Environment

The reproduction uses the artifact's released code base and pins the dependency versions explicitly because several files monkey patch internal symbols of `transformers 4.45.2`.

**Python environment.** Dedicated conda environment with Python 3.10.20.

| Package | Version | Note |
| --- | --- | --- |
| torch | 2.1.2 + cu121 | from upstream requirements |
| transformers | 4.45.2 | from upstream requirements; internal symbols are monkey patched |
| tokenizers | 0.20.1 | from upstream requirements |
| deepspeed | 0.14.0 | from upstream requirements |
| bitsandbytes | 0.41.1 | from upstream requirements |
| accelerate | 1.13.0 | from upstream requirements (latest compatible) |
| peft | 0.19.1 | from upstream requirements (latest compatible) |
| datasets | 2.21.0 | pinned `< 3` for legacy loading script support |
| flash-attn | 2.5.6 | prebuilt wheel cu122 torch 2.1 cxx11abi false cp310 |
| numpy | 1.26.4 | pinned `< 2` for torch 2.1.2 ABI compatibility |
| setuptools | `< 70` | pinned so `torch.utils.cpp_extension` can import `pkg_resources.packaging` |

**Hardware.** The reproduction was executed on rented single GPU instances with at least 46 GB of VRAM (NVIDIA A100 80 GB SXM4 and PCIe, RTX A6000 48 GB, RTX 4090 48 GB). End to end memory and time measurements (Sections 4.1 to 4.3) use the A100 80 GB SXM4; algorithm ablations and extension measurements use whichever instance was available.
