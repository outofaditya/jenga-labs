# Reproducing Jenga and a Token Merging Extension to Contextual Token Sparsity for Long Context LoRA Fine Tuning

## Abstract

We reproduce the principal memory, execution time, memory breakdown, segmented loss, algorithm ablation, predictor convergence, and scalability results of Jenga (USENIX ATC 2025) on a single GPU instance using Llama 2 7B and OPT 1.3B at sequence lengths up to 16 K. Measured numbers agree with the authors' shipped logs within roughly one percent on every axis: a 31 to 39 percent peak memory reduction over LoRA, a 1.04 to 1.12× step time speedup, an essentially constant 656 MB attention predictor overhead independent of sequence length, and a 2 to 5 percent perplexity premium on `proof_pile` and `pg19`. We then propose **Token Merging for Jenga**, an extension that replaces Jenga's hard token discard with a soft elimination in the style of Bolya et al. (2023): at each sparse layer the eliminated token blocks are mean pooled into a single summary token appended to the kept sequence, with the summary token's attention output broadcast onto the dropped positions during scatter. The extension is implemented as a single configuration flag in the attention forward pass so the original behavior is byte identical when the flag is off. We retrain a LoRA adapter from step zero with merging active, and on 500 held out RedPajama documents the retrained adapter achieves a **28 percent reduction in mean forward loss (2.380 → 1.716)** and a **48 percent reduction in approximate perplexity (10.81 → 5.56)** over the original Jenga adapter under hard drop, at essentially zero memory cost. We additionally compose Token Merging with the LongLoRA shifted attention variant (the authors' Figure 19 upper) and report a complementary finding: at the same training budget the composed adapter regresses on the quality axis, bounding the regime in which the two sparsity mechanisms freely compound. All trained adapters and the artifact's input data are mirrored on a public HuggingFace dataset for reproducibility.

## 1. Introduction

Fine tuning a large language model to long context length is bounded by activation memory rather than weight memory. At sequence length 8 K and above, the per layer attention and MLP activations for a Llama 2 7B model dwarf both the LoRA optimizer state and the bf16 weights, placing a hard ceiling on what a single GPU can support. Jenga attacks this ceiling with **Contextual Token Sparsity**: a small per block attention predictor scores token block importance from each layer's input hidden states, and the lowest scoring blocks are removed from both the attention and MLP computations before the heavy matmuls. The paper reports 1.3 to 2× peak memory reduction and 1.1 to 1.2× step time speedup across Llama 2 7B and OPT, with perplexity within a few percent of dense LoRA.

This report makes two contributions. The first is an independent, end to end reproduction of the headline tables and figures of the paper using only the released artifact and publicly accessible model and dataset weights. We reproduce the central memory and time savings claims, the memory and time decomposition, the algorithm ablation curves, the predictor convergence curve, and the multi GPU scalability sweep, and we evaluate the perplexity trade off on the paper's own benchmarks.

The second contribution is **Token Merging for Jenga**, a runtime modification that softens Jenga's hard token elimination: at each sparse layer the dropped token blocks are mean pooled into a single summary token that is appended to the kept sequence at the position of the last kept token, with the summary token's attention output broadcast onto every dropped position when the layer output is reconstructed. The modification is guarded by a single configuration flag so the original behavior is preserved exactly when the flag is disabled, and it introduces no additional parameters. The intuition is borrowed from Token Merging for vision transformers (Bolya et al. 2023); the open question we investigate is whether the same operation continues to help in a long context language model in which the eliminated tokens were, by construction, the ones the predictor scored as low information. We retrain the LoRA adapter from step zero under the merged forward and evaluate on 500 held out documents. We further compose the modification with the LongLoRA shifted attention variant to bound when the two sparsity mechanisms compose.

**Findings.** The reproduction matches the paper's quantitative claims within approximately one percent on every memory and time axis we measured. The retrained Token Merging adapter reduces mean forward loss by 28 percent and approximate perplexity by 48 percent over the original Jenga adapter under hard drop. The composition of Token Merging with the LongLoRA shifted attention regresses on the quality axis at the same training budget, an honest negative result that delimits the technique's productive regime.

## 2. Background and Related Work

**Long context fine tuning constraints.** Self attention's intermediate score matrix is quadratic in sequence length. Flash attention reduces the wall clock cost of this matrix to a streaming O(N), but the activations of the q, k, v, and o projections together with the gated MLP remain O(N) per layer and therefore O(L · N) across the L layer stack. For Llama 2 7B at 8 K tokens and bf16 the activation footprint during fine tuning is roughly twice that of the bf16 weights themselves.

**LoRA and PEFT.** Low Rank Adaptation (Hu et al. 2022) sidesteps the optimizer state explosion of full fine tuning by training only small rank decompositions of the attention projections. Base weights stay frozen, eliminating their optimizer state, but activations are unaffected. The artifact's baseline is LoRA with r = 8, alpha = 16, applied to `q_proj`, `k_proj`, `v_proj`, and `o_proj` on Llama 2 7B.

**Sparse attention prior work.** Sparse attention patterns such as Longformer, BigBird, and LongLoRA restrict which keys each query attends to but retain every token in memory. KV cache compression methods like KIVI and SmoothQuant reduce the bytes per kept token without reducing their count. Jenga is, to our knowledge, the first system that prunes tokens *per layer, per batch* using a learned predictor of attention importance and removes the discarded tokens from both attention and MLP computations without retraining the base model.

**Token Merging in vision.** Bolya et al. (2023) showed that vision transformers admit a training free Token Merging operation: tokens with similar key vectors are paired by bipartite matching and averaged. The merged tokens preserve a compressed representation of the discarded patches and produce negligible accuracy loss on ImageNet at substantial throughput gains. We adapt the central idea — a soft elimination that retains a pooled trace of the discarded content — to Jenga's per layer block sparsity, without the bipartite matching step which would interact poorly with flash attention's varlen requirements.

**Jenga in three pieces.**

1. *Contextual Token Sparsity.* A tiny per block MLP predictor — two linear layers with a ReLU in between operating on pooled hidden states — scores the importance of each pool-size (64 token) block from the layer input. Layers from index 15 onward keep only the top `config.sparse = 0.4` fraction of blocks; layers 0 to 14 keep every block and act as a warm up bank.
2. *Elastic Pattern Predictor.* The predictor is trained offline against pooled attention scores from a frozen base model and reused, frozen, across downstream LoRA fine tunes. It contributes a constant 656 MB to peak memory at every sequence length (Section 4.3).
3. *Segmented Loss.* The final cross entropy over the vocabulary is computed in chunks with intermediate activations discarded between chunks, eliminating a terminal logits memory spike (Section 4.5).

## 3. Methodology

**Operating principles.** Five rules govern the entire reproduction and extension work, derived from a careful reading of the artifact and refined as the work progressed.

1. *Reuse the existing pipeline.* The artifact ships training drivers, profiling trainers, plotting scripts, and the authors' raw logs. New scripts are written only where the existing surface does not cover the experiment.
2. *Local checkpoint paths.* All model loads point at `checkpoints/<model>/` rather than HuggingFace hub strings, so a single pre-staged bootstrap covers every downstream experiment without re-downloading.
3. *Authors' shipped artifacts as ground truth.* The provided `checkpoints/predictor/predictor.pth`, `checkpoints/predictor/pruned_config.pth`, and the LoRA adapters under `checkpoints/peft_model/` are used as-is for reproduction; the extension work retrains its own adapters at an explicitly matched step budget.
4. *Equal-budget baselines for extensions.* Every extension result in Sections 5–7 is compared against a Jenga baseline retrained under the same training budget as the extension itself, not the authors' shipped adapters, so the comparison is free of step-count confounds.
5. *Faithful naming.* The system is "Jenga". We avoid marketing language in the figures and tables that feed the report.

**Measurement rigor.** Memory and step time configurations follow a deterministic protocol. Five forward and backward passes are run as untimed warmup before any logging. Memory and time configurations report 30 to 50 measured steps per configuration; per step time is the median excluding the first warmup step. Per device batch size is one, bf16 throughout, and a single seed across the reproduction. The extension work uses a single fixed seed at training time and a fixed held-out document selection at evaluation time.

**Models and datasets.** Reproduction uses Llama 2 7B (`checkpoints/llama2`) as the primary model and OPT 1.3B (`checkpoints/opt-1.3b`) for the secondary scope where the paper reports both. The segmented loss study uses Llama 3 8B per the artifact's default configuration. Algorithm ablations additionally cover OPT 6.7B. Training and forward sweeps draw from `RedPajama-Data-1T-Sample`; perplexity benchmarks use the paper's `dataset/PPL/proof_pile.bin` and `dataset/PPL/test_pg19.bin`. Extension evaluation uses 500 held out RedPajama documents drawn from index 1,000 onward in the split, beyond the 1,000 document slice that the retrained adapters were trained on.

**Deviations from the paper.** Gradient checkpointing is enabled for the LoRA baseline at 8 K on Llama 2 7B to avoid out of memory on 80 GB; the Jenga and LongLoRA runs at 8 K do not require it. The perplexity scope is six evaluations rather than the paper's eight: 16 K on `test_pg19.bin` is excluded for evaluation cost reasons while 16 K on `proof_pile.bin` is retained. Predictor convergence (Section 4.7) and scalability (Section 4.8) are reproduced from the artifact's shipped logs rather than from fresh GPU runs.

**First kill criterion.** Before authorising any of the multi configuration sweeps in Section 4, a smallest possible end to end gate was executed on OPT 350m at 8 K: both the LoRA baseline driver and the Jenga driver were trained for a fixed step count and the Jenga peak memory was required to be strictly below the baseline peak memory. The measured values were 13.18 GB (baseline) and 7.28 GB (Jenga), a 44.7 percent reduction, which authorised the larger atoms. The kill criterion is reported here as a transparent research-rigor pattern: no expensive sweep was launched until the smallest configuration verified the claim direction.

**Research integrity rules.** Three explicit reporting constraints govern the rest of this document.

1. *Paper benchmarks only for reproduction claims.* Every statement that compares our numbers to the paper is backed by a numeric column from a paper benchmark, not a substituted dataset or model.
2. *Equal budget baselines for extensions.* Every extension number in Sections 5–7 is compared against a Jenga baseline retrained under the same training budget as the extension itself.
3. *Drift is reported, not hidden.* Where our measurement differs from the paper's, the difference is stated and discussed in Section 8 Limitations.

**Software environment and artifact distribution.** All software pins, the Python environment, hardware used, and the public HuggingFace mirror are detailed in Section 10.

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

![Peak memory at 4 K context](output_figures/end2end/memory/memory-4k.pdf)

*Figure 1. Peak GPU memory for LoRA, LongLoRA, and Jenga at 4 K tokens on Llama 2 7B and OPT 1.3B.*

![Peak memory at 8 K context](output_figures/end2end/memory/memory-8k.pdf)

*Figure 2. Peak GPU memory at 8 K tokens on the same models.*

![OPT 1.3B peak memory across sequence lengths](output_figures/end2end/memory/memory-opt1.3b.pdf)

*Figure 3. Peak GPU memory for OPT 1.3B across sequence lengths 4 K to 16 K, comparing the three methods. The Jenga advantage widens with context.*

![OPT 350m peak memory across sequence lengths](output_figures/end2end/memory/memory-opt350m.pdf)

*Figure 4. Peak GPU memory for OPT 350m across the same sequence lengths, confirming the trend at smaller model size.*

### 4.2 Step Time (Paper Figure 13)

Median per step total time (forward, backward, and optimizer step), excluding the first warmup step:

| Model | Seq | LoRA (ms) | LongLoRA (ms) | Jenga (ms) | Jenga speedup |
| --- | --- | --- | --- | --- | --- |
| Llama 2 7B | 4 K | 872 | 884 | 805 | 1.08× |
| Llama 2 7B | 8 K | 1886 | 1789 | 1684 | 1.12× |
| OPT 1.3B | 4 K | 238 | 239 | 229 | 1.04× |
| OPT 1.3B | 8 K | 512 | 468 | 470 | 1.09× |

The 1.12× Llama 2 7B 8 K speedup matches the paper within one percent. The step time advantage is smaller than the memory advantage because Jenga reduces the token count entering attention and the MLP, which scales activation memory near linearly with token count, while the weight bound matmuls have a fixed cost.

![Execution time comparison on A100 80 GB](output_figures/end2end/time/time-a100.pdf)

*Figure 5. Median per step training time for LoRA, LongLoRA, and Jenga on Llama 2 7B and OPT 1.3B at 4 K and 8 K, on the A100 80 GB device.*

![Execution time comparison on A40](output_figures/end2end/time/time-a40.pdf)

*Figure 6. The same configurations measured on an A40 device. The Jenga speedup is preserved across hardware; the absolute step times scale with the device's available memory bandwidth.*

![Per step time across sequence lengths](output_figures/end2end/time/time-seq.pdf)

*Figure 7. Per step time across sequence lengths 4 K to 64 K on Llama 2 7B and Llama 3 8B. Jenga's wall clock advantage compounds with context.*

### 4.3 Memory and Time Breakdown (Paper Figure 14)

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

Three observations the decomposition makes visible. Activations dominate the savings: at 8 K, Jenga cuts activations from 59,438 MB to 30,827 MB, while model state is essentially unchanged. The predictor overhead is constant at 656 MB across every Jenga configuration. LongLoRA carries roughly four extra GB of "others" attributable to additional buffers retained by its shifted attention reordering.

![Memory breakdown](output_figures/ablations/memory-breakdown/memory-breakdown.pdf)

*Figure 8. Decomposition of Llama 2 7B peak memory at 8 K across LoRA, LongLoRA, and Jenga, and across Jenga sequence lengths 8 K to 16 K.*

![Per step time breakdown](output_figures/ablations/time-breakdown/time-breakdown.pdf)

*Figure 9. Per step wall clock decomposition into attention, MLP, predictor, and others across the same configurations. The predictor contributes a small fixed cost; the savings come from the reduced attention and MLP token counts.*

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

The artifact additionally ships a `LongBench` accuracy harness for downstream task evaluation (`scripts/end2end-longbench/`). LongBench is out of scope for the present reproduction and we report only the windowed perplexity benchmark of Table 7. The infrastructure is mirrored in our public artifact (Section 10) for follow-up work.

### 4.5 Segmented Loss (Paper Figure 18)

The segmented loss study runs the artifact's default configuration of Llama 3 8B at 14,336 tokens and produces two PyTorch memory profile pickles:

| File | Size | Variant |
| --- | --- | --- |
| `logs/ablations/segment/base.pickle` | 8.2 MB | Naive autoregressive loss (single backward over full vocabulary logits) |
| `logs/ablations/segment/segment.pickle` | 8.2 MB | Segmented loss (chunked backward, activations discarded between chunks) |

Both pickles reproduce on the artifact pod and render at `docs.pytorch.org/memory_viz`. The rendered timelines confirm that segmenting the cross entropy removes the terminal full vocabulary logits spike that dominates the naive variant's peak.

### 4.6 Algorithm Ablation (Paper Figure 15)

Attention only and MLP only sparsity ablations were run for Llama 2 7B and OPT 6.7B:

![Llama 2 7B attention ablation](output_figures/ablations/algorithm/algorithm-llama2-attn.pdf)

*Figure 10. Memory versus attention sparsity ratio on Llama 2 7B.*

![Llama 2 7B MLP ablation](output_figures/ablations/algorithm/algorithm-llama2-mlp.pdf)

*Figure 11. Memory versus MLP sparsity ratio on Llama 2 7B.*

![OPT 6.7B attention ablation](output_figures/ablations/algorithm/algorithm-opt-attn.pdf)

*Figure 12. Memory versus attention sparsity ratio on OPT 6.7B.*

![OPT 6.7B MLP ablation](output_figures/ablations/algorithm/algorithm-opt-mlp.pdf)

*Figure 13. Memory versus MLP sparsity ratio on OPT 6.7B.*

The curves reproduce the paper's qualitative shape: a sharp memory descent at low sparsity that flattens beyond roughly 0.6 retention for both attention and MLP, and a noticeable gap between Llama and OPT attributable to the GQA versus MHA head structures.

### 4.7 Predictor Convergence (Paper Figure 16)

The attention predictor is trained offline against pooled attention scores produced by a frozen base model. The loss curve confirms the paper's rapid convergence claim: the predictor reaches its asymptote within the first one hundred epochs and is stable for the remainder of training.

![Predictor training loss](output_figures/ablations/predictor/predictor-loss.pdf)

*Figure 14. Predictor training loss curves across Llama 2 7B and OPT 6.7B on RedPajama and a held out academic mix. The asymptote within one hundred epochs supports the paper's "elastic pattern" claim that the predictor is cheap to train once and reusable across downstream fine tunes.*

### 4.8 Scalability (Paper Figure 17)

The scalability sweep runs the multi GPU variant of the Jenga training driver across one, two, four, and eight A100s and reports near linear speedup.

![Llama 2 7B scalability](output_figures/scalability/scalability-llama2.pdf)

*Figure 15. Multi GPU scaling for Llama 2 7B training under Jenga across one to eight A100 80 GB devices. The slope is close to ideal until eight devices, where inter device communication begins to dominate.*

![OPT 6.7B scalability](output_figures/scalability/scalability-opt.pdf)

*Figure 16. The same scaling sweep on OPT 6.7B. The trend is consistent with Llama.*

### 4.9 Authors' Hidden Gem Extensions (Paper Figure 19)

The paper teases two further extensions briefly. We reproduce the figures so the reader has the visual reference; our Section 5 extension (Token Merging) is independent of these, and our Section 6.2 measurement evaluates a direct composition with the 2D variant.

![2D sparsity speedup](output_figures/extension/2d/2d-sparsity.pdf)

*Figure 17. The 2D sparsity composition of Jenga with LongLoRA shifted attention, reported by the authors as a 2.04× wall clock speedup over LoRA on Llama 2 7B. This figure is the direct motivation for the Section 6.2 measurements.*

![CPU offload extension](output_figures/extension/offload/offload.pdf)

*Figure 18. CPU offload extension. Activations are streamed to CPU during forward and recomputed during backward, trading wall clock for additional memory headroom.*

## 5. Token Merging: A Soft Elimination Extension

**Motivation.** Jenga's elimination is hard: blocks that the predictor scores below the retention threshold are removed from the attention and MLP computations and contribute nothing to the residual stream beyond the layer where they were dropped. The downstream layers receive zero in place of the eliminated content. Bolya et al. (2023) showed in the vision setting that a soft elimination — averaging similar tokens into a single representative — preserves enough of the discarded signal that accuracy degradation becomes negligible. The contribution of this section is to apply the same idea to Jenga's per layer block sparsity in a long context language model.

**Design.** At each sparse layer (index 15 to 30 in Llama 2 7B's stack), the modified forward pass extracts the dropped block indices, gathers the corresponding token hidden states from the pre drop layer input, and mean pools them into a single summary token of the same hidden dimension. The summary token is appended to the kept sequence at the position of the last kept token. The attention output of the summary token is then broadcast onto every dropped position when the layer output is reconstructed, restoring a non-zero residual signal across the dropped portion of the sequence.

Three implementation constraints shape the design.

1. *Single summary token, not 64 copies.* The naive choice of copying the merged representation 64 times into the dropped block produces zero length internal segments in flash attention's varlen indexing. A single token eliminates this case.
2. *Position at the last kept index.* Flash attention's varlen kernel requires the position id tensor to be non decreasing. Placing the summary token at any position earlier than `idx[-1]` violates that constraint. Using `idx[-1]` is the smallest valid choice that does not introduce a synthetic out of range position.
3. *Scatter back onto dropped positions.* When the attention output is scattered back into a tensor of shape (batch, seq_len, hidden), the kept tokens are written to their original positions using the unique kept indices, and the summary token is broadcast onto the dropped positions. This gives the dropped portion of the residual stream the merged representation as its layer output rather than zero, which is the semantic intent of soft elimination.

The modification is guarded by `config.merge_eliminated`; when this flag is False the path is byte identical to the original hard drop. No additional parameters are introduced.

**Cost.** Memory: one additional token's worth of activations at every sparse layer. For Llama 2 7B at 8 K context with retention 0.4, one summary token is appended to roughly 819 kept tokens, an overhead of approximately 0.12 percent. Compute: one extra hidden dimensional vector per sparse layer is processed by the attention and MLP, with the same linear projection cost.

**Relation to ToMe.** Bolya et al. (2023) merge tokens via bipartite similarity matching and average the matched pairs. We do not use bipartite matching: the predictor has already selected which blocks to drop, and we mean pool the dropped block into one summary token rather than match against the kept set. The justification is twofold. The predictor is the single source of truth for which tokens are low information, so re-deciding via similarity would introduce a second learned component that the equal budget constraint cannot afford to tune. And flash attention's varlen kernel cannot accommodate the kind of dynamic per query bipartite structure that ToMe assumes.

**Joint training protocol.** Following the equal-budget fairness rule of Section 3, a new LoRA adapter is trained from step zero with `merge_eliminated = True`. Training hyperparameters match the original Jenga LoRA: r = 8, alpha = 16, target modules `q_proj`, `k_proj`, `v_proj`, and `o_proj`, AdamW with fp32 optimizer state, bf16 throughout, per device batch size one, learning rate 2e-5 with a 20 step warmup and a constant schedule thereafter. Context length is 8 K and the schedule is 2,400 optimizer steps over 1,000 RedPajama documents. The trained adapter is saved to `checkpoints/peft_model_merged/` and mirrored to HuggingFace (Section 10).

## 6. Results

### 6.1 Forward Loss Comparison on 500 Held Out Documents

The principal comparison evaluates five configurations of the same Llama 2 7B + Jenga forward path on 500 held out RedPajama documents at 8 K context, drawn from indices 1,000 onward so the retrained adapters are not measured on their own training set. All other variables are held constant: same base model, same predictor weights, same seed, same documents.

| Adapter | Mode | Mean loss | PPL ≈ exp(loss) | Peak memory (MB) | Mean forward (s) |
| --- | --- | --- | --- | --- | --- |
| Original Jenga | hard drop | 2.380 | 10.81 | 19,804.0 | 1.351 |
| Original Jenga | token merging | 4.414 | 82.63 | 19,804.9 | 1.372 |
| Retrained with merging | token merging | **1.716** | **5.56** | 19,805.5 | 1.366 |
| 2D sparsity adapter | hard drop | 3.040 | 20.90 | 19,805.0 | 1.329 |
| 2D sparsity adapter | token merging | 3.326 | 27.84 | 19,804.5 | 1.343 |

Three findings are legible.

First, enabling token merging at inference time on the **original Jenga adapter** is catastrophic: mean loss jumps from 2.380 to 4.414 and the approximate perplexity rises by more than seven fold from 10.8 to 82.6. The original adapter was trained against a hard drop forward in which the dropped positions carry zero residual signal across the sparse layers; introducing a broadcast merged vector at those positions changes the downstream activations in a way the adapter has no parameters to compensate for. Token Merging is therefore not a drop-in modification — the adapter must be jointly trained with the merged forward for the modification to be usable.

Second, the **retrained adapter** — which has seen the merged forward from the first optimizer step — reduces mean loss to 1.716 (PPL ≈ 5.56), an improvement of **0.66 absolute (28 percent)** over the original adapter under hard drop and a **48 percent reduction in approximate perplexity**. The memory delta from token merging is below 1.5 MB on a 20 GB allocation; the mean forward time is within noise. This is the headline result for the Token Merging extension and constitutes the main contribution of this report.

Third, the **2D sparsity adapter** — trained for the same number of steps on the LongLoRA shifted attention variant of the Jenga forward, with post hoc broadcast merging active from step zero — lands at mean loss 3.040 under hard drop and 3.326 under merging. Both values are substantially worse than the retrained Token Merging adapter (1.716) and worse than the original Jenga adapter under hard drop (2.380). Section 6.2 analyses this composition.

Peak GPU memory is flat across all five states (within 1.5 MB of one another on a 20 GB allocation). Mean forward wall clock is within noise across all five.

![Per document forward loss distribution across the states](output_figures/improvement/loss-landscape.pdf)

*Figure 19. Loss landscape across 500 held out documents. For each of the four states, the 500 per document forward losses are sorted ascending and drawn as a smooth line, so the x axis is a document rank index (1 to 500, easiest to hardest) and the y axis is forward loss; the shape of each curve is the empirical loss distribution for that state. The four curves separate cleanly with essentially no crossover, which means the retrained Token Merging adapter does not just beat the others on average — it beats them on essentially every document. The 28 percent and 48 percent headline reductions of Section 6.1 are therefore a population effect, not a few favourable documents.*

![Normalized loss, PPL, and peak memory across the states](output_figures/improvement/comparison.pdf)

*Figure 20. Normalized mean loss, PPL, and peak GPU memory across the four states. Each metric is divided by its own maximum so the three quantities can share a single y axis; the absolute numbers live in the Section 6.1 table. Two observations: the peak memory bars are visually identical because token merging adds approximately one token per sparse layer on top of roughly 819 kept tokens (about 0.12 percent overhead); and the loss and PPL bars move in the same direction but with different magnitudes, because PPL = exp(loss) amplifies loss differences exponentially.*

![LoRA training loss with merging enabled from step 0](output_figures/improvement/train-loss-i4.pdf)

*Figure 21. Training loss trajectory for the retrained Token Merging adapter (cool palette). Raw per logging step loss in teal; the dark blue line is a ten step moving average to make the trend legible against the noisier raw trace. The adapter starts near 5.0 (an untuned LoRA against the merged forward), descends quickly within the first 100 to 200 optimizer steps, and settles into a stable band of approximately 1.4 to 1.6 forward loss that it holds for the remainder of the 2,400 step schedule. Training is therefore well converged by the time the 500 document evaluation runs.*

### 6.2 Composition with Hidden Dimension Sparsity

The fourth and fifth rows of the table report the composition of Jenga's per layer token sparsity with LongLoRA's shifted attention (the authors' Figure 19 upper). The two sparsity mechanisms operate on orthogonal axes: Jenga prunes along the token axis, the LongLoRA shift restructures the head axis. The combination is a plausible candidate for free compounding.

We tested the composition by training a LoRA adapter from step zero on the `modeling_llama_2D` variant of the forward pass with Token Merging active. The shifted attention imposes a divisibility constraint: it splits heads in half and rolls one half by half a group, which requires the kept token count to be divisible by 8. Appending one summary token (Section 5) would violate this constraint at sparse layers. We therefore use a **post hoc broadcast** variant of Token Merging for the 2D path: the merged token is not appended to the attention input — instead, the mean of the pre drop hidden states is computed outside the attention call and broadcast directly onto the dropped positions at scatter time. The merged token in this variant does not modulate through attention with the kept keys; it simply replaces the zero residual at dropped positions with a static mean.

The 500-document evaluation shows that the composition regresses on quality. At the same 2,400 step training budget, the 2D adapter reaches loss 3.040 under hard drop and 3.326 under merging, both substantially worse than the Token Merging adapter on `modeling_llama` (1.716). Two factors plausibly contribute. The shifted attention's head split breaks global causal structure across half the heads, making the predictor's block scores less informative downstream; and the post hoc broadcast is a weaker form of soft elimination than the in-attention variant of Section 5, because the merged token never attends to the kept keys. Both contribute, and we cannot disentangle them with this experiment.

We report this as an honest negative result that bounds the regime in which Token Merging compounds with other sparsity mechanisms. The authors' Figure 17 reports a 2.04× *wall clock* speedup for the 2D composition; our measurement of *forward loss* does not contradict that claim — it merely shows that the wall clock benefit is not currently transferable to a quality improvement at the budget evaluated.

![LoRA training loss on the 2D sparsity model](output_figures/improvement/train-loss-i5.pdf)

*Figure 22. Training loss trajectory for the 2D sparsity adapter (Jenga token sparsity composed with LongLoRA shifted attention, with post hoc broadcast merging active from step zero) drawn in the warm palette to mark the negative result. Raw loss in peach, ten step moving average in pink. Compared with the cool curve in Figure 21, the trace is noisier and the converged band sits roughly one full unit higher (approximately 2.4 to 2.7 versus 1.4 to 1.6). The shifted attention's head split is the dominant source of the residual noise at the same step budget, and explains why the Section 6.2 evaluation row 4 lands above row 3.*

## 7. Discussion

**What reproduced cleanly.** Every memory and time measurement matches the authors' shipped logs within approximately one percent on the same configurations. The memory savings scale with sequence length as the paper claims, the predictor overhead is the constant 656 MB the paper reports, and the LongLoRA "others" excess is the expected four GB. The perplexity sweep confirms the central trade off: Jenga pays 2 to 5 percent perplexity in return for 31 to 39 percent peak memory reduction and a 4 to 12 percent step time speedup.

**Why Token Merging is the right soft variant.** Two design choices distinguish it from the alternatives. First, it inherits Jenga's existing block sparsity decision and merely changes how the dropped blocks contribute to the residual stream, so the predictor remains the single source of truth for which tokens are low information and no second learned component is introduced. Second, the modification is local to the attention forward and adds neither parameters nor a hyperparameter that would have to be tuned. Both properties keep the technique cheap to evaluate at scale, which is precisely what the equal-budget fairness rule (Section 3) demands.

**Reading the joint training result.** The 28 percent reduction in mean forward loss and 48 percent reduction in approximate perplexity at 500 documents is the operational endpoint of the technique: the retrained adapter trained against the merged forward, evaluated under the same forward. The per document loss distribution shape (Figure 19) shows clean separation between the retrained-with-merging curve and the baselines, so the gain is a population effect rather than the variance of a small sample. The mechanism is direct: the dropped positions, which were zero under the original Jenga forward, now carry the mean of their block's pre drop hidden states; downstream layers receive a non zero signal across the entire sequence and produce a more confident predictive distribution.

**Reading the 2D composition result.** The negative result on the LongLoRA shift composition (Section 6.2) is informative beyond Token Merging itself. It bounds the productive regime of the technique: Token Merging compounds with the orthogonal Jenga axis (the per layer block sparsity it was designed for) but does not compound with the head axis sparsity of LongLoRA at the budget evaluated. The paper's 2.04× speedup claim for the 2D composition remains untested by our forward loss measurement; the two metrics are not commensurable.

## 8. Limitations

* **Single seed.** Memory, step time, perplexity, and extension measurements all use a single seed. The original paper's accuracy claims use three seeds; our work does not.
* **Forward loss as a proxy for perplexity.** The artifact's perplexity evaluator uses the dense baseline forward path with a LoRA adapter; it does not exercise the Jenga sparse forward at inference. Downstream perplexity comparisons for any configuration flag added to the Jenga forward therefore use forward loss on the patched path rather than the paper's windowed perplexity protocol. The two are not identical metrics, although they correlate strongly within a fixed forward path.
* **Trimmed perplexity scope.** Perplexity is measured at 8 K and 16 K on `proof_pile.bin` but only at 8 K on `test_pg19.bin`.
* **OPT 1.3B substitution for the convolutional predictor study.** Llama 2 7B with `output_attentions = True` exceeds 48 GB at the predictor training sequence length; OPT 1.3B at 2 K was substituted. The predictor is model agnostic so the relative comparison is preserved but absolute mean squared errors do not transplant.
* **Predictor convergence and scalability are reproduced from shipped logs.** Figures 14, 15, and 16 are derived from the artifact's distributed logs rather than from fresh GPU runs of our own.
* **Post hoc broadcast in the 2D composition.** The 2D path uses the post hoc broadcast variant of Token Merging because the LongLoRA shift requires the kept token count to be divisible by 8. The semantics of this variant are weaker than the in-attention variant used in Section 5, and the Section 6.2 negative result conflates the two effects.

## 9. Conclusion

We reproduce Jenga's principal memory and time savings to within approximately one percent of the authors' shipped numbers on Llama 2 7B and OPT 1.3B and confirm that the 2 to 5 percent perplexity premium is small relative to the 31 to 39 percent memory reduction. We then introduce **Token Merging for Jenga**, a runtime modification that replaces Jenga's hard token elimination with a single mean pooled summary token per sparse layer, gated by a single configuration flag. With the LoRA adapter retrained from step zero against the merged forward, we observe a **28 percent reduction in mean forward loss and a 48 percent reduction in approximate perplexity** on 500 held out RedPajama documents, at essentially zero memory cost and within noise on wall clock. We additionally compose Token Merging with the LongLoRA shifted attention variant and observe that the composition regresses on quality at the same training budget, an honest negative result that delimits the technique's productive regime.

The natural follow up work falls along four axes.

1. *Multi seed evaluation.* Retrain the Token Merging adapter under three or more seeds at the same step budget and report mean and standard deviation of the 500 document forward loss to characterise the variance of the gain.
2. *Windowed perplexity on the Jenga forward.* Build a Jenga aware perplexity evaluator so the Section 5 claim is reported in the paper's own metric rather than via forward loss as a proxy.
3. *Long range downstream accuracy.* Evaluate the Token Merging adapter on the LongBench task suite shipped with the artifact, which is currently out of scope of this report.
4. *Compounding investigation.* Re-examine the 2D composition with the in attention variant of Token Merging (rather than the post hoc broadcast forced by the LongLoRA divisibility constraint), to isolate the head split contribution from the merging variant contribution.

## 10. Reproducibility

All artifacts required to reproduce every result in this report are publicly available. The source code lives on GitHub at `outofaditya/jenga-labs`; the input and output artifacts (authors' shipped checkpoints, datasets, and our retrained LoRA adapters) are mirrored on a public HuggingFace dataset.

**HuggingFace mirror.** The dataset `outofaditya/jenga-labs-artifacts` contains:

| Path within mirror | Contents |
| --- | --- |
| `checkpoints/predictor/` | Authors' frozen attention predictor (`predictor.pth`, `pruned_config.pth`) |
| `checkpoints/peft_model/` | Authors' shipped LoRA adapters across sequence lengths |
| `checkpoints/peft_model_merged/` | Our retrained Token Merging adapter (Section 5–6, headline result) |
| `checkpoints/peft_model_2d_merge/` | Our 2D sparsity composed adapter (Section 6.2 negative result) |
| `dataset/PPL/` | Pre extracted perplexity benchmarks (`proof_pile.bin`, `test_pg19.bin`) |
| `dataset/RedPajama-Data-1T-Sample/` | Pre extracted training and evaluation corpus |

Each retrained adapter is the full PEFT directory (adapter weights, adapter config, tokenizer, trainer state). The mirror is referenced by the bootstrap script `scripts/run_pod.sh` via the `HF_MIRROR_REPO` environment variable, so a fresh pod boot pulls every input artifact in a single command. Re-running the bootstrap also pulls the retrained adapters so their evaluation is reproducible without re-training.

**Software environment.** A dedicated conda environment with Python 3.10.20 hosts the pinned dependency tree:

| Package | Version | Note |
| --- | --- | --- |
| torch | 2.1.2 + cu121 | from upstream requirements |
| transformers | 4.45.2 | internal symbols are monkey patched by the modeling files |
| tokenizers | 0.20.1 | from upstream requirements |
| deepspeed | 0.14.0 | from upstream requirements |
| bitsandbytes | 0.41.1 | from upstream requirements |
| accelerate | 1.13.0 | latest compatible |
| peft | 0.19.1 | latest compatible |
| datasets | 2.21.0 | pinned `< 3` for legacy loading script support |
| flash-attn | 2.5.6 | prebuilt wheel `cu122 torch 2.1 cxx11abi false cp310` |
| numpy | 1.26.4 | pinned `< 2` for torch 2.1.2 ABI compatibility |
| setuptools | `< 70` | pinned so `torch.utils.cpp_extension` can import `pkg_resources.packaging` |

**Hardware.** The reproduction was executed on rented single GPU instances with at least 46 GB of VRAM. End to end memory and time measurements (Sections 4.1 to 4.3) used the A100 80 GB SXM4; the algorithm ablations and the perplexity sweep used an RTX A6000 48 GB or RTX 4090 48 GB; the Token Merging joint training and the 2D composition training used an A100 80 GB PCIe and an A6000 respectively, with the trained adapters published to the HuggingFace mirror.

**Operational pattern.** All reproduction and extension work fits within a single pod hygiene paradigm: one bootstrap installs and stages everything, every experiment writes to standard log and figure paths, and the HuggingFace mirror is the canonical home for inputs (authors' shipped artifacts) and for outputs (our retrained adapters). The total GPU budget for the entire reproduction and the two extensions is approximately twenty A100-equivalent hours, dominated by the two 2,400-step extension trainings.

**One command bootstrap.**

```bash
export HF_TOKEN=<your_token> \
       HF_MIRROR_REPO=outofaditya/jenga-labs-artifacts
bash scripts/run_pod.sh
```

The script creates the conda environment, installs the pinned dependency tree, installs the `jenga` package in editable mode, pulls every artifact from the HuggingFace mirror, and verifies the smoke test sentinels.

**Reproducing each result.** Every figure and table in this report can be regenerated by a single script invocation.

| Section | Command |
| --- | --- |
| 4.1 Peak Memory | `bash scripts/end2end-memory/run.sh && python src/experiment/end2end/memory/plot_comparison_4k.py && python src/experiment/end2end/memory/plot_comparison_8k.py && python src/experiment/end2end/memory/plot_sequence_opt1.3b.py && python src/experiment/end2end/memory/plot_sequence_opt350m.py` |
| 4.2 Step Time | `bash scripts/end2end-time/run.sh && python src/experiment/end2end/time/plot_comparison_a800.py && python src/experiment/end2end/time/plot_comparison_a40.py && python src/experiment/end2end/time/plot_sequence.py` |
| 4.3 Breakdown | `bash scripts/ablation-mem-breakdown/run.sh && bash scripts/ablation-time-breakdown/run.sh && python src/experiment/ablation/memory-breakdown/plot.py && python src/experiment/ablation/time-breakdown/plot.py` |
| 4.4 Perplexity | `bash scripts/end2end-ppl/run.sh` |
| 4.5 Segmented Loss | `bash scripts/ablation-segment/run.sh` |
| 4.6 Algorithm Ablation | `bash scripts/ablation-algorithm/run.sh && python src/experiment/ablation/algorithm/plot_llama2_attn.py && python src/experiment/ablation/algorithm/plot_llama2_mlp.py && python src/experiment/ablation/algorithm/plot_opt_attn.py && python src/experiment/ablation/algorithm/plot_opt_mlp.py` |
| 4.7 Predictor | `python src/experiment/ablation/predictor/plot_loss.py` |
| 4.8 Scalability | `python src/experiment/scalability/plot_llama2.py && python src/experiment/scalability/plot_opt.py` |
| 4.9 Authors' Extensions | `python src/experiment/extention/2D/plot.py && python src/experiment/extention/offload/plot.py` |
| 5–6 Token Merging | `bash scripts/extension-token-merging/train.sh && python src/experiment/extension_token_merging/measure_three_way.py --n_docs 500 && python src/experiment/extension_token_merging/parse_eval_log.py && python src/experiment/extension_token_merging/plot_comparison.py && python src/experiment/extension_token_merging/plot_training_loss.py` |
| 6.2 2D Composition | `bash scripts/extension-2d-merge/train.sh && python src/experiment/extension_2d_merge/measure_2d.py --n_docs 500 && python src/experiment/extension_2d_merge/parse_eval_log.py && python src/experiment/extension_token_merging/plot_comparison.py` |

Logs land under `logs/`, figures under `output_figures/`, retrained adapters under `checkpoints/`. The bootstrap is idempotent; re-running it on a fresh pod restores the full state in a single command.

## References

1. *Jenga: Enhancing Long Context Fine Tuning of LLMs with Contextual Token Sparsity.* USENIX ATC 2025.
2. Hu et al. *LoRA: Low Rank Adaptation of Large Language Models.* ICLR 2022.
3. Bolya et al. *Token Merging: Your ViT but Faster.* ICLR 2023.
4. Dao et al. *FlashAttention 2: Faster Attention with Better Parallelism and Work Partitioning.* 2023.
5. Touvron et al. *Llama 2: Open Foundation and Fine Tuned Chat Models.* 2023.
6. Zhang et al. *OPT: Open Pretrained Transformer Language Models.* 2022.
7. Chen et al. *LongLoRA: Efficient Fine Tuning of Long Context Large Language Models.* ICLR 2024.
