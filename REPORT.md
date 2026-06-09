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

## Title

Reproducing Jenga and Three Exploratory Extensions to Contextual Token Sparsity for Long Context Fine Tuning

## Abstract

To be drafted last.

## 1. Introduction

Brief framing of long context fine tuning cost. One paragraph summary of Jenga's three pillar idea (contextual token sparsity, the elastic predictor, the segmented loss). One paragraph on our contribution scope: a budget bounded reproduction on A100 80GB plus three exploratory extensions ordered by integration risk. One sentence headline finding.

## 2. Background

Long context fine tuning constraints. PEFT and LoRA baselines. Sparse attention prior work. Jenga's place in that taxonomy. Cite the paper, the artifact repository, and any directly compared prior systems (LongLoRA, base PEFT).

## 3. Reproduction Methodology

Hardware (A100 80GB on Vast.ai). Software pins (`torch==2.1.2`, `transformers==4.45.2`, `flash-attn==2.5.6`). Model and dataset choices (`checkpoints/llama2`, `checkpoints/opt-1.3b`, `dataset/RedPajama-Data-1T-Sample`, `dataset/PPL/proof_pile.bin`, `dataset/PPL/pg.bin`). Measurement protocol (five warmup steps, thirty to fifty measured steps, three seeds for accuracy). Disclose deviations from the paper such as gradient checkpointing enabled on the Llama 16K baseline.

## 4. Reproduction Results

### 4.1 Memory (Reproduces Paper Figure 12)

Insert grouped bar chart of peak memory across LoRA and Jenga at sequence lengths 8192 and 16384 for Llama 2 7B and OPT-1.3B. Annotate the relative memory reduction.

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

Pinned versions, Vast.ai instance image, `pip freeze` output if space permits.
