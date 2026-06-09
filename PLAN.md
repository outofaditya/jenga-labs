# PLAN.md

Reproduction and Extension Plan for the Jenga paper. Course deliverable: one 6 to 8 page LaTeX report. Deadline four weeks from start. Hardware A100 80GB rented on Vast.ai or RunPod. Initial budget approximately twenty dollars with top up authorized for promising directions.

## Operating Principles

1. **Reuse the existing pipeline.** This repository already ships the training drivers, profiling trainers, plotting scripts, and authors' raw logs. New scripts are written only where the existing surface does not cover the experiment.
2. **Local checkpoint paths.** All model loads point at `checkpoints/<model>/` (matching the existing shell scripts) rather than Hugging Face hub strings.
3. **Authors' shipped artifacts.** The provided `checkpoints/predictor/predictor.pth`, `checkpoints/predictor/pruned_config.pth`, and `checkpoints/peft_model/` adapters are used as is for the reproduction phase. Retraining the predictor is optional and budget gated (see Part 2).
4. **Naming.** The system is "Jenga". Avoid "JENGA" and marketing language in artifacts that feed the report.
5. **Fairness rule for extensions.** Every extension in Part 3 is compared against a Jenga baseline retrained under the same training budget as the extension itself. Comparing an extension trained for a few hundred steps against the authors' fully trained adapters is rejected as unfair to the extension.

## Measurement Rigor

* **Warmup.** Five forward and backward passes before any logging to absorb CUDA initialization, allocator caching, and bf16 kernel autotuning.
* **Steps Measured.** Thirty to fifty measured steps per memory or time configuration. Report mean and standard deviation rather than a single average.
* **Seeds.** Single seed acceptable for memory and time profiling. Three seeds required for any accuracy claim (perplexity, retention ratios, predictor loss curves) so the report can carry error bars.
* **Pod Hygiene.** Bring up one long lived Vast.ai pod, install once, download data and checkpoints once, snapshot the image. Tear down only after the final run. Boot and download time is paid GPU time.

## First Kill Criterion

Before any Llama run is paid for, the following must pass on OPT-350m at sequence length 8192:

1. Baseline LoRA training step completes without NaN loss for ten steps.
2. Jenga training step completes without NaN loss for ten steps.
3. Jenga peak memory is strictly below baseline peak memory.

If any of the three fails the environment is broken and further spending is paused until it is fixed.

## Budget Ledger

| Phase | Est A100 80GB Hours | Est Cost USD | Status |
| --- | --- | --- | --- |
| Part 0 sanity | 0.5 | 0.50 | pending |
| Part 1 reproduction | 6 | 6 | pending |
| Part 2 ablations | 3 | 3 | pending |
| Part 3 extensions | 8 | 8 | pending |
| Buffer | 2.5 | 2.50 | reserve |
| **Total** | **20** | **20** | |

A100 80GB is priced here at one dollar per GPU hour as a Vast.ai working estimate. Update this table with actual hours and cost at the end of each part.

## Timeline

| Week | Focus |
| --- | --- |
| 1 | Environment, Part 0, Part 1 baseline and Jenga end to end |
| 2 | Part 1 perplexity, Part 2 ablations |
| 3 | Part 3 extensions in order 3.1 then 3.2 then 3.3, plus sensitivity sweeps |
| 4 | Report drafting, figure polish, final readthrough |

Five days at the end of week four are reserved for the report. No new experiment is started in week four unless it directly fills a gap the writeup already opened.

## Risk Register

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| Hugging Face Llama gated access not granted in time | medium | Default to Llama 2 7B for which access is already requestable; Llama 3 8B is a stretch goal not the hero |
| flash attn build fails on Vast.ai image | medium | Pin `torch==2.1.2` with `cu121` and install `flash-attn==2.5.6 --no-build-isolation`; fall back to attn implementation eager if rebuild fails |
| Llama at 16384 OOMs on 80GB without gradient checkpointing | medium | Enable `gradient_checkpoint=True` on the baseline runs at 16K and disclose the asymmetry in the report |
| Jenga plus QLoRA crashes in `PrunedLlamaMLPFunction` because `bnb.nn.Linear4bit` is not a float tensor | high | Restrict 4 bit quantization to attention projections; leave MLP in bf16 for the first pass |
| Tsinghua cloud links throttle from outside China | high | Mirror `dataset.zip`, `peft_model.zip`, and `predictor.zip` to a personal Hugging Face dataset on first download and pull from there afterwards |
| Single seed gives spurious improvement signal | medium | All accuracy comparisons run at three seeds |

---

## Part 0: Free Sanity Check

**Objective.** Validate environment installation and regenerate every paper figure for free from authors' shipped logs before paying any GPU time.

**Steps.**

1. Install per `README.md`:
   * `pip install -r requirements.txt`
   * `pip install flash-attn --no-build-isolation`
   * `pip install -e .`
2. Stage checkpoints and dataset per `README.md` layout. Mirror to a personal Hugging Face dataset on first download.
3. Run `bash hello-world.sh` on the GPU pod. Confirm the printed checks pass.
4. Run `bash RUNME-a.sh` on CPU (no GPU needed). This regenerates the full `output_figures/` tree from `logs/` in roughly two minutes and gives the visual ground truth that every fresh run is compared against.

**Output.** A populated `output_figures/` tree matching the paper.

---

## Part 1: End to End Reproduction

**Objective.** Measure peak memory, step time, and perplexity for baseline LoRA and Jenga across two sequence lengths and confirm the qualitative shape of Figures 12 and 13 and Table 7 of the paper.

**Primary models.** `checkpoints/llama2` (Llama 2 7B) and `checkpoints/opt-1.3b`. `checkpoints/llama3` is promoted into scope only if access lands in week one.

**Sequence lengths.** 8192 and 16384.

**Hyperparameters.** From the existing shell scripts: `pool_size=64`, `thresh=0.4`, `bf16`, LoRA r=8 alpha=16 on q k v o projections, AdamW (`adamw_torch`) with FP32 optimizer states, `gradient_accumulation_steps=1`, batch size one per device. Linear RoPE scaling applied automatically by `set_RoPE` in the existing drivers.

### 1.1 Baseline LoRA Reference Numbers

* Use `scripts/end2end-memory/llama-base.sh` and `scripts/end2end-time/llama-base.sh` (and the corresponding `opt-base.sh`).
* Driver: `src/experiment/end2end/memory/llama_base.py` and `src/experiment/end2end/time/llama_base.py`.
* Model class: `jenga.models.modeling_llama_base.LlamaForCausalLM`.
* Enable `--gradient_checkpoint True` for Llama at 16384 to avoid OOM. Disclose the asymmetry in the report.

### 1.2 Jenga End to End

* Use `scripts/end2end-memory/llama-jenga.sh` and `scripts/end2end-time/llama-jenga.sh` (and `opt-jenga.sh`).
* Driver: `src/experiment/end2end/memory/llama_jenga.py` and `src/experiment/end2end/time/llama_jenga.py`.
* Model class: `jenga.models.modeling_llama.LlamaForCausalLM`.
* Authors' predictor weights loaded via the existing `torch.load` and `state_dict.copy_` pattern. Do not retrain the predictor here.

### 1.3 Perplexity

* Use `src/experiment/end2end/accuracy/ppl.py` driven by `scripts/end2end-ppl/ppl_pg.sh` and `scripts/end2end-ppl/ppl_pp.sh`.
* Benchmark files: `dataset/PPL/proof_pile.bin` and `dataset/PPL/pg.bin` as shipped with the artifact. These are the bins the paper's Table 7 uses; substituting `deepmind/pg19` invalidates any "matches Table 7" claim.
* A separate `deepmind/pg19` evaluation may be added as a generalization probe under a different table label if budget allows.

### 1.4 Logging and Plotting

* Memory and time logs land under `logs/end2end/memory/` and `logs/end2end/time/` per the existing shell scripts.
* JSON schema for any new numeric output the report consumes:
  `{"model": "<model_id>", "method": "<lora|jenga>", "seq_len": <int>, "avg_step_ms": <float>, "step_ms_std": <float>, "peak_memory_gb": <float>, "n_steps_measured": <int>, "seed": <int>}`
* Perplexity schema:
  `{"model": "<model_id>", "seq_len": <int>, "benchmark": "<proof_pile|pg|pg19>", "baseline_ppl": <float>, "jenga_ppl": <float>, "seed": <int>}`
* Plotting via existing `src/experiment/end2end/memory/plot_*.py` and `src/experiment/end2end/time/plot_*.py`. New plots only if the existing scripts do not cover a specific comparison.

**Reproduction Bar.** Same qualitative trends as Figures 12, 13, and Table 7. Numeric drift from the paper is acceptable and discussed in the report. The first kill criterion above is a strict precondition.

---

## Part 2: Granular Ablations

**Objective.** Decompose the end to end gains into their algorithmic components.

### 2.1 Memory Breakdown (Recreates Figure 14 Upper)

* Driver: `src/experiment/ablation/memory-breakdown/plot.py` and the matching `scripts/ablation-mem-breakdown/run.sh`.
* Profiling trainer: `jenga.trainer.memory_profile.Trainer`. This is the source of truth for the breakdown; do not derive activation memory by subtraction.
* Output schema:
  `{"seq_len": <int>, "model_state_gb": <float>, "activations_gb": <float>, "predictor_gb": <float>, "others_gb": <float>}`
* Output: stacked horizontal bar chart matching Figure 14 upper.

### 2.2 Predictor Convergence (Optional, Budget Gated)

* Driver: `src/experiment/ablation/predictor/llama_rp.py`, `llama_la.py`, `opt_rp.py`, `opt_la.py`. Orchestration via `scripts/ablation-predictor/run.sh`.
* This sub part is the most expensive single experiment in the paper at roughly three hours on A800. It is included only if Parts 0, 1, and 2.1 finish under budget. The authors' shipped `predictor.pth` is used everywhere else.
* If skipped, the report's section on predictor convergence cites the paper's Figure 16 directly and discloses we did not re run it.

### 2.3 Segment Based Peak Cutting (Recreates Figure 18, Not Figure 19)

* Drivers: `src/experiment/ablation/segment/base.py` and `src/experiment/ablation/segment/seg.py`. Orchestration via `scripts/ablation-segment/run.sh`, `base.sh`, and `segment.sh`.
* Compares naive cross entropy over the full sequence vocabulary against a chunked variant that runs backward per chunk and discards activations before proceeding.
* Output schema:
  `{"method": "<naive_loss|segmented_loss>", "peak_loss_mem_gb": <float>, "n_chunks": <int>}`
* The figure being reproduced is Figure 18 in the paper. The PyTorch memory viz pickle files this driver emits are dragged into `docs.pytorch.org/memory_viz` for the final visual.

### 2.4 Algorithm Ablation (Optional, Cheap)

* Driver and plotting under `src/experiment/ablation/algorithm/` and `scripts/ablation-algorithm/run.sh`.
* Approximately five minutes total. Isolates whether the speedup comes from attention sparsity or MLP sparsity. Directly informs the framing for the CNN predictor extension below.

---

## Part 3: Three Exploratory Extensions

**Objective.** Test three modifications on top of Jenga. The aim is honest measurement. A negative result is acceptable if it is well controlled. Extensions are ordered ascending by integration risk. Each is compared against a Jenga baseline retrained under the same training budget as the extension itself, not the authors' shipped adapters.

### 3.1 Extension A: Dynamic Adaptive Thresholds (Lowest Risk)

**Hypothesis.** Replacing the static per layer `config.thresh` with a runtime heuristic adapted to the predicted attention entropy of the batch yields a Pareto improvement in perplexity at equal token retention, or vice versa.

**Implementation.**

* Modify `src/jenga/models/modeling_llama.py` and `src/jenga/ops/get_meta.py` where the static threshold is currently consumed. Plumb a per batch override.
* Compute the normalized Shannon entropy of the predictor's positive informativeness scores in the forward pass.
* Apply `t_dynamic = t_base + lam * (1 - entropy_norm)` per layer. Initialize `t_base` to the existing `config.thresh` value of 0.4.
* Expose `lam` as a command line argument.

**Experiments.**

* Sensitivity sweep over `lam in {0.05, 0.1, 0.2}` at Llama 2 7B sequence length 8192. Three seeds per setting.
* Perplexity evaluation reuses the Part 1.3 harness on the paper's `proof_pile.bin` and `pg.bin`.
* Log token retention ratio per layer per batch.

**Output Schema.**

* `{"method": "jenga_adaptive", "lam": <float>, "seed": <int>, "ppl_proof_pile": <float>, "ppl_pg": <float>, "mean_retention_ratio": <float>}`
* `{"batch_idx": <int>, "layer_idx": <int>, "entropy": <float>, "retention_ratio": <float>}`

**Figures.**

* Scatter of predictor entropy versus token retention ratio.
* Bar chart of perplexity across the three `lam` values plus the static baseline retrained under equivalent budget.

### 3.2 Extension B: 1D CNN Predictors

**Hypothesis.** Replacing the per block MLP predictor with a small 1D convolutional predictor over the block dimension captures local sequential context that the MLP cannot, yielding faster offline convergence or a lower final mean squared error against the dense attention ground truth.

**Implementation.**

* Extend `src/jenga/models/predictor.py` with a `CNNAttnPredictor` class. Architecture: two `nn.Conv1d` layers with `kernel_size=3` and `padding=1`, ReLU between them, plus a final linear projection back to the predictor output dimension. The convolutional axis is the block index, not the embedding dimension.
* Reuse the offline predictor training harness in `src/experiment/ablation/predictor/`. Gate the choice of predictor class via a `--predictor_type {mlp,cnn}` argument.

**Experiments.**

* Train the MLP predictor and the CNN predictor on the same RedPajama subset for the same number of epochs. The cap is whichever epoch count fits in the budget remaining after Parts 0 to 2; flag the cap in the writeup.
* Record MSE loss every five epochs.

**Output Schema.**

* `{"epoch": <int>, "predictor_type": "<mlp|cnn>", "seed": <int>, "train_loss": <float>}`

**Figures.**

* Dual line plot of epoch versus MSE loss. MLP dashed, CNN solid.

### 3.3 Extension C: Jenga Plus QLoRA (Highest Risk)

**Hypothesis.** Stacking 4 bit weight quantization on top of token level activation sparsity compounds memory savings without breaking predictor convergence.

**Known Integration Risk.** `PrunedLlamaMLPFunction` in `src/jenga/ops/llama_ops.py` calls `torch.matmul(x, gate_w.t())` directly on the gate, up, and down projection weights. After QLoRA wraps the model with `bnb.nn.Linear4bit` these are packed uint8 tensors, not floats. The function will fail or produce nonsense.

**Mitigation.**

* First pass: restrict 4 bit quantization to attention projections only via the LoRA target module list; leave the MLP in bf16 so `PrunedLlamaMLPFunction` continues to receive float weights.
* Second pass (only if the first works): replace direct `gate_w.t()` accesses with `bnb.functional.dequantize_4bit` calls that materialize the weight transiently for the sparse matmul.

**Implementation.**

* Create `src/experiment/end2end/extension_qlora/llama_jenga_qlora.py` modeled on `src/experiment/end2end/time/llama_jenga.py`.
* Import `BitsAndBytesConfig` from `transformers`. Use `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=torch.bfloat16`.
* Call `peft.prepare_model_for_kbit_training` before LoRA wrapping.

**Experiments.**

* Single configuration at Llama 2 7B sequence length 16384. Compare peak memory and step time against LoRA baseline, Jenga, and Jenga plus QLoRA.

**Output Schema.**

* `{"method": "lora|jenga|jenga_qlora", "seq_len": 16384, "peak_memory_gb": <float>, "avg_step_ms": <float>, "predictor_final_loss": <float|null>}`

**Figures.**

* Grouped bar chart of peak memory at 16384 across the four methods.

**Acceptance.** This extension is the planned "swing" of the three. A clean negative result that documents the integration failure mode is a valid report contribution.

---

## Part 4: Reporting

### Deliverable

One LaTeX report, six to eight pages, conference style. Living markdown skeleton in `REPORT.md` updated alongside the experiments and converted to LaTeX in week four.

### Required Sections

1. Introduction (Jenga summary, our scope, our extensions, headline finding)
2. Background and related work
3. Reproduction methodology (hardware, software pins, model and dataset choices, deviations from paper)
4. Reproduction results (Figures 12, 13, 14 upper, 18, Table 7)
5. Extension design (Extensions A, B, C with hypotheses)
6. Extension results (one subsection per extension, including the QLoRA failure mode if it surfaces)
7. Discussion including a dedicated paragraph on negative results
8. Limitations (single seed for time, budget cap, training step caps, dataset substitutions)
9. Conclusion
10. References

### Honesty Constraints

* Every reproduction claim that says "matches the paper" must be backed by a numeric column from a paper benchmark, not a substituted one.
* Every extension claim that compares against Jenga must compare against an equal budget retrained Jenga baseline.
* Drift from paper numbers is reported, not hidden.
