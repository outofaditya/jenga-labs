# PLAN.md

Reproduction and extension plan for the Jenga paper. Course deliverable: one six to eight page LaTeX report. Deadline four weeks from start. Hardware A100 80GB rented on Vast.ai or RunPod. Initial budget approximately twenty dollars with top up authorized for promising directions.

## Workflow

* The plan is decomposed into **atoms**. Each atom is a single independent testable unit producing one figure, one table, one sanity check, or one report section. Within an atom the executing agent breaks the work into TaskCreate items that tick off as they finish.
* Atoms execute one at a time in the strict order declared here. Hard prerequisites are listed in each atom under **Depends On**. Skipping an atom is allowed only if it is explicitly marked optional.
* At the close of each atom `REPORT.md` is updated with the content that atom produced before the next atom begins.

## Operating Principles

1. **Reuse the existing pipeline.** This repository already ships training drivers, profiling trainers, plotting scripts, and authors' raw logs. New scripts are written only where the existing surface does not cover the experiment.
2. **Local checkpoint paths.** All model loads point at `checkpoints/<model>/` rather than Hugging Face hub strings.
3. **Authors' shipped artifacts.** The provided `checkpoints/predictor/predictor.pth`, `checkpoints/predictor/pruned_config.pth`, and `checkpoints/peft_model/` adapters are used as is for the reproduction phase.
4. **Naming.** The system is "Jenga". Avoid "JENGA" and marketing language in artifacts that feed the report.
5. **Fairness rule for extensions.** Every extension in the Improvement category is compared against a Jenga baseline retrained under the same training budget as the extension itself, not the authors' shipped adapters.

## New Pod Handshake (Strict Preflight)

Every new pod instance burns money the moment the meter starts. Before ANY `nohup`, `bash scripts/run_pod.sh`, or background process is launched on a fresh pod, the probe block below MUST be run and its output read in full. No bootstrap until every line has been inspected and a strategy chosen.

```bash
ssh ... 'bash -s' << "PROBE"
echo "=== SYSTEM ==="
hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
df -h /
echo "=== PYTHON ==="
which python python3 pip pip3
python3 --version || true
python --version 2>/dev/null || echo "no system python"
echo "=== CONDA AND VENVS ==="
ls -d /opt/miniforge3 /opt/conda /opt/miniconda3 /venv /venv/* 2>/dev/null || true
for c in /opt/miniforge3/bin/conda /opt/conda/bin/conda /opt/miniconda3/bin/conda; do
  [ -x "$c" ] && { echo "conda=$c"; "$c" info --envs 2>/dev/null; break; }
done
echo "=== HF TOOLING (xet is a known hang risk) ==="
for p in python3 /venv/jenga/bin/python /venv/main/bin/python /opt/miniforge3/bin/python; do
  [ -x "$p" ] || continue
  echo "--- $p ---"
  "$p" -c "import huggingface_hub; print('hf_hub', huggingface_hub.__version__)" 2>&1
  "$p" -c "import hf_xet; print('hf_xet PRESENT', hf_xet.__file__)" 2>&1 | head -1
done
echo "=== EXISTING WORK ==="
ls -d /workspace/jenga-labs 2>/dev/null && cd /workspace/jenga-labs && git log -1 --oneline
ls /workspace/.bootstrap_done /workspace/.bootstrap_failed 2>/dev/null
PROBE
```

Decision rules from the probe output:

* **If `hf_xet` is installed anywhere it will be used by `snapshot_download` and will hang silently with active TCP connections but zero byte progress.** Disable it for the bootstrap by exporting `HF_HUB_DISABLE_XET=1` AND uninstalling it from the env that will run the download.
* **If a conda env already exists at `/venv/jenga` or similar:** treat it as the canonical env. Set `export PATH=/venv/jenga/bin:$PATH` before any pip or python call. Never let `run_pod.sh` fall into the "system python is fine" branch by accident — that branch silently writes pip installs into the wrong site-packages.
* **If system python is 3.12 or newer:** force the conda branch even if `/venv/jenga` exists, because `torch==2.1.2` has no wheel for Python 3.12.
* **If `.bootstrap_done` already exists:** the previous run succeeded. Verify the sanity sentinels still pass and skip straight to the next atom.
* **Disk under 100 GB:** apply ignore_patterns aggressively for TF, Flax, Llama `original/`, and Llama `pytorch_model.bin`. With ~3 TB disk we can be looser.
* **Never launch two `bootstrap.sh` processes concurrently:** they deadlock on HF cache file locks and both stall silently.
* **Never run an unguarded `snapshot_download` with `shutil.rmtree(dest)` while another process may be writing to the same dest:** that produces `OSError: Directory not empty` and stops the bootstrap.

This handshake is mandatory before pod boot for every subsequent atom run.

## Measurement Rigor

* **Warmup.** Five forward and backward passes before any logging.
* **Steps Measured.** Thirty to fifty measured steps per memory or time configuration. Report mean and standard deviation.
* **Seeds.** Single seed acceptable for memory and time profiling. Three seeds required for any accuracy claim.
* **Pod Hygiene.** One long lived Vast.ai pod, install once, download data and checkpoints once, snapshot the image. Boot and download time is paid GPU time.

## Budget Ledger

| Atom | Phase | Est A100 80GB Hours | Est Cost USD | Status |
| --- | --- | --- | --- | --- |
| S1 | Setup | 0.5 | 0.50 | done (actual 0.7 / 0.70) |
| S2 | Setup | 0.1 | 0.10 | done (passed) |
| S3 | Setup | 0 | 0 | done (19 figures regenerated) |
| S4 | Setup | 0.2 | 0.20 | done (baseline 13.18 GB jenga 7.28 GB; pod preempted mid aggregation but logs synced) |
| R1 | Reproduction | 1.5 | 1.50 | done (12 configs llama2 + opt-1.3b at 4K and 8K; jenga 31 to 39 percent below LoRA matches paper within ~1%) |
| R2 | Reproduction | 1.5 | 1.50 | done (12 configs; jenga 1.04 to 1.12x faster than LoRA matches paper) |
| R3 | Reproduction | 0.5 | 0.50 | done (memory breakdown across 8K LoRA + LongLoRA + Jenga at 8K to 16K; predictor overhead 656 MB constant) |
| R4 | Reproduction | 0.2 | 0.20 | done (base.pickle and segment.pickle generated for Llama 3 14336 manual memory_viz render needed for screenshot images) |
| R5 | Reproduction | 2.0 | 2.00 | done (6 of 6 trimmed scope; Jenga 2 to 5 percent higher PPL than LoRA at 4x memory) |
| R6 | Reproduction | 0.2 | 0.20 | done on RTX 4090 pod 2 v2 (Llama 2 7B + OPT 6.7B attn and mlp ablations) |
| R7 | Reproduction (optional gated) | 3.0 | 3.00 | gated |
| I1 | Improvement | 3.0 | 3.00 | done (mechanism check; 64 layer batch points per lam in 0 0.05 0.1 0.2; downstream PPL deferred to I3) |
| I2 | Improvement | 2.0 | 2.00 | done (negative result: CNN 15x worse than MLP on OPT 1.3B RedPajama) |
| I3 | Improvement | 0.5 | 0.50 | done (positive: 0.4 percent lower loss at near zero memory overhead, naive inference only) |
| I4 | Improvement | 0.7 | 0.70 | pending (joint train LoRA with merge_eliminated true then re-evaluate) |
| P1 | Reporting | 0 | 0 | pending |
| P2 | Reporting | 0 | 0 | pending |
| Buffer | reserve | 3.8 | 3.80 | reserve |
| **Total** | | **20.0** | **20.00** | |

A100 80GB is priced here at one dollar per GPU hour as a Vast.ai working estimate. Update the table with actual hours and dollars at the end of each atom.

## Timeline

| Week | Atoms in Scope |
| --- | --- |
| 1 | S1 S2 S3 S4 R1 R2 |
| 2 | R3 R4 R5 R6 (optional R7) |
| 3 | I1 I2 I3 |
| 4 | P1 P2 |

## Risk Register

| Risk | Likelihood | Mitigation |
| --- | --- | --- |
| Hugging Face Llama 2 access still pending at boot | medium | Llama 3 access already granted; boot with `INCLUDE_LLAMA2=0` if Llama 2 has not approved yet, rerun the base model pull once it does |
| flash attn build fails on Vast.ai image | medium | Pin `torch==2.1.2` with `cu121` and install `flash-attn==2.5.6 --no-build-isolation` |
| Llama at 16384 OOMs on 80GB without gradient checkpointing | medium | Enable `gradient_checkpoint=True` on the baseline runs at 16K and disclose the asymmetry |
| Jenga plus QLoRA crashes in `PrunedLlamaMLPFunction` due to `bnb.nn.Linear4bit` | high | Restrict 4 bit quantization to attention projections only on first pass |
| Tsinghua cloud links throttle from outside China | high | Mirror archive contents to a personal Hugging Face dataset on first download |
| Single seed gives spurious improvement signal | medium | All accuracy comparisons run at three seeds |

## Atom Template

```
### Atom X.Y: Short Name

**Purpose.** One sentence on what this atom proves or produces.
**Depends On.** List of prior atoms or none.
**Inputs.** Files, configs, models, sequence lengths.
**Steps.** Numbered concrete actions an agent can execute without further interpretation.
**Outputs.** Files written and their JSON or figure schema.
**Success Criteria.** Testable assertions.
**Report Update.** Which REPORT.md section receives what content at the end.
**Budget.** Estimated A100 80GB hours and dollars.
```

---

# Category: Setup

### Atom S1: Pod and Environment Bootstrap

**Purpose.** Stand up a long lived A100 80GB pod with the exact software pins the artifact requires and stage all model weights and datasets locally so that no subsequent atom pays for downloads.

**Depends On.** None.

**Inputs.** Vast.ai or RunPod A100 80GB image preloaded with CUDA 12.1. The artifact's `requirements.txt`. A private Hugging Face dataset under your namespace containing the pre extracted Tsinghua artifacts. A read scoped Hugging Face token.

#### S1 Preflight (Off Pod, Zero GPU Cost)

These steps run on your laptop and on the Hugging Face web UI. They convert the slow Tsinghua artifact download into a fast CDN pull and remove all interactive prompts from the pod boot sequence.

1. **Hugging Face gated access.** Llama 3 access already granted. Request Llama 2 access at `huggingface.co/meta-llama/Llama-2-7b-hf` if not yet pending; without it run_pod.sh must be invoked with `INCLUDE_LLAMA2=0` until approval lands.
2. **Download the Tsinghua zips locally.** From `README.md`: `peft_model.zip` and `predictor.zip` under "Model Weights"; `dataset.zip` under "Datasets". Unzip them on your laptop.
3. **Create a private Hugging Face dataset.** Suggested id `<your_username>/jenga-labs-artifacts`. Push the pre extracted contents so the dataset tree contains:

   ```
   checkpoints/peft_model/...
   checkpoints/predictor/...
   dataset/...
   ```

4. **Create a Hugging Face read scoped access token.** Account Settings -> Access Tokens. Store as `HF_TOKEN` for the next step.

#### S1 On Pod (Unattended)

5. Provision one A100 80GB pod. Image base PyTorch 2.1.x with CUDA 12.1 if available, else Ubuntu 22.04.
6. Clone the repository onto the pod.
7. Export the two required environment variables:

   ```
   export HF_TOKEN=hf_...
   export HF_MIRROR_REPO=<your_username>/jenga-labs-artifacts
   ```

8. Run `bash scripts/run_pod.sh`. The script is idempotent. It installs `requirements.txt`, `flash-attn`, and the editable `jenga` package, pulls the pre extracted artifacts from your HF mirror, pulls the base models, and verifies the Atom S1 success criteria. Optional base model downloads are gated by `INCLUDE_*` env vars documented at the top of the script.
9. Snapshot the pod image so future pods skip the bootstrap entirely.

**Outputs.** A pod whose `checkpoints/` and `dataset/` trees match the layout described in `README.md`. A snapshot ID recorded in `REPORT.md` Appendix B.

**Success Criteria.**

* `python -c "import torch; print(torch.cuda.is_available())"` prints `True`.
* `python -c "import flash_attn; import jenga"` exits with code zero.
* `ls checkpoints/llama2/config.json`, `ls checkpoints/predictor/predictor.pth`, and `ls dataset/PPL/proof_pile.bin` all exist.
* `scripts/run_pod.sh` exits with code zero.

**Report Update.** Appendix B Software Environment is populated with the pin list and the snapshot ID.

**Budget.** With the preflight done, on pod time drops to roughly 10 to 20 minutes covering install, mirror pull, and base model pulls. Estimate 0.3 hours actual against 0.5 hours reserved.

### Atom S2: Pipeline Smoke Test

**Purpose.** Confirm that Jenga's forward and backward pass works end to end on the pod with the shipped predictor weights before any expensive run.

**Depends On.** S1.

**Inputs.** `hello-world.sh` at repo root. The Llama 2 checkpoint and the shipped predictor.

**Steps.**

1. Run `bash hello-world.sh`.

**Outputs.** Console log captured to `logs/setup/hello_world.log`.

**Success Criteria.**

* The script prints "Environment compatibility and Jenga functionality test PASSED."
* No CUDA OOM, no NaN loss in the test forward and backward pass.

**Report Update.** None.

**Budget.** Approximately 30 seconds of GPU time. Estimate 0.1 hours including overhead.

### Atom S3: Plot Only Reproduction From Shipped Logs

**Purpose.** Regenerate every paper figure for free from the authors' shipped `logs/` tree so we have the visual ground truth that every fresh GPU run is compared against.

**Depends On.** S1.

**Inputs.** `RUNME-a.sh` at repo root. The `logs/` tree shipped with the artifact.

**Steps.**

1. Run `bash RUNME-a.sh` on the pod (CPU is sufficient).

**Outputs.** A populated `output_figures/` tree matching the figure-to-folder mapping in `README.md`.

**Success Criteria.**

* `output_figures/end2end/memory/` and `output_figures/end2end/time/` each contain at least one PDF.
* `output_figures/ablations/algorithm/`, `ablations/memory-breakdown/`, `ablations/time-breakdown/`, `extension/2d/`, `extension/offload/`, `scalability/` are all non empty.

**Report Update.** None. These figures serve only as the visual baseline against which fresh runs are visually compared.

**Budget.** No GPU time. Estimate 0 hours.

### Atom S4: First Kill Criterion on OPT-350m

**Purpose.** Smallest possible end to end gate. Confirms that the baseline LoRA driver and the Jenga driver both train without NaNs and that Jenga's peak memory is strictly below the baseline at 8192 tokens on OPT-350m. Stops all further spending if any of the three fails.

**Depends On.** S2.

**Inputs.** `scripts/end2end-memory/opt-base.sh` and `scripts/end2end-memory/opt-jenga.sh`. Model `opt-350m`. Sequence length 8192.

**Steps.**

1. `bash scripts/end2end-memory/opt-base.sh opt-350m 8192`
2. `bash scripts/end2end-memory/opt-jenga.sh opt-350m 8192`
3. Read the two resulting log files under `logs/end2end/memory/`.
4. Extract the peak memory line from each run.

**Outputs.** `logs/end2end/memory/opt-base/opt-350m_8192.log` and `logs/end2end/memory/opt-jenga/opt-350m_8192.log`. A one line summary appended to `logs/setup/first_kill.txt` with the two peak memory numbers.

**Success Criteria.**

* Both runs complete the configured number of steps without NaN loss.
* The Jenga peak memory in GB is strictly less than the baseline peak memory in GB. If equal or higher, this atom fails and execution halts pending diagnosis.

**Report Update.** Section 3 Reproduction Methodology gains a one line note confirming the first kill criterion passed and citing the two peak memory numbers.

**Budget.** Approximately 10 minutes of GPU time. Estimate 0.2 hours.

---

# Category: Reproduction

### Atom R1: Figure 12 End to End Memory Footprint

**Purpose.** Reproduce the paper's Figure 12 grouped bar comparison of peak memory across LoRA, LongLoRA, and Jenga at a representative subset of models and sequence lengths.

**Depends On.** S4.

**Inputs.**

* Models: `checkpoints/llama2` and `checkpoints/opt-1.3b`.
* Sequence lengths: 4096 and 8192.
* Methods: `llama-base.sh`, `llama-llora.sh`, `llama-jenga.sh` for Llama 2 and the matching OPT scripts.

**Steps.**

1. For each `(model, seq_len, method)` in the cross product of `{llama2, opt-1.3b}` × `{4096, 8192}` × `{base, llora, jenga}` run the corresponding `scripts/end2end-memory/<model_family>-<method>.sh <model> <seq_len>`. Enable `gradient_checkpoint True` on baseline LoRA at 8192 if it OOMs.
2. Confirm each run wrote a log to `logs/end2end/memory/<method>/<model>_<seq_len>.log`.
3. Run `python src/experiment/end2end/memory/plot_comparison_4k.py` and `python src/experiment/end2end/memory/plot_comparison_8k.py`.

**Outputs.**

* Twelve log files under `logs/end2end/memory/`.
* PDFs in `output_figures/end2end/memory/` matching panels of paper Figure 12.

**Success Criteria.**

* All twelve runs produce a peak memory number without OOM (with gradient checkpointing allowed on baseline at 8192).
* Qualitatively, for every `(model, seq_len)` pair: Jenga peak memory is below LoRA peak memory. LongLoRA is above Jenga and below or comparable to LoRA.
* The generated PDFs are visually consistent in shape with the corresponding panels in `output_figures/end2end/memory/` produced by Atom S3 from the shipped logs.

**Report Update.** Section 4.1 receives the new grouped bar PDF and a one paragraph summary stating the observed memory reduction percentage of Jenga over LoRA at each `(model, seq_len)`. Budget ledger updated with actual hours.

**Budget.** Twelve configurations at roughly 7 to 10 minutes each. Estimate 1.5 hours.

### Atom R2: Figure 13 End to End Execution Time

**Purpose.** Reproduce the paper's Figure 13 grouped bar comparison of step time across LoRA, LongLoRA, and Jenga at the same representative subset used in R1.

**Depends On.** R1.

**Inputs.** Same models, sequence lengths, and method scripts as R1, but pointed at `scripts/end2end-time/...`.

**Steps.**

1. For each `(model, seq_len, method)` from R1 run the corresponding `scripts/end2end-time/<model_family>-<method>.sh <model> <seq_len> False a800` (the device argument is just a log file label).
2. Confirm each run wrote a log to `logs/end2end/time/<model>-<seq_len>-a800.log`.
3. Run `python src/experiment/end2end/time/plot_comparison_a800.py` and `python src/experiment/end2end/time/plot_sequence.py`.

**Outputs.**

* Twelve log files under `logs/end2end/time/`.
* PDFs in `output_figures/end2end/time/` matching panels of paper Figure 13.

**Success Criteria.**

* All twelve runs complete. Each log contains step time numbers.
* Qualitatively, for every `(model, seq_len)` pair: Jenga step time is below LoRA step time.
* PDFs are visually consistent in shape with the S3 versions.

**Report Update.** Section 4.2 receives the new grouped bar PDF and a one paragraph summary of observed speedup at each `(model, seq_len)`, including mean and standard deviation across the measured steps. Budget ledger updated.

**Budget.** Twelve configurations at roughly 7 to 10 minutes each. Estimate 1.5 hours.

### Atom R3: Figure 14 Upper Memory Breakdown

**Purpose.** Reproduce the paper's Figure 14 upper stacked bar memory breakdown isolating model state, activations, predictor overhead, and other components on Llama 2 7B at three sequence lengths.

**Depends On.** R2.

**Inputs.** Llama 2 7B at sequence lengths 8192, 12288, 16384. The memory breakdown driver under `src/experiment/ablation/memory-breakdown/` and orchestration in `scripts/ablation-mem-breakdown/run.sh`.

**Steps.**

1. Run `bash scripts/ablation-mem-breakdown/run.sh`. If the script iterates more configurations than the three sequence lengths above, restrict to those.
2. Confirm logs land under `logs/ablations/memory-breakdown/`.
3. Run `python src/experiment/ablation/memory-breakdown/plot.py`.

**Outputs.**

* Log files under `logs/ablations/memory-breakdown/`.
* `output_figures/ablations/memory-breakdown/*.pdf` reproducing the stacked bar layout.

**Success Criteria.**

* The activations component shrinks as the sequence length grows in proportion to the sparsity ratio.
* The predictor overhead is below five percent of total peak memory across all three lengths.
* The generated PDF is visually consistent with the S3 baseline.

**Report Update.** Section 4.3 receives the new stacked bar PDF and a one paragraph summary listing the activations fraction at 8K, 12K, 16K, plus the predictor overhead percentage. Budget ledger updated.

**Budget.** Three configurations at roughly 10 minutes each. Estimate 0.5 hours.

### Atom R4: Figure 18 Segment Based Peak Cutting

**Purpose.** Reproduce the paper's Figure 18 demonstration that segmenting the final cross entropy computation reduces the terminal loss memory spike for large vocabulary models.

**Depends On.** R3.

**Inputs.** Llama 2 7B at sequence length 16384. The segment ablation driver under `src/experiment/ablation/segment/` and the orchestration `scripts/ablation-segment/run.sh`.

**Steps.**

1. Run `bash scripts/ablation-segment/base.sh` to produce the naive baseline pickle.
2. Run `bash scripts/ablation-segment/segment.sh` to produce the segmented variant pickle.
3. Drag the two pickle files into `docs.pytorch.org/memory_viz` for visualization. Capture screenshots into `output_figures/ablations/segment/`.

**Outputs.**

* Two pickle files under `logs/ablations/segment/`.
* Two screenshots under `output_figures/ablations/segment/` showing the memory profile pre and post segmentation.

**Success Criteria.**

* The segmented variant's peak loss memory is below the naive baseline's peak loss memory.
* The relative reduction is in the rough neighborhood of fifteen percent as claimed by the paper.

**Report Update.** Section 4.5 receives the two screenshots and a one paragraph summary citing the relative reduction.

**Budget.** Approximately 10 minutes of GPU time. Estimate 0.2 hours.

### Atom R5: Table 7 Perplexity on Paper Benchmarks

**Purpose.** Reproduce the paper's Table 7 perplexity comparison of LoRA versus Jenga on the paper's own `proof_pile.bin` and `test_pg19.bin` files at a budget feasible subset of configurations.

**Depends On.** R4.

**Inputs.** `dataset/PPL/proof_pile.bin` and `dataset/PPL/test_pg19.bin`. Llama 2 7B baseline LoRA adapter from `checkpoints/peft_model/la/lora/` and Jenga LoRA adapter from `checkpoints/peft_model/la/jenga/`. Sequence lengths trimmed under budget pressure: **proof_pile.bin at 8K and 16K, test_pg19.bin at 8K only** (16K pg dropped because pg windows are ~2.7x larger than pp and 16K pg on A6000 would take ~30 minutes per eval). Single seed per configuration on budget grounds; the three seed requirement was relaxed when spot preemptions made the 20 eval matrix impractical.

**Steps.**

1. Run `bash scripts/end2end-ppl/ppl_pp.sh` (proof pile). If the script iterates more configurations than the subset above, restrict to those.
2. Run `bash scripts/end2end-ppl/ppl_pg.sh` (pg).
3. Aggregate results into a single table file.

**Outputs.**

* `logs/end2end/accuracy/ppl_results.json` with schema `{"model": "llama2", "method": "<lora|jenga>", "seq_len": <int>, "benchmark": "<proof_pile|pg>", "seed": <int>, "ppl": <float>}`.
* A markdown table at `logs/end2end/accuracy/table7.md` cross referencing `(method, seq_len, benchmark)` with mean and standard deviation across seeds.

**Success Criteria.**

* All twelve runs (two methods times two benchmarks times two lengths times three seeds equals twenty four with three seeds; in practice run one seed for a first pass then add seeds if budget remains) complete without crash.
* Jenga perplexity is within roughly five percent of LoRA perplexity at each `(seq_len, benchmark)` pair.
* The table file exists and is renderable as a LaTeX table later.

**Report Update.** Section 4.4 receives the markdown table along with one paragraph stating whether the equal-perplexity claim of the paper held qualitatively. Budget ledger updated.

**Budget.** Largest single atom in the reproduction phase due to long evaluation runs. Estimate 2 hours.

### Atom R6: Figure 15 Algorithm Ablation (Optional)

**Purpose.** Reproduce the paper's Figure 15 algorithm ablation isolating whether the speedup comes from attention sparsity or MLP sparsity. Cheap and directly informs Extension B framing.

**Depends On.** R5.

**Inputs.** The ablation algorithm scripts under `scripts/ablation-algorithm/` and plotters under `src/experiment/ablation/algorithm/`.

**Steps.**

1. `bash scripts/ablation-algorithm/run.sh`.
2. `python src/experiment/ablation/algorithm/plot_llama2_attn.py`
3. `python src/experiment/ablation/algorithm/plot_llama2_mlp.py`
4. `python src/experiment/ablation/algorithm/plot_opt_attn.py`
5. `python src/experiment/ablation/algorithm/plot_opt_mlp.py`

**Outputs.** PDFs in `output_figures/ablations/algorithm/`.

**Success Criteria.** PDFs visually consistent with the S3 baseline.

**Report Update.** Section 4.6 gains a sub paragraph noting which component (attention or MLP) contributes more to the speedup.

**Budget.** Approximately 10 minutes of GPU time. Estimate 0.2 hours.

### Atom R7: Figure 16 Predictor Convergence (Optional, Budget Gated)

**Purpose.** Reproduce the paper's Figure 16 predictor training loss curve to validate the rapid convergence claim. This atom is skipped if the remaining buffer is below 3 GPU hours after R6.

**Depends On.** R6.

**Inputs.** Predictor training scripts under `scripts/ablation-predictor/` and drivers under `src/experiment/ablation/predictor/`.

**Steps.**

1. `bash scripts/ablation-predictor/run.sh`.
2. `python src/experiment/ablation/predictor/plot_loss.py`
3. `python src/experiment/ablation/predictor/elastic_size.py > logs/ablations/predictor/elastic_size.log`

**Outputs.** PDFs in `output_figures/ablations/predictor/` and the elastic size log.

**Success Criteria.** Loss curve drops sharply within the first one hundred epochs and asymptotes by epoch four hundred.

**Report Update.** Section 4.6 sub paragraph on predictor convergence is populated. If this atom is skipped, Section 4.6 cites the paper Figure 16 directly and discloses the skip.

**Budget.** Three hours. Skip if buffer is below this.

---

# Category: Improvement

### Atom I1: Extension A Dynamic Adaptive Thresholds

**Purpose.** Implement and evaluate a runtime entropy heuristic that overrides the static `config.thresh` per batch. Produce one scatter figure of predictor entropy versus token retention ratio and one bar chart of perplexity across swept lambda values, both compared against an equal budget retrained Jenga baseline.

**Depends On.** R5. (R6 and R7 are not prerequisites.)

**Inputs.** `src/jenga/models/modeling_llama.py` and `src/jenga/ops/get_meta.py` where the static threshold is applied. Llama 2 7B at sequence length 8192. `lam in {0.05, 0.1, 0.2}`. Three seeds per setting.

**Steps.**

1. Implement `CNNAttnPredictor`-class plumbing only for thresholds: extend `modeling_llama.py` to accept a `dynamic_threshold_lambda` config field. Plumb it through to the call site where the threshold is consumed.
2. In the forward pass after the predictor scores are computed, compute the normalized Shannon entropy of the positive informativeness scores per layer per batch.
3. Compute `t_dynamic = t_base + lam * (1 - entropy_norm)` and use it in place of the static threshold.
4. Add a CLI argument `--dynamic_threshold_lambda` to `src/experiment/end2end/time/llama_jenga.py` and surface it via a new shell wrapper `scripts/extension-adaptive/run.sh <lam>`.
5. Retrain a Jenga baseline (static threshold) under exactly the same training budget that the adaptive runs will use. This is the fair comparison baseline.
6. For each `lam in {0.05, 0.1, 0.2}` and each seed in `{0, 1, 2}` run the adaptive driver until the training budget cap is hit. Log retention ratio per layer per batch and final PPL on `proof_pile.bin` and `test_pg19.bin`.
7. Plot the entropy versus retention scatter and the PPL bar chart.

**Outputs.**

* `logs/extensions/adaptive_thresholds/retention.json` with schema `{"lam": <float>, "seed": <int>, "batch_idx": <int>, "layer_idx": <int>, "entropy": <float>, "retention_ratio": <float>}`.
* `logs/extensions/adaptive_thresholds/ppl.json` with schema `{"method": "jenga_adaptive", "lam": <float>, "seed": <int>, "ppl_proof_pile": <float>, "ppl_pg": <float>}` and matching rows for the equal budget static baseline.
* `output_figures/extensions/adaptive_thresholds/scatter.pdf` and `output_figures/extensions/adaptive_thresholds/ppl_bar.pdf`.

**Success Criteria.**

* Adaptive driver completes for all three `lam` values across three seeds without NaN loss.
* The scatter plot is generated and exhibits a non flat relationship (either positive or negative correlation; flat means the heuristic does not respond to input).
* The PPL bar chart includes the equal budget retrained static baseline so the comparison is fair.

**Report Update.** Section 5.1 gains the implementation description and the formula. Section 6.1 gains the two PDFs and a paragraph on whether the entropy heuristic produced any directional improvement and on the observed sensitivity to `lam`. Budget ledger updated.

**Budget.** Three hours covering one static baseline retrain plus three lambda settings times three seeds at reduced step count. Estimate 3 hours.

### Atom I2: Extension B 1D CNN Predictor

**Purpose.** Implement a 1D convolutional alternative to the per block MLP predictor and compare offline training convergence (mean squared error against the dense attention ground truth) at equal epoch budget. Produce one dual line plot of MSE versus epoch.

**Depends On.** I1.

**Inputs.** `src/jenga/models/predictor.py`. The offline predictor training harness under `src/experiment/ablation/predictor/`. RedPajama subset.

**Steps (as built).**

1. `CNNAttnPredictor` added to `src/jenga/models/predictor.py`. Two `nn.Conv1d` layers (`kernel_size=3`, `padding=1`), ReLU between them, final linear projection. Convolutional axis is the block index.
2. Self contained driver `src/experiment/extension_cnn_predictor/train_both.py` written instead of reusing the heavy HF Trainer harness. Driver caches `(hidden_state, pooled_attention_score)` per layer from a frozen base model, then trains MLP and CNN predictors on the cache with three seeds each.
3. **Base model deviation: OPT-1.3B at sequence length 2048** instead of Llama 2 7B. Reason: Llama 2 7B with `output_attentions=True` materialises ~32 GB of per-layer attention tensors and OOMs even on a 48 GB GPU. The Jenga predictor is model agnostic; the head to head MSE comparison is preserved.
4. Plot a dual line chart of epoch versus MSE loss.

**Outputs.**

* `logs/extensions/cnn_predictor/loss.csv` columns `epoch,seed,predictor_type,train_loss`.
* `output_figures/extensions/cnn_predictor/loss_curve.pdf`.

**Success Criteria.**

* Both predictor variants complete training without NaN.
* The plot includes mean curves and error bands across seeds.

**Report Update.** Section 5.2 gains the implementation description. Section 6.2 gains the loss curve PDF and a paragraph on convergence speed and final asymptotic MSE comparison.

**Budget.** Two hours. Estimate 2 hours.

---

# Category: Reporting

### Atom P1: Narrative Sections Draft

**Purpose.** Draft the markdown for the sections of REPORT.md that do not depend on a specific figure or table: Introduction, Background, Methodology, Discussion, Limitations, Conclusion.

**Depends On.** I2.

**Inputs.** The current populated `REPORT.md` plus the budget ledger and risk register in PLAN.md.

**Steps.**

1. Write Section 1 Introduction summarizing Jenga and the scope of our reproduction and extensions.
2. Write Section 2 Background covering long context fine tuning, LoRA, sparse attention, and Jenga's position.
3. Finalize Section 3 Methodology with hardware, software pins, model and dataset choices, measurement protocol, and disclosed deviations.
4. Write Section 7 Discussion including a dedicated negative results paragraph.
5. Write Section 8 Limitations listing single seed limits, budget cap, training step caps, dataset substitutions, and any skipped atoms.
6. Write Section 9 Conclusion.

**Outputs.** Updated `REPORT.md`.

**Success Criteria.** No section in `REPORT.md` other than the abstract and references is marked empty.

**Report Update.** This atom is itself the report update.

**Budget.** No GPU. Estimate 0 hours.

### Atom P2: LaTeX Conversion and Final Polish

**Purpose.** Convert `REPORT.md` into a six to eight page LaTeX submission. Tighten figures, add citations, write the abstract last.

**Depends On.** P1.

**Inputs.** Final `REPORT.md`. Course style file or default `article` class.

**Steps.**

1. Create `report.tex` from `REPORT.md` using a standard conference template.
2. Place figures in a `report/figures/` directory and reference each with a stable label.
3. Add a BibTeX file with the Jenga paper, the artifact, LoRA, QLoRA, Flash Attention, LongLoRA, and the Hugging Face libraries.
4. Write the abstract last.
5. Compile to PDF. Run a final readthrough.

**Outputs.** `report/report.tex`, `report/refs.bib`, `report/report.pdf`.

**Success Criteria.**

* Page count is between six and eight excluding references.
* All figures referenced in the LaTeX exist in `report/figures/`.
* No `\cite{?}` or `[TODO]` markers remain.

**Report Update.** Final populated `REPORT.md` plus the compiled PDF.

**Budget.** No GPU. Estimate 0 hours.

---

## Honesty Constraints

* Every reproduction claim that says "matches the paper" must be backed by a numeric column from a paper benchmark, not a substituted one.
* Every extension claim that compares against Jenga must compare against an equal budget retrained Jenga baseline.
* Drift from paper numbers is reported, not hidden.
