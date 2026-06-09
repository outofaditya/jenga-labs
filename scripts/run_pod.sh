#!/usr/bin/env bash
# Unattended Atom S1 bootstrap for a fresh Vast.ai or RunPod A100 80GB pod.
#
# Required env vars:
#   HF_TOKEN         A Hugging Face read scoped token. Must have access to any gated
#                    base models you list in INCLUDE_LLAMA2 or INCLUDE_LLAMA3.
#   HF_MIRROR_REPO   Your private HF dataset holding the pre extracted artifacts
#                    laid out as:
#                      checkpoints/peft_model/...
#                      checkpoints/predictor/...
#                      dataset/...
#
# Optional env vars (defaults shown):
#   REPO_DIR             /workspace/jenga-labs
#   REPO_URL             https://github.com/outofaditya/jenga-labs.git
#   INCLUDE_LLAMA2       1   set to 0 to skip the Llama 2 7B pull
#   INCLUDE_LLAMA3       1   set to 0 to skip Llama 3 8B (gated access required)
#   INCLUDE_OPT_350M     1
#   INCLUDE_OPT_1_3B     1
#   INCLUDE_OPT_2_7B     0
#   INCLUDE_OPT_6_7B     0   set to 1 to also pull OPT 6.7B
#
# Usage on the pod:
#   export HF_TOKEN=hf_...
#   export HF_MIRROR_REPO=<your_username>/jenga-labs-artifacts
#   bash scripts/run_pod.sh

set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN to your Hugging Face read token}"
: "${HF_MIRROR_REPO:?Set HF_MIRROR_REPO to your private mirror dataset id}"
REPO_DIR="${REPO_DIR:-/workspace/jenga-labs}"
REPO_URL="${REPO_URL:-https://github.com/outofaditya/jenga-labs.git}"
INCLUDE_LLAMA2="${INCLUDE_LLAMA2:-1}"
INCLUDE_LLAMA3="${INCLUDE_LLAMA3:-1}"
INCLUDE_OPT_350M="${INCLUDE_OPT_350M:-1}"
INCLUDE_OPT_1_3B="${INCLUDE_OPT_1_3B:-1}"
INCLUDE_OPT_2_7B="${INCLUDE_OPT_2_7B:-0}"
INCLUDE_OPT_6_7B="${INCLUDE_OPT_6_7B:-0}"

log()  { printf "\n[run_pod] %s\n" "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

ensure_bin() {
  local name="$1" alt="$2"
  if have "$name"; then return 0; fi
  if have "$alt"; then
    local shimdir="$HOME/.local/bin"
    mkdir -p "$shimdir"
    ln -sf "$(command -v "$alt")" "$shimdir/$name"
    case ":$PATH:" in
      *":$shimdir:"*) ;;
      *) export PATH="$shimdir:$PATH" ;;
    esac
    log "shimmed $name -> $(command -v "$alt")"
  else
    echo "[run_pod] neither $name nor $alt on PATH" >&2
    exit 1
  fi
}

# torch 2.1.2 requires Python 3.8 to 3.11. If the system python is outside that
# range, fall back to a conda env on Python 3.10 (the artifact's tested version).
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "")
ENV_BIN=""
case "$PYVER" in
  3.8|3.9|3.10|3.11)
    log "System python $PYVER is compatible with torch 2.1.2"
    ensure_bin python python3
    ensure_bin pip pip3
    ;;
  *)
    CONDA_BASE=""
    for cand in /opt/miniforge3 /opt/conda /opt/miniconda3 "$HOME/miniforge3" "$HOME/miniconda3"; do
      if [ -x "$cand/bin/conda" ]; then CONDA_BASE="$cand"; break; fi
    done
    if [ -z "$CONDA_BASE" ]; then
      echo "[run_pod] Python $PYVER incompatible with torch 2.1.2 and no conda found" >&2
      exit 1
    fi
    ENV_NAME="jenga"
    ENV_PREFIX=$("$CONDA_BASE/bin/conda" info --envs 2>/dev/null | awk -v n="$ENV_NAME" '$1==n {print $NF}')
    if [ -z "$ENV_PREFIX" ] || [ ! -x "$ENV_PREFIX/bin/python" ]; then
      log "Creating conda env $ENV_NAME (python 3.10) via $CONDA_BASE"
      "$CONDA_BASE/bin/conda" create -n "$ENV_NAME" -c conda-forge python=3.10 pip -y
      ENV_PREFIX=$("$CONDA_BASE/bin/conda" info --envs 2>/dev/null | awk -v n="$ENV_NAME" '$1==n {print $NF}')
    fi
    if [ -z "$ENV_PREFIX" ] || [ ! -x "$ENV_PREFIX/bin/python" ]; then
      echo "[run_pod] failed to locate conda env $ENV_NAME after create" >&2
      "$CONDA_BASE/bin/conda" info --envs >&2 || true
      exit 1
    fi
    ENV_BIN="$ENV_PREFIX/bin"
    export PATH="$ENV_BIN:$PATH"
    log "Activated $ENV_NAME at $ENV_BIN with $(python --version 2>&1)"
    ;;
esac

log "Verifying NVIDIA driver and CUDA visible"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
python -c "import sys; print('python', sys.version)"

if [ ! -d "$REPO_DIR/.git" ]; then
  log "Cloning $REPO_URL into $REPO_DIR"
  mkdir -p "$(dirname "$REPO_DIR")"
  git clone "$REPO_URL" "$REPO_DIR"
fi
cd "$REPO_DIR"

PIP_FLAGS=(--no-cache-dir --break-system-packages)

log "Python dependencies (requirements.txt)"
pip install "${PIP_FLAGS[@]}" -r requirements.txt

log "flash-attn (no build isolation)"
if ! python -c "import flash_attn" >/dev/null 2>&1; then
  pip install "${PIP_FLAGS[@]}" flash-attn --no-build-isolation
else
  log "flash-attn already importable, skipping"
fi

log "Editable install of jenga"
pip install "${PIP_FLAGS[@]}" -e .

log "huggingface_hub login"
python - <<'PY'
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
PY

mkdir -p checkpoints dataset

pull_mirror_path() {
  local subdir="$1"   # path inside the mirror repo
  local dest="$2"     # local destination
  if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
    log "Mirror path $subdir already populated at $dest, skipping"
    return
  fi
  log "Pulling mirror path $subdir into $dest"
  python - "$HF_MIRROR_REPO" "$subdir" "$dest" <<'PY'
import os, sys, shutil
from huggingface_hub import snapshot_download
repo, subdir, dest = sys.argv[1], sys.argv[2], sys.argv[3]
src = snapshot_download(
    repo_id=repo,
    repo_type="dataset",
    allow_patterns=[f"{subdir}/**"],
)
src_root = os.path.join(src, subdir)
os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
if os.path.isdir(dest):
    shutil.rmtree(dest)
shutil.copytree(src_root, dest, symlinks=True)
print(f"copied {src_root} -> {dest}")
PY
}

log "Pulling pre extracted artifacts from $HF_MIRROR_REPO"
pull_mirror_path "checkpoints/peft_model" "checkpoints/peft_model"
pull_mirror_path "checkpoints/predictor"  "checkpoints/predictor"
pull_mirror_path "dataset"                "dataset"

pull_base_model() {
  local hf_id="$1"
  local dest="$2"
  if [ -f "$dest/config.json" ]; then
    log "Base model $hf_id already present at $dest, skipping"
    return
  fi
  log "Downloading base model $hf_id -> $dest"
  python - "$hf_id" "$dest" <<'PY'
import sys
from huggingface_hub import snapshot_download
hf_id, dest = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=hf_id, local_dir=dest, local_dir_use_symlinks=False)
print(f"downloaded {hf_id} -> {dest}")
PY
}

if [ "$INCLUDE_LLAMA2" = "1" ]; then
  pull_base_model "meta-llama/Llama-2-7b-hf"     "checkpoints/llama2"
fi
if [ "$INCLUDE_LLAMA3" = "1" ]; then
  pull_base_model "meta-llama/Meta-Llama-3-8B"   "checkpoints/llama3"
fi
if [ "$INCLUDE_OPT_350M" = "1" ]; then
  pull_base_model "facebook/opt-350m"            "checkpoints/opt-350m"
fi
if [ "$INCLUDE_OPT_1_3B" = "1" ]; then
  pull_base_model "facebook/opt-1.3b"            "checkpoints/opt-1.3b"
fi
if [ "$INCLUDE_OPT_2_7B" = "1" ]; then
  pull_base_model "facebook/opt-2.7b"            "checkpoints/opt-2.7b"
fi
if [ "$INCLUDE_OPT_6_7B" = "1" ]; then
  pull_base_model "facebook/opt-6.7b"            "checkpoints/opt-6.7b"
fi

log "Sanity checks (Atom S1 success criteria)"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible'; print('cuda ok', torch.cuda.get_device_name(0))"
python -c "import flash_attn, jenga; print('flash_attn ok', flash_attn.__version__); print('jenga ok')"

required=(
  "checkpoints/predictor/predictor.pth"
  "checkpoints/predictor/pruned_config.pth"
  "dataset/PPL/proof_pile.bin"
  "dataset/PPL/test_pg19.bin"
)
if [ "$INCLUDE_LLAMA2" = "1" ]; then required+=("checkpoints/llama2/config.json"); fi
if [ "$INCLUDE_OPT_350M" = "1" ]; then required+=("checkpoints/opt-350m/config.json"); fi
if [ "$INCLUDE_OPT_1_3B" = "1" ]; then required+=("checkpoints/opt-1.3b/config.json"); fi

missing=0
for f in "${required[@]}"; do
  if [ ! -e "$f" ]; then
    echo "MISSING: $f"
    missing=$((missing+1))
  else
    printf "ok   %s\n" "$f"
  fi
done
if [ "$missing" -gt 0 ]; then
  echo "[run_pod] $missing required file(s) missing; aborting"
  exit 1
fi

log "Atom S1 bootstrap complete. Next: bash hello-world.sh (Atom S2)."
