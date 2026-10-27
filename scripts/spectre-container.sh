#!/usr/bin/env bash
# Spectre server (Whisper + Kokoro in-process) for a CUDA container / Proxmox LXC with the GPU passed through.
#
#   scripts/spectre-container.sh setup      # once: system packages, Python venv, models, config file
#   scripts/spectre-container.sh selftest   # GPU check: Kokoro speaks, Whisper hears it back, LLM answers
#   scripts/spectre-container.sh add-chatterbox    # optional: install the Chatterbox-Turbo voice too
#   scripts/spectre-container.sh selftest --compare  # time Kokoro vs Chatterbox, save WAVs to listen to
#   scripts/spectre-container.sh undo-chatterbox   # put the environment back as it was before add-chatterbox
#   scripts/spectre-container.sh bench-chatterbox  # Chatterbox speed: original vs fast loop, with/without LLM load
#   scripts/spectre-container.sh run        # start the server (port 8765)
#   scripts/spectre-container.sh service    # install + start a systemd unit so it survives reboots
#
# Settings live in spectre.env at the repo root (created by setup; edit SPECTRE_LLM_URL).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER="$ROOT/server"
PY="$SERVER/.venv/bin/python"
ENV_FILE="${SPECTRE_ENV_FILE:-$ROOT/spectre.env}"

log() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mWARN:\033[0m %s\n' "$*" >&2; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

check_gpu() {
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found: is the GPU passed through to this container?"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
    || die "nvidia-smi can't talk to the driver (check the LXC device passthrough)"
}

load_env() {
  [ -f "$ENV_FILE" ] || die "$ENV_FILE missing: run '$0 setup' first"
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
  # faster-whisper (CTranslate2) needs cuBLAS + cuDNN; use the copies pip installed alongside PyTorch,
  # so the container doesn't need a matching system CUDA toolkit.
  local nv
  nv="$("$PY" - <<'EOF' 2>/dev/null || true
import importlib
dirs = []
for mod in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
    try:
        dirs.append(list(importlib.import_module(mod).__path__)[0])
    except Exception:
        pass
print(":".join(dirs))
EOF
)"
  [ -n "$nv" ] && export LD_LIBRARY_PATH="$nv${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  return 0
}

cmd_setup() {
  log "Checking GPU"
  check_gpu

  if command -v apt-get >/dev/null; then
    log "Installing system packages (espeak-ng is Kokoro's fallback for unknown words)"
    $SUDO apt-get update -qq
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq espeak-ng curl ca-certificates git >/dev/null
  else
    warn "no apt-get: install espeak-ng and curl yourself"
  fi

  if ! command -v uv >/dev/null; then
    log "Installing uv (Python package manager)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi

  log "Creating Python 3.12 venv"
  cd "$SERVER"
  # --seed adds pip: misaki (Kokoro's phonemiser) shells out to pip to fetch its spaCy model.
  [ -x "$PY" ] || uv venv --seed --python 3.12 .venv
  log "Installing Spectre + faster-whisper + Kokoro (PyTorch with CUDA is a few GB, be patient)"
  uv pip install --python "$PY" -e '.[gpu,kokoro,dev]'
  "$PY" -m spacy download en_core_web_sm >/dev/null || warn "spaCy model download failed; Kokoro will retry at startup"

  log "Checking PyTorch and CTranslate2 can see the GPU"
  "$PY" - <<'EOF'
import ctranslate2, torch
print(f"  torch {torch.__version__} (CUDA {torch.version.cuda}) gpu available: {torch.cuda.is_available()}")
print(f"  ctranslate2 {ctranslate2.__version__} cuda devices: {ctranslate2.get_cuda_device_count()}")
if not torch.cuda.is_available():
    raise SystemExit("PyTorch can't use the GPU. Usually the driver is older than this PyTorch's CUDA build: "
                     "check `nvidia-smi` (driver) vs the CUDA version above.")
EOF

  if [ ! -f "$ENV_FILE" ]; then
    log "Writing $ENV_FILE"
    cat >"$ENV_FILE" <<'EOF'
# Spectre settings (see server/app/config.py for everything available).
# Where llama-server lives (the other container). Must listen on 0.0.0.0, not 127.0.0.1.
SPECTRE_LLM_URL=http://CHANGE-ME-llm-container:8080/v1

SPECTRE_STT_MODEL=large-v3-turbo
SPECTRE_STT_DEVICE=cuda
SPECTRE_STT_COMPUTE_TYPE=float16

SPECTRE_TTS_BACKEND=kokoro-local
SPECTRE_TTS_VOICE=bm_fable
SPECTRE_TTS_DEVICE=cuda
# After `add-chatterbox`: SPECTRE_TTS_BACKEND=chatterbox, and optionally a 10 s+ WAV of the voice to clone
# (yours, or one you have permission to use). Empty = Chatterbox's built-in voice.
#SPECTRE_TTS_VOICE_REF=/root/boolian-voice.wav

SPECTRE_PORT=8765
# Optional shared secret; then open the page with ?token=...
#SPECTRE_AUTH_TOKEN=
EOF
  fi

  log "Done. Next:"
  echo "   1. edit SPECTRE_LLM_URL in $ENV_FILE"
  echo "   2. $0 selftest     (downloads Whisper + Kokoro weights the first time, ~2 GB)"
  echo "   3. $0 run          or   $0 service"
}

cmd_selftest() {
  load_env
  check_gpu
  cd "$SERVER"
  exec "$PY" "$ROOT/scripts/selftest_gpu.py" "$@"
}

# Chatterbox pins torch 2.6 / transformers 5.2, which replaces the versions Kokoro and Whisper were set up with.
# Keep a copy of the working venv so undo-chatterbox is instant if anything breaks.
BACKUP="$SERVER/.venv-before-chatterbox"

cmd_add_chatterbox() {
  [ -x "$PY" ] || die "run '$0 setup' first"
  export PATH="$HOME/.local/bin:$PATH"
  if [ ! -d "$BACKUP" ]; then
    log "Backing up the current environment to $BACKUP"
    cp -a "$SERVER/.venv" "$BACKUP"
  fi
  log "Installing Chatterbox-Turbo (downgrades torch to 2.6; a few GB)"
  cd "$SERVER"
  uv pip install --python "$PY" -e '.[gpu,kokoro,chatterbox,dev]'
  "$PY" - <<'EOF'
import ctranslate2, torch
from chatterbox.tts_turbo import ChatterboxTurboTTS  # noqa: F401
print(f"  torch {torch.__version__} (CUDA {torch.version.cuda}) gpu available: {torch.cuda.is_available()}")
print(f"  ctranslate2 {ctranslate2.__version__} cuda devices: {ctranslate2.get_cuda_device_count()}")
EOF
  log "Done. Next: $0 selftest --compare   (then set SPECTRE_TTS_BACKEND=chatterbox in spectre.env to use it)"
}

cmd_bench_chatterbox() {
  load_env
  check_gpu
  cd "$SERVER"
  exec "$PY" "$ROOT/scripts/bench_chatterbox.py"
}

cmd_undo_chatterbox() {
  [ -d "$BACKUP" ] || die "no backup at $BACKUP: add-chatterbox was never run"
  log "Restoring the environment from before add-chatterbox"
  rm -rf "$SERVER/.venv"
  mv "$BACKUP" "$SERVER/.venv"
  grep -q '^SPECTRE_TTS_BACKEND=chatterbox' "$ENV_FILE" 2>/dev/null \
    && warn "spectre.env still says SPECTRE_TTS_BACKEND=chatterbox: set it back to kokoro-local"
  log "Done. Restart the server (systemctl restart spectre)"
}

cmd_run() {
  load_env
  cd "$SERVER"
  log "Starting Spectre on port ${SPECTRE_PORT:-8765} (LLM at ${SPECTRE_LLM_URL:-?})"
  exec "$PY" -m app.main
}

cmd_service() {
  [ -d /run/systemd/system ] || die "systemd isn't running in this container; use '$0 run' (e.g. in tmux) instead"
  local unit=/etc/systemd/system/spectre.service
  log "Installing $unit"
  $SUDO tee "$unit" >/dev/null <<EOF
[Unit]
Description=Spectre ghost server (Whisper + Kokoro)
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=$ROOT/scripts/spectre-container.sh run
Restart=always
RestartSec=3
User=$(id -un)
Environment=HOME=$HOME

[Install]
WantedBy=multi-user.target
EOF
  $SUDO systemctl daemon-reload
  $SUDO systemctl enable --now spectre.service
  log "Running. Logs: journalctl -fu spectre"
}

case "${1:-}" in
  setup) cmd_setup ;;
  selftest) shift; cmd_selftest "$@" ;;
  add-chatterbox) cmd_add_chatterbox ;;
  undo-chatterbox) cmd_undo_chatterbox ;;
  bench-chatterbox) cmd_bench_chatterbox ;;
  run) cmd_run ;;
  service) cmd_service ;;
  *) sed -n '2,13p' "$0"; exit 1 ;;
esac
