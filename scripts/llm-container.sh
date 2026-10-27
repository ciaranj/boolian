#!/usr/bin/env bash
# llama-server for Boolian's brain, in a CUDA container / Proxmox LXC with the GPU passed through.
#
#   scripts/llm-container.sh setup     # build llama.cpp with CUDA (skipped if llama-server already exists)
#   scripts/llm-container.sh run       # serve the model on 0.0.0.0:8080
#   scripts/llm-container.sh service   # install + start a systemd unit
#
# Choose the model with either (env vars or llm.env next to this repo's root):
#   LLM_HF=bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M   # downloaded from Hugging Face on first run
#   LLM_MODEL=/models/some-model.gguf                          # a local GGUF file (takes precedence)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${LLM_ENV_FILE:-$ROOT/llm.env}"
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"

log() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }
SUDO=""
[ "$(id -u)" -ne 0 ] && SUDO="sudo"

find_server() {
  if command -v llama-server >/dev/null; then command -v llama-server
  elif [ -x "$LLAMA_DIR/build/bin/llama-server" ]; then echo "$LLAMA_DIR/build/bin/llama-server"
  else return 1
  fi
}

cmd_setup() {
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found: is the GPU passed through?"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

  if bin="$(find_server)"; then
    log "llama-server already installed: $bin"
  else
    command -v nvcc >/dev/null || [ -x /usr/local/cuda/bin/nvcc ] || die "building llama.cpp with CUDA needs nvcc. \
Use a CUDA *devel* image (e.g. nvidia/cuda:12.4.1-devel-ubuntu22.04) or install the cuda-toolkit package."
    export PATH="/usr/local/cuda/bin:$PATH"
    if command -v apt-get >/dev/null; then
      log "Installing build tools"
      $SUDO apt-get update -qq
      $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential cmake git libcurl4-openssl-dev >/dev/null
    fi
    [ -d "$LLAMA_DIR" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
    log "Building llama.cpp with CUDA (several minutes)"
    cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=ON -DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build "$LLAMA_DIR/build" --config Release -j "$(nproc)" --target llama-server
    log "Built $LLAMA_DIR/build/bin/llama-server (commit $(git -C "$LLAMA_DIR" rev-parse --short HEAD))"
  fi

  if [ ! -f "$ENV_FILE" ]; then
    log "Writing $ENV_FILE"
    cat >"$ENV_FILE" <<'EOF'
# Model for Boolian. 8B fits easily in 16 GB and stays in character better than 3B.
LLM_HF=bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M
# Or a local file (wins over LLM_HF):
#LLM_MODEL=/models/model.gguf
LLM_PORT=8080
LLM_CTX=4096
EOF
  fi
  log "Done. Next: $0 run   (first run downloads the model)"
}

cmd_run() {
  # shellcheck disable=SC1090
  [ -f "$ENV_FILE" ] && { set -a; . "$ENV_FILE"; set +a; }
  bin="$(find_server)" || die "llama-server not found: run '$0 setup'"
  local model_args
  if [ -n "${LLM_MODEL:-}" ]; then model_args=(-m "$LLM_MODEL")
  else model_args=(-hf "${LLM_HF:-bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M}")
  fi
  log "Serving ${model_args[1]} on 0.0.0.0:${LLM_PORT:-8080}"
  # --parallel 1: one conversation at a time, all the KV cache for it. --jinja: model's own chat template.
  exec "$bin" "${model_args[@]}" --host 0.0.0.0 --port "${LLM_PORT:-8080}" \
    -ngl 99 -c "${LLM_CTX:-4096}" --parallel 1 --jinja
}

cmd_service() {
  [ -d /run/systemd/system ] || die "systemd isn't running here; use '$0 run' (e.g. in tmux)"
  local unit=/etc/systemd/system/spectre-llm.service
  $SUDO tee "$unit" >/dev/null <<EOF
[Unit]
Description=llama-server for Spectre
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=$ROOT/scripts/llm-container.sh run
Restart=always
RestartSec=3
User=$(id -un)
Environment=HOME=$HOME

[Install]
WantedBy=multi-user.target
EOF
  $SUDO systemctl daemon-reload
  $SUDO systemctl enable --now spectre-llm.service
  log "Running. Logs: journalctl -fu spectre-llm"
}

case "${1:-}" in
  setup) cmd_setup ;;
  run) cmd_run ;;
  service) cmd_service ;;
  *) sed -n '2,11p' "$0"; exit 1 ;;
esac
