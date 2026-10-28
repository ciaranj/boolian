#!/usr/bin/env bash
# llama-server for Boolian's brain, in a CUDA container / Proxmox LXC with the GPU passed through.
#
#   scripts/llm-container.sh setup     # build llama.cpp with CUDA (skipped if llama-server already exists)
#   scripts/llm-container.sh run       # serve the model on 0.0.0.0:8080
#   scripts/llm-container.sh service   # install + start a systemd unit
#
# Choose the model with either (env vars or llm.env next to this repo's root):
#   LLM_HF=unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL   # downloaded from Hugging Face on first run
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

install_cuda_toolkit() {
  # The compiler only, from NVIDIA's apt repo. Never the driver: in an LXC that comes from the Proxmox host.
  . /etc/os-release
  local repo="${ID}${VERSION_ID//./}"  # ubuntu2204, ubuntu2404, debian12, ...
  local url="https://developer.download.nvidia.com/compute/cuda/repos/$repo/x86_64/cuda-keyring_1.1-1_all.deb"
  log "Installing the CUDA toolkit (nvcc) from NVIDIA's $repo repo, a few GB"
  curl -fsSL -o /tmp/cuda-keyring.deb "$url" || die "NVIDIA has no CUDA repo for $repo ($url). \
Install cuda-toolkit by hand, or use an Ubuntu 22.04/24.04 or Debian 12 container."
  $SUDO dpkg -i /tmp/cuda-keyring.deb >/dev/null
  $SUDO apt-get update -qq
  # 12.8 if the repo has it (works with any driver >= 570); otherwise whatever the repo's newest is.
  local pkg=cuda-toolkit-12-8
  apt-cache show "$pkg" >/dev/null 2>&1 || pkg=cuda-toolkit
  $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "$pkg" >/dev/null
  export PATH="/usr/local/cuda/bin:$PATH"
  log "nvcc: $(nvcc --version | tail -1)"
}

cmd_setup() {
  command -v nvidia-smi >/dev/null || die "nvidia-smi not found: is the GPU passed through?"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

  if bin="$(find_server)"; then
    log "llama-server already installed: $bin"
  else
    export PATH="/usr/local/cuda/bin:$PATH"
    if command -v apt-get >/dev/null; then
      log "Installing build tools"
      $SUDO apt-get update -qq
      $SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential cmake git curl ca-certificates libcurl4-openssl-dev libssl-dev >/dev/null
      command -v nvcc >/dev/null || install_cuda_toolkit
    fi
    command -v nvcc >/dev/null || die "building llama.cpp with CUDA needs nvcc. \
Use a CUDA *devel* image (e.g. nvidia/cuda:12.4.1-devel-ubuntu22.04) or install NVIDIA's cuda-toolkit package."
    [ -d "$LLAMA_DIR" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
    log "Building llama.cpp with CUDA (several minutes)"
    # native: compile only for the GPU in this box, which is much faster than llama.cpp's default list.
    cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DGGML_CUDA=ON -DLLAMA_CURL=ON -DLLAMA_OPENSSL=ON -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=native >/dev/null
    cmake --build "$LLAMA_DIR/build" --config Release -j "$(nproc)" --target llama-server
    log "Built $LLAMA_DIR/build/bin/llama-server (commit $(git -C "$LLAMA_DIR" rev-parse --short HEAD))"
  fi

  if [ ! -f "$ENV_FILE" ]; then
    log "Writing $ENV_FILE"
    cat >"$ENV_FILE" <<'EOF'
# Model for Boolian. Gemma 4 12B at 4-bit is ~7.5 GB, leaving room on a 16 GB card for Whisper + Kokoro.
# Bigger (e.g. unsloth/gemma-4-26B-A4B-it-GGUF) only fits if this GPU isn't shared with the spectre container.
LLM_HF=unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL
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
  else model_args=(-hf "${LLM_HF:-unsloth/gemma-4-12b-it-GGUF:UD-Q4_K_XL}")
  fi
  log "Serving ${model_args[1]} on 0.0.0.0:${LLM_PORT:-8080}"
  # --parallel 1: one conversation at a time, all the KV cache for it. --jinja: model's own chat template.
  # No thinking (a reasoning preamble would cost seconds per reply) and no vision projector (wasted VRAM).
  # --swa-full: Gemma's sliding-window cache otherwise makes reusing the cached system prompt slower
  # than reprocessing it (~400 ms per turn); the full cache fits at 4k context (8k runs out of VRAM beside the spectre container).
  # --cache-ram 0: newer builds keep a copy of each finished conversation's cache in system RAM (up to 8 GB), which
  # with --swa-full is the whole 4k cache per visitor group; it filled the container's memory over an evening and
  # slowed everything to a crawl. One conversation at a time reuses the live cache anyway. (Older builds lack the flag.)
  local cache_args=()
  case "$("$bin" --help 2>&1)" in *--cache-ram*) cache_args=(--cache-ram 0) ;; esac
  exec "$bin" "${model_args[@]}" --host 0.0.0.0 --port "${LLM_PORT:-8080}" \
    -ngl 99 -c "${LLM_CTX:-4096}" --parallel 1 --jinja --no-mmproj --swa-full "${cache_args[@]}" \
    --chat-template-kwargs '{"enable_thinking":false}'
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
