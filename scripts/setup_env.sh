#!/usr/bin/env bash
# Idempotent env bootstrap for Claude Code cloud / fresh containers. Safe to run repeatedly.
# CPU JAX by default; set WITH_CUDA=1 on an NVIDIA host. Also enables the tracked git hooks.
set -e
cd "$(dirname "$0")/.."
git config core.hooksPath .githooks 2>/dev/null || true
STAMP=".git/.requirements.sha"
HASH=$(sha256sum requirements.txt requirements-cuda.txt | sha256sum | cut -d' ' -f1)
if [ "$(cat "$STAMP" 2>/dev/null)" != "$HASH$WITH_CUDA" ]; then
  if [ "${WITH_CUDA:-0}" = "1" ]; then python -m pip install -q -r requirements-cuda.txt
  else python -m pip install -q -r requirements.txt; fi
  echo "$HASH$WITH_CUDA" > "$STAMP"
fi
