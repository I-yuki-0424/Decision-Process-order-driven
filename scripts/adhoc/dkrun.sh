#!/bin/bash
# usage: dkrun.sh <cmd...>  -- foreground docker run with GPU, worktree mounted, persistent JAX compile cache
cd "$(dirname "$0")/../.."
export GIT_COMMIT=$(git rev-parse HEAD)
MSYS_NO_PATHCONV=1 docker run --rm --gpus all -e GIT_COMMIT -e JAX_COMPILATION_CACHE_DIR=/workspace/.jaxcache -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-.6} -e PYTHONUNBUFFERED=1 -v "$(pwd -W):/workspace" -w /workspace dpod-local:latest "$@" 2>&1 | grep -v "^$" | grep -v "NVIDIA\|CUDA\|=====\|license\|Container image\|By pulling\|^A copy"
