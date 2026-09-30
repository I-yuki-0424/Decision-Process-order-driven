#!/bin/bash
# usage: dk.sh <container-name> <cmd...>   -- runs a command in dpod-local with GPU and the worktree mounted, detached
name=$1; shift
cd "$(dirname "$0")/../.."
MSYS_NO_PATHCONV=1 docker run -d --name "$name" --gpus all -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e XLA_PYTHON_CLIENT_MEM_FRACTION=${MEMFRAC:-.30} -e PYTHONUNBUFFERED=1 -v "$(pwd -W):/workspace" -w /workspace dpod-local:latest "$@"
