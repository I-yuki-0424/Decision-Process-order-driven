#!/bin/bash
# usage: drun.sh <args to python in container>
MSYS_NO_PATHCONV=1 docker run --rm --gpus all -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e JAX_COMPILATION_CACHE_DIR=/workspace/.jaxcache -v "$(pwd -W):/workspace" -w /workspace dpod-local:latest python "$@"
