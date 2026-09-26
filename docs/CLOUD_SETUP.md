# Dependencies & Claude Code cloud setup

`requirements.txt` (CPU JAX + all imports) is **auto-generated** by `scripts/update_requirements.py`
(AST scan of tracked `.py` files). Hand-edited version specifiers are preserved; new packages are unpinned.
`requirements-cuda.txt` = `-r requirements.txt` + `jax[cuda12]`.

## Automation (per clone, once)
```bash
python scripts/install_hooks.py   # sets core.hooksPath=.githooks (keeps the Git LFS hooks)
```
- `pre-commit`: regenerates and stages `requirements.txt`.
- `pre-push`: fails if `requirements.txt` is stale (`--check`), after the LFS push.

## Claude Code cloud (claude.ai/code) container
1. Environment settings → **Setup script**: `bash scripts/setup_env.sh` (runs when the container is created; cached afterwards).
2. Network access must allow PyPI (`pypi.org`, `files.pythonhosted.org`).
3. `.claude/settings.json` also runs the same script on `SessionStart`, so a fresh or resumed session self-heals.
4. Cloud has no GPU: CPU JAX is installed. Long GPU runs stay on Kaggle (ADR-002). Use `WITH_CUDA=1` only on NVIDIA hosts.
5. Git LFS: `git lfs install` if `git-lfs` is present; `output/**` is not needed for tests.

## Docker
`docker compose -f docker/compose.yaml up --build` installs `requirements-cuda.txt`.
