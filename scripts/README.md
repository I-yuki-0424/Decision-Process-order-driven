# scripts/ index (flat on purpose: scripts import each other as siblings; run from repo root)

- Lint/env: `update_requirements.py` (auto requirements.txt), `install_hooks.py`, `setup_env.sh`, `pre_commit_ban_check.py`, `check_gpu.py`, `set-remotes.ps1`, `tk_*.sh`, `print_banner.sh`
- Kaggle orchestration: `kaggle_run.py` (main), `kaggle_run_candidates_smoke.py`, `kaggle_run_experiment.py`, `kaggle_host_orchestrator.py`, `monitor_kaggle.py`, `build_remote_kaggle_notebook.py`, `make_notebook.py`
- Experiments (run): `run_candidate_experiment.py`, `run_worldmodel_experiment.py`, `run_idea6_policy.py`, `run_hybrid_worldmodel.py`, `run_partial_obs.py`, `run_physics_benchmark.py`, `run_wm_planning.py`, `real_end_to_end_pipeline.py`, `record_craftax_replays.py`
- Aggregation/plots: `aggregate_*.py`, `plot_idea6_summary.py`, `generate_benchmark_plots.py`
- `adhoc/` throwaway checks, not part of validation.
