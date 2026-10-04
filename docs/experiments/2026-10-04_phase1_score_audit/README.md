# Phase-1 score reliability audit (TASK-20261004-022, 2026-10-04)

Operator request: re-evaluate whether the Phase-1 scores are reliable; document any fabrication, leak or rule violation and
correct it. Scope: every result file under `output/phase1/` (TASK-017 and TASK-020, 250 files), the code that produced them,
and the claims built on them. Machine checks: `python scripts/audit_phase1_results.py` (report: `audit_report.json` here).

## Verdict

| question | finding |
|---|---|
| Fabricated or edited numbers? | **None found.** In all 250 files every per-seed `reward_pct` and `score_pct` is reproduced exactly from the stored per-achievement rates, every rate is a whole number of the 256 evaluation episodes, every mean/SE matches its per-seed list, and no outcome is duplicated across different runs (160 groups of identical copies are the same run filed twice, e.g. `*_all` and `*_derived`). |
| Answer leakage / simulator access? | **None found.** The policy sees only the 1345-d observation and its own carry (`Arm.step(params, carry, obs)`); no env state reaches it. The world model is trained on executed transitions only; no hand-coded game table (`CRAFTAX_RESOURCE_EFFECTS`, 39-d adapters) is imported by the EP-A path; ChunkPPO k=1 uses arm `ignore`, δ = 0 (no oracle forecast). Evaluation uses a key stream disjoint from training, first episode, sampled policy, final params. |
| Budget / seeds / protocol? | **OK.** All EP-A* files ≤ 1,000,000 env steps (999,424), no auxiliary data, tuning on seeds ≥ 1000 only, finals on 0–9 (TASK-017) or 42–51 / test 424–433 (TASK-020), ≥ 256 eval episodes, 0 censored, params_total ≥ params_deployed. Selection scripts read tuning files only. |
| Provenance (reporting rule: the recorded commit must identify the code)? | **Violated for 16 files** (table below): the arm they name did not exist at their recorded `git_commit`, i.e. they were produced by uncommitted code that cannot be identified now. No file records an evaluator commit, a dirty-tree flag or library versions; `requirements*.txt` are unpinned and the `dpod-local` image has since been deleted, so no earlier run can be re-executed in its original environment. |
| Reproducible? | **No for Truck.** Same config and seed, two sessions: `ppo_gru` 33.11 → 31.64 and 33.26 → 34.46 (≈ ±1.5), but Truck 41.35 → 34.06 and 43.11 → 33.97, `cnn_gru` 28.96 → 36.90 and 33.38 → 32.92. Both Truck and `cnn_gru` first runs come from the unidentified code above, so code drift and GPU non-determinism cannot be told apart from the files. |

### Status of the claims

* **"Truck (tf_gru) beats pure RL: 39.19 ± 1.27 vs 33.57 ± 0.83, Δ +5.6 ± 1.5, passes the surpass test" (TASK-017, README §4, `docs/Latex/ClaftaxResult.tex`) — UNRELIABLE, do not cite.** The ten final runs are real measurements of committed code (9859360, which contains `TFGRUArm`), but (1) the tuning runs that chose its learning rate came from uncommitted code (code_absent), (2) the architecture was designed after looking at the other arms' results on the same evaluation seeds 0–9 (P8 grey zone: not a confirmatory test), and (3) it did not replicate: TASK-020 measured Truck at **34.19 ± 1.62** on fresh seeds 42–51 (Δ vs `ppo_gru` +1.74, not distinguishable), and re-running its own tuning seeds with the same config gave 34.06 / 33.97 instead of 41.35 / 43.11. TASK-020's STATE entry already concluded "the claim is NOT established"; this audit adds the provenance cause.
* **TASK-020 ladder (ppo_gru 32.45 ± 0.70 / 8.16 ± 0.37; cnn_gru 34.64 ± 0.49 / 9.30 ± 0.36; Truck 34.19 ± 1.62 / 9.19 ± 0.74) — the most reliable Phase-1 numbers available.** All 128 runs come from one code state (no `src/` change between bb771a4, ef1109b and aa66385), integrity checks pass, seeds are fresh. Caveats: 2-seed tuning (selection SE ≈ 2.5, winner's curse), no recorded library versions, runs are not bit-reproducible.
* **G1.0 FAIL** (best baseline ≈ 33 vs required 42.7) — robust: no noise level seen here bridges a 10-point gap.
* WM-feature null results, ChunkPPO and pure-Transformer verdicts (TASK-017): integrity OK, code identifiable; unaffected except the exploratory `tf_gru_wm` n = 1 check (code_absent).

### Provenance gaps (code_absent)

| files | recorded commit | missing at that commit | used for |
|---|---|---|---|
| `tune_1000k_derived/tf_gru__*` (3), `tune_1000k_derived/cnn_gru__*` (3) | cb358995 | `tf_gru`, `cnn_gru` arms | lr selection of the TASK-017 Truck / CNN-GRU finals |
| `tune_1000k_derived3/tf_gru_wm*` (2) | 9859360 | `tf_gru_wm`, `tf_gru_wm_random` | exploratory n = 1 WM check |
| `tune_100k/tf_sc__*` (4) | 320f1a4d | `critic` option | diagnostic (aborted) |
| `tune_100k_explore/tf_aux*` (4) | 0aa4a857 | `tf_aux` arm | exploratory 100k table |

### Rule deviations already recorded elsewhere, confirmed

* P9 (TASK-017): unequal tuning — `ppo_gru` got 5 extra single-change variants, Truck's lr 3e-3 point had one seed. Favoured the baseline. TASK-020 fixed it (16 identical trials per arm).
* P8 grey zone (TASK-017): Truck designed after the 1M results of other arms on evaluation seeds 0–9 (README §5 says so).

## Corrections made (TASK-022/023 commit)

1. **Errata** at the top of `docs/experiments/2026-09-30_phase1_model_families/README.md` and in `docs/Latex/ClaftaxResult.tex` (the PDF predates the erratum and was not rebuilt); `MODEL_REGISTRY.yaml` description of Idea4_O01_D01_S00 no longer says "Phase-1 best"; STATE notes on TASK-017/020. No result file was edited, moved or deleted: they are real measurements, now labelled by this audit (`audit_report.json`).
2. **Provenance is now recorded and enforced.** Every new result file carries `evaluator_commit` and `provenance` (git commit, `git_dirty`, sha256 of the imported `src/` code + runner, package versions, backend/device, `XLA_FLAGS`). `--role final` refuses a dirty tree unless `--allow-dirty` (recorded). The sweep driver passes the host's commit/dirty/evaluator-commit into the container and records `pip freeze` of the image (`env` stage). `run_epa_mini.py --save-params` keeps final parameters for later analysis.
3. **Reproducibility is tested before any new claim** (TASK-023 stage S0, scheduled, not run): the same seed twice with and without XLA's deterministic GPU ops (100k steps, bit-for-bit), and a full replication of the TASK-017 Truck and GRU finals (default config, seeds 0–9). Decision rule fixed now: if Truck's replication mean is more than 2 combined SE below 39.19, the TASK-017 number is declared non-reproducible in STATE; if it is within, the TASK-020 config (t04) is declared the cause.
4. **Selection noise reduced** for the new sweep: 3 tuning seeds per selected point (TASK-020 used 2); finals on replicate seeds 52–61 (test 434–443), unseen so far.
5. **Audit tool** `scripts/audit_phase1_results.py` (integrity + provenance + reproducibility pairs; exit 1 on integrity errors, `--strict` also on provenance gaps). Re-run it after every stage.

Not changed: the environment reward. The operator asked (2026-10-04) whether the +0.3 net reward of being woken by a zombie hit (+1 `wake_up` − 0.7 health) should be set to 0. Decision: **no** — EP-A requires the reward unchanged and P5 forbids reward shaping; any such change would make every number non-comparable and gate-ineligible. The TASK-023 death diagnostic measures how often agents die asleep instead.
