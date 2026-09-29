# Noop-anticipation heuristic planner (TASK-20260927-009)

Tests the operator's follow-up proposal (train a "do nothing" predictor decoupled from the policy; use its mismatch with a
target to pick actions; retrain it on episodes where the player lives longer; use its own predictions autoregressively)
without an RL training loop, since TASK-20260927-008 showed this repo's REINFORCE learner has no measurable power at all
(even a literal reward leak did not move it). Data/code: `output/experiments/2026-09-27_noop_anticipation_heuristic/full/result.json`,
`scripts/run_noop_heuristic_planner.py`. Compute: local RTX 3060 Ti, one run, ~13 min wall-clock.

## Why a heuristic, and the decisive structural point
A noop-only prediction is added identically to every action's predicted outcome, so `argmax` over actions is invariant to
it — on its own it cannot rank actions. It can only matter combined with something that IS action-dependent: here, the
existing hand-specified `CRAFTAX_RESOURCE_EFFECTS` table (`src/environment/craftax_env_adapter.py`; a static, approximate,
non-oracle table of each action's known effect on health/food/drink/energy — health has zero entries everywhere, since
health only reacts to starvation/damage, not to a deliberate action). The test: does adding the world model's noop
anticipation to that table — i.e. "which vital is about to become the bottleneck" — change the action ranking enough to
matter for survival.

## Result (384 episodes/arm, T=250 cap, single seed for behaviour, held-out split by episode for the WM)
| arm | mean survival length | median | crafter score |
|---|---|---|---|
| random | 139.1 +- 64.7 | 142 | 2.29 |
| **noop_always** | **152.9 +- 65.6** | 158 | 0.00 |
| effects_only (known table, no WM) | 149.8 +- 67.4 | 149 | 1.03 |
| effects_wm_h1 (table + WM anticipation, 1 step) | 149.6 +- 69.2 | 147 | 1.43 |
| effects_wm_h8 (table + WM anticipation, 8 steps) | 146.3 +- 66.4 | 138 | 1.38 |
| effects_wm_h1, WM trained "episodic" (below) | 149.8 +- 67.6 | 149 | 1.13 |
| round-2 effects_wm_h1, WM bootstrapped (below) | 150.2 +- 65.8 | 139 | 0.96 |

- **effects_wm_h1 does not beat effects_only** (149.6 vs 149.8; s.e.m. ~3.5 at n=384): the WM's anticipation of the noop
  future adds nothing measurable once the known action-effect table is already used.
- **effects_wm_h8 is worse than effects_wm_h1** (146.3 vs 149.6): the longer, less accurate anticipation horizon hurts, the
  compounding-error risk flagged before running this.
- **None of the "smart" arms beat doing nothing** (noop_always = 152.9, the best of all arms): every non-noop action costs
  energy per the known-effects table (-0.1/step) with no reliable food/water source for an untooled random-walk agent, so
  a rule that only compares current-or-anticipated vitals cannot help — it does not address exploration/way-finding, which
  is the actual bottleneck to living longer here. This matches the pre-registered caveat that the effects table is a weak,
  approximate floor.

## Training-procedure ablation (operator: update W once per whole episode, not per step)
Held-out MSE (raw vitals^2, lower is better), horizons 1/2/4/8, same data both ways:
| training | h=1 | h=2 | h=4 | h=8 |
|---|---|---|---|---|
| pooled (shuffled minibatches across episodes) | 0.066 | 0.116 | 0.174 | 0.244 |
| episodic (one gradient step per whole episode) | 0.068 | 0.121 | 0.194 | 0.284 |

Episodic is uniformly worse (higher-variance, correlated batches). The operator's proposed schedule is not an improvement
here; it is a plausible-but-untrue intuition, not a flaw that need be adopted "as specified" without evidence per CLAUDE.md.

## Bootstrap ("retrain the WM on episodes where the player lives longer")
The best-surviving arm by construction was `noop_always`, so the bootstrap retrained the WM on do-nothing episodes only
(a low-diversity distribution). Held-out MSE on its own distribution improved (0.043 at h1, down from 0.066), but MSE on
the ORIGINAL random-policy distribution got an order of magnitude worse (4.48 at h8, up from 0.244) — the model overfit to
a narrow state distribution, a textbook curriculum/coverage failure, not a gain. Feeding this bootstrapped model back into
the effects_wm_h1 heuristic gave no improvement (150.2 vs 149.6, within noise).

## Verdict
This is a stronger, cleaner negative result than TASK-008's, because it removes the confound of a powerless RL learner: a
deterministic decision rule that CAN use the world model's anticipation, combined with a real (if approximate) action-effect
model, still shows no benefit from that anticipation, at either horizon, and the proposed training/bootstrap procedure makes
prediction accuracy worse or overfit rather than better. The blocking issue for Craftax survival is not world-model quality;
it is that the known cheap decision rules available here (constant table, single-step anticipation) do not capture what
actually keeps the player alive (finding food/water/shelter), which requires exploration this mechanism does not provide.

## Addendum (TASK-20260927-010): operator clarification + learner-power budget check
The operator clarified the proposal after this report: Phase 1 trains the WM on synchronized real no-op branches ("World B"),
not to rank actions; Phase 2 trains the policy separately on the acting branch ("World A") and gives it the WM's counterfactual
baseline as a *feature*, not by adding the same no-op value to every action (the mechanism refuted above was a strawman of the
intended design); a coupled curriculum lets a better policy produce broader WM training data; Phase 3 is a batch action-scoring
scheme where the WM baseline is meant to reduce brittleness on out-of-distribution states, not raise average return.

Phase 2's "WM as policy feature" is exactly what TASK-20260927-008 already tested (its `oracle`/`wm_full`/`wm_untrain` arms).
That test is not invalidated by the strawman correction; it is blocked by the same confound already flagged there: a REINFORCE
learner with no demonstrated power. Two things were checked in response:

**Architecture is not a wiring dead end.** Read `src/model/candidates/transformer_branch.py` and
`shared_stages.self_attention_now_goal_actions`: state, target and action tokens fully bidirectionally cross-attend in stage 1,
and `action_logits` are per-action dot products against that state-aware representation. A signal placed in the state channel
(as TASK-008's oracle arms did) can reach the action scores in one hop -- this is not why the earlier null results occurred.

**Budget diagnostic.** Re-ran the strongest possible positive control (`oracle_act`: true one-step reward of every action) and
`base` at 4x the batch size (64 vs 16) and ~2.7x the episodes (12,800 vs 4,800), 4 seeds each (2 local RTX 3060 Ti + 2 Kaggle):

| arm | mean last-3-eval return (4 seeds) | per-seed |
|---|---|---|
| base | 3.25 +- 0.14 | 3.17, 3.31, 3.08, 3.44 |
| oracle_act | 3.32 +- 0.06 | 3.24, 3.36, 3.30, 3.39 |

A small directional gap appears with markedly lower variance for `oracle_act`, but it is not statistically separated (base's
best seed, 3.44, exceeds oracle_act's best, 3.39). This softens TASK-008's "flat, no power at all" to "no detectable power at
the original budget; an inconclusive, small signal for the strongest possible leak at ~3x the compute." Since a passive
no-op-future feature is weaker/less decision-relevant than a full per-action reward leak, re-testing `wm_full`/`oracle` at this
budget would very likely still show nothing above noise -- the confound is softened, not resolved.

**Recommendation before any further Craftax WM test:** either scale compute further to confirm the `oracle_act` trend cleanly
(cheap: more seeds/updates), or move to an actor-critic-style learner (a structural fix; more engineering). Phase 3's
OOD-robustness claim and the coupled curriculum are both real, distinct, and still untested -- but testing them on top of an
unconfirmed learner would reproduce the same ambiguity, so that is the next blocking step, not the world model itself.

**Tooling note:** the Kaggle output fetch initially failed with a Windows `PermissionError` against `C:/Program Files/Git/...`.
Cause: the `--out /kaggle/working/exp` argument was silently mangled by Git Bash's automatic POSIX-path conversion on the
*local* notebook-build call (that specific call was missing `MSYS_NO_PATHCONV=1`) -- the remote run itself succeeded and wrote
to a garbled-but-harmless nested path on Kaggle's Linux filesystem; the data was recovered by listing kernel output files
directly via the `kagglesdk` client and stripping the bogus prefix, rather than through `kaggle kernels output`.

## Addendum (2026-09-29): corrected crafter_score after the auto-reset-leak fix
STATE.yaml TASK-20260928-015 item 3 found that Craftax `step()` auto-resets on death and the achievement-accumulation
code in this script (before it was patched in place by fix commit `5f5db96`) credited a post-death life's achievements
to the life being scored, inflating `crafter_score`. The table above is the ORIGINAL (pre-fix) run, kept as evidence.
A real re-run with identical arguments (`--envs 384 --eval-envs 384 --steps 250 --seed 0 --train-steps 4000
--episodic-epochs 3`) against the now-fixed script is at `output/experiments/2026-09-27_noop_anticipation_heuristic/full_v2/`:

| arm | crafter_score (old) | crafter_score (new) |
|---|---|---|
| random | 2.29 | 1.86 |
| noop_always | 0.00 | 0.00 |
| effects_only | 1.03 | 0.88 |
| effects_wm_h1 | 1.43 | 1.06 |
| effects_wm_h8 | 1.38 | 1.12 |
| effects_wm_h1 (episodic model) | 1.13 | 0.96 |
| round2 effects_wm_h1 (bootstrapped) | 0.96 | 0.83 |

`mean_len`/`median_len`/`std_len` (the survival-length numbers the headline verdict above is actually based on) are
unaffected by this fix and reproduce within ordinary run-to-run GPU noise. The headline verdict ("effects_wm_h1 does
not beat effects_only", judged on survival length) is unchanged; only the secondary crafter_score column was stale.

## Blockers / limits
- Single behaviour seed for data collection and heuristic evaluation (episode-level variance is already captured via n=384,
  but a second full replicate was not run).
- The `effects` table ignores action feasibility (e.g. "drink" with no water in view has zero real effect but is not
  modelled as failing), which weakens the `effects_only`/`effects_wm_*` floor by construction; a stronger known-effects
  model (state-conditioned, e.g. via the local tile map) was not built.
- Only one bootstrap round; the operator's proposal implies iterating, but round 1 already shows the mechanism does not
  help, and the "best" round-1 arm being `noop_always` makes a second meaningful iteration moot without first fixing the
  underlying decision rule.
