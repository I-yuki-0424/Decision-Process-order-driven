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

## Blockers / limits
- Single behaviour seed for data collection and heuristic evaluation (episode-level variance is already captured via n=384,
  but a second full replicate was not run).
- The `effects` table ignores action feasibility (e.g. "drink" with no water in view has zero real effect but is not
  modelled as failing), which weakens the `effects_only`/`effects_wm_*` floor by construction; a stronger known-effects
  model (state-conditioned, e.g. via the local tile map) was not built.
- Only one bootstrap round; the operator's proposal implies iterating, but round 1 already shows the mechanism does not
  help, and the "best" round-1 arm being `noop_always` makes a second meaningful iteration moot without first fixing the
  underlying decision rule.
