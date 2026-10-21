# TASK-20261022-026: sample-efficiency ideas for G1.2 (pre-registration, written 2026-10-22 ~01:00 JST, before any result of this task)

Operator request (2026-10-22, unreachable until 11:30 JST, no approvals needed): using mainly Kaggle, try (1) a learning-rate schedule that
passes the steep-loss phase quickly, (2) a reward coefficient learned from part of the 1M steps, (3) plain parameter adjustment, to improve
sample efficiency towards G1.2 (MFRL_best: reward > 55.49 AND score > 16.77, surpass = mean - 2 SE over 10 seeds). Any candidate that comes out
of the results may be tried.

## Context found after the first launch (amendment 0)
Local main was stale when this task started. TASK-20261019-026 (PR #18, `docs/experiments/2026-10-19_phase1_g12_tuning/`) had already shown on tuning
seeds 4000-4003 that single knobs around GRU-TP do not help (lr 2e-3 48.70, lr 5e-4 44.82, entropy 0.02 46.88, gamma 0.99 45.84, width 512 48.93, 8 epochs
41.18, other batch shapes worse; R0 48.83 / 17.67) and that the same model reaches 57.7 / 66.3 / 65.5 reward at 2M / 3M / 4M steps (diagnostic, not EP-A):
the G1.2 gap is sample efficiency (~1.7-2x). Consequences: (a) the round-1 kernels (already running, seeds 4000-4003, so the seeds are NOT fresh: they
overlap TASK-026's and carry that selection caveat; new tuning seeds from round 2 on are 5000-5999) repeat some TASK-026 knobs (lr2, ent005, lam9, w512);
(b) the new content of round 1 is the lr SHAPE (knee/cosine), the health-reward diagnostic and TP1h-based variants; (c) later rounds look for the
~1.7x: richer generic auxiliaries and learned-model (Dyna-style) use of the TP head, plus combinations of whatever round 1 finds.

## Critical review (before running)
1. **Starting point.** Best arms (TASK-025): GRU-TP (G4TP1, 1.25M) 48.5 / 18.5 pooled, TP1h 49.4 / 18.6. Score already clears G1.2 (16.77); **reward
   needs about +7 (+9 with -2SE margin at TP1h's spread)**. Screening noise: GRU-TP seed sd ~1; TP1h sd ~3-5.
2. **Idea 1 (LR schedule).** The harness already uses linear-to-0 annealing with 5 % warm-up (peak 1e-3). So "gradually reduce the LR" is
   not new; what is new is the SHAPE (higher peak, fast first decay "knee", cosine). Legitimate and generic (a learner hyper-parameter).
3. **Idea 2 (reward coefficient) conflicts with P5 / EP-A ("reward unchanged", "no reward shaping beyond the environment reward").** Any learned or
   fixed re-weighting of the training reward is reward shaping, even if the coefficient is learned from the agent's own (budget-counted) steps.
   A per-achievement weighting (e.g. inverse frequency) additionally needs the achievement identity, which is not in the reward signal: that is
   external achievement knowledge (P5) and is NOT tried. The only variant tried is a coefficient on the |r| < 0.5 terms (the +-0.1 health
   terms, identifiable from the scalar reward alone) as a **labelled diagnostic: it can never count toward a gate** unless the operator amends
   P5. Rationale for running it anyway: it tells whether the reward signal is a bottleneck at all; if it is large, the operator can decide.
   Plain reward scaling (a positive scalar on all rewards) is a no-op under value normalisation and advantage standardisation: not tried.
4. **Idea 3 (parameters).** One-at-a-time neighbours of GRU-TP (width, TP head, entropy, lambda, gamma, minibatches, envs, epochs, clip, grad norm).
   TASK-020/024 searched the plain baseline's recipe; TP changes the optimum possibly.
5. **Honest expectation.** Single-lever tuning of a PPO-GRU rarely moves reward by +7 (TASK-025: K1/K2/EA/gamma/width/layers: no gain). G1.2 is more
   likely to need a structurally different source of sample efficiency; candidates that come out of the screening (combinations of winners) are
   tried; a G1.2 claim needs the full EP-A gate protocol (10 fresh seeds, a clean commit) and is made only if it is reached.

## Protocol
* Screening role `tune`, tuning seeds **4000-4003** in round 1 (see amendment 0: overlap with TASK-026; disjoint from every final seed), 5000-5999 afterwards, EP-A-tuning settings otherwise
  (1M steps, 256 fresh eval envs, sampled policy, final params). Platform Kaggle T4 for all arms of a comparison; baselines (G4TP1, TP1h) are
  re-run on the same seeds and platform in the same kernel set.
* Candidate rule: mean(reward + score) >= baseline (same seeds, platform) + 3.0 (same as T025). Candidates may be combined in later rounds.
* No result of this task is gate evidence unless produced by a `final` run on fresh seeds from a clean tree, as in TASK-025.
* Env steps: unchanged 999,424 per run; no extra interaction, no oracle, no hand-specified game knowledge (except the labelled health-coefficient diagnostic).

## Amendments
(each appended before the runs it covers)

## Amendment 1 (2026-10-22 ~03:00 JST, after round 1 [kernels t026-a/b, seeds 4000-4003, Kaggle T4], before round 2/3/4 results)
Round 1 (reward / score, r+s; baseline GRU-TP G4TP1 47.11 / 17.39 = 64.50, TP1h 45.94 / 16.52 = 62.46, same seeds and platform):
lr 2e-3 48.49 / 17.75 (66.2); cosine 2e-3 47.28 / 16.15 (63.4); knee 2e-3 (x0.3 at 25 %) 47.99 / 18.99 (67.0); Truck lr 2e-3 46.99 / 18.57 (65.6);
Truck knee 45.84 / 16.48 (62.3); width 512 50.12 / 19.55 (69.7; TASK-026 found 48.93 vs 48.83 on the same seeds, so not trusted); entropy 0.005 48.84 / 18.81 (67.7);
lambda 0.9 39.32 / 12.80 (52.1, harmful); **health-term coefficient 0 (rs0, P5-contested diagnostic) 50.89 +- 0.32 / 21.15 +- 0.91 (72.0, sd 0.63 over seeds)**;
coefficient 2: 47.98 / 17.03 (65.0).
Reading: (1) lr SHAPE does not help (all below the +3.0 candidate line; the linear anneal already used is as good). (2) Zeroing the +-0.1 health terms of the
training reward is the only large effect (+3.8 reward, +3.8 score, tight spread) - it is reward shaping (P5 / EP-A "reward unchanged"), so it is NOT gate evidence;
any EP-A claim built on it needs the operator to amend P5. (3) Candidates by the rule: rs0, entropy 0.005 (+3.15, weak), width 512 (+3.2, contradicted by TASK-026).
Next (fixed now): R2 (kernel c, seeds 5000-5003): LP auxiliary (4 variants) vs G4TP1 re-run; R3 (kernel d): rs0 stacked with knee / lr 2e-3 / gamma 0.99 /
lambda 0.9 / entropy 0.02 and rs0 alone on the new seeds; R4 (kernel e): rs 0.25 and 0.5 (sensitivity), rs0 + entropy 0.005 / width 512 / both, Truck + rs0.

## Amendment 2 (2026-10-22 ~04:55 JST, after R2 [kernel c] and R3 [kernel d], seeds 5000-5003, Kaggle T4; before R4 [e], R5 [f] results)
Baseline G4TP1 on 5000-5003: 47.43 +- 1.10 / 17.39 +- 1.05 (r+s 64.8). R2 (LP = SPR-style multi-step latent prediction on top of TP, 1.52M params):
lp 0.3 (k=3) 48.39 / 17.93 (66.3); **lp 1.0 (k=3) 50.03 +- 1.48 / 20.56 +- 1.59 (70.6; seeds 47.9 47.7 50.3 54.1)**; lp 3.0 47.83 / 19.44 (67.3); lp 1.0 k=6 48.63 / 19.65 (68.3).
R3 (health-coefficient 0 = rs0, P5-contested): rs0 alone **replicates on fresh seeds: 50.83 +- 0.94 / 21.03 +- 1.03 (71.9)** (round 1: 50.89 / 21.15); rs0+knee 50.33 / 20.13;
rs0+lr 2e-3 50.42 / 20.42; rs0+gamma 0.99 48.21 / 18.27; rs0+entropy 0.02 45.97 / 17.78; rs0+lambda 0.9 40.30 / 13.68.
Reading: rs0 is a stable +3.4 reward / +3.6 score and nothing tried so far stacks on it (lr shape neutral, gamma/entropy/lambda harmful); LP is a legitimate (P5-clean)
gain of about +2.6 reward / +3.2 score at lp 1.0, noisy at 4 seeds (qualifies by the +3.0 r+s rule). Combinations in R5 (kernel f): lp1 + entropy 0.005 / knee / width 512,
rs0 + lp1, and lp1 re-run with a baseline re-run on fresh seeds 6000-6003.
Final blocks (fixed now, seeds unused anywhere in output/): block J train 172-181 / test 554-563, block K train 182-191 / test 564-573 (offset 382, role final, clean tree).
Gate evaluation, if run, follows TASK-025 exactly (10 seeds, mean - 2 SE vs MFRL_best 55.49 / 16.77 for G1.2; params_total reported). Any arm using rs0 is labelled
"P5-contested diagnostic" in its result file protocol field and cannot close a gate unless the operator amends P5.

## Amendment 3 (2026-10-22 ~06:20 JST, after R4 [kernel e], seeds 5000-5003; before R5 [f] and R6 [g] results)
R4 (reward / score, r+s; baseline G4TP1 47.43 / 17.39 = 64.8): health coefficient 0.5: 48.50 / 18.82 (67.3); 0.25: 49.44 / 19.66 (69.1); 0 (R3): 50.83 / 21.03 (71.9) - monotone in the coefficient,
so any "learned" coefficient would have to end at 0 (= ignore the +-0.1 health terms). rs0 + entropy 0.005: 49.72 / 19.07 (68.8); rs0 + width 512 (3.64M): 51.40 / 21.61 (73.0);
**rs0 + entropy 0.005 + width 512: 52.93 +- 1.18 / 21.72 +- 1.59 (74.65; seeds 49.6 55.2 53.6 53.3)**; Truck + rs0 (T_rs0): 48.17 +- 2.10 / 18.97 (67.1; no gain, wide spread).
R6 (kernel g, queued after the LP result of R2): rs0+lp1+w512+ent005, rs0+lp1+w512, rs0+lp1+ent005, and the P5-clean stack lp1+w512+ent005, same seeds.
