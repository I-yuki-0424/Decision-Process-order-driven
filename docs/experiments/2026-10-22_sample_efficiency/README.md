# TASK-20261022-026: sample-efficiency ideas for G1.2 (pre-registration, written 2026-10-22 ~01:00 JST, before any result of this task)

Operator request (2026-10-22, unreachable until 11:30 JST, no approvals needed): using mainly Kaggle, try (1) a learning-rate schedule that
passes the steep-loss phase quickly, (2) a reward coefficient learned from part of the 1M steps, (3) plain parameter adjustment, to improve
sample efficiency towards G1.2 (MFRL_best: reward > 55.49 AND score > 16.77, surpass = mean - 2 SE over 10 seeds). Any candidate that comes out
of the results may be tried.

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
* Screening role `tune`, tuning seeds **4000-4003** (disjoint from T025's 3000-3999 and from every final seed), EP-A-tuning settings otherwise
  (1M steps, 256 fresh eval envs, sampled policy, final params). Platform Kaggle T4 for all arms of a comparison; baselines (G4TP1, TP1h) are
  re-run on the same seeds and platform in the same kernel set.
* Candidate rule: mean(reward + score) >= baseline (same seeds, platform) + 3.0 (same as T025). Candidates may be combined in later rounds.
* No result of this task is gate evidence unless produced by a `final` run on fresh seeds from a clean tree, as in TASK-025.
* Env steps: unchanged 999,424 per run; no extra interaction, no oracle, no hand-specified game knowledge (except the labelled health-coefficient diagnostic).

## Amendments
(each appended before the runs it covers)
