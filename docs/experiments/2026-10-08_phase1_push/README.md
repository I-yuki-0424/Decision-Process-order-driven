# TASK-20261008-025: Phase-1 push (pre-registration, written 2026-10-08 ~19:00 JST, before any result of this task existed)

Operator request (2026-10-08, unreachable until the report): close Phase 1 and clearly, significantly beat the baseline by means other than adding
parameters; local + Kaggle authorized; deadline 2026-10-09 11:30 JST. Machine clock check: Windows time zone is Tokyo Standard Time, the
system UTC clock agrees with HTTP Date headers (google / github), so the machine clock IS JST (the memory note "TST = JST - 1 h" was wrong).

## Critical review (before running)
1. **The gate relation was misapplied before.** Roadmap `surpass` = mean - 2 x SE (10 seeds) > reference. TASK-024's "block 1 alone passed G1.1"
   (48.54 +- 1.50) is not a pass under this definition (48.54 - 3.00 = 45.54 < 47.40). G1.1 needs roughly mean >= 50 at Truck's seed spread;
   G1.2 needs reward - 2SE > 55.49 AND score - 2SE > 16.77, i.e. about +10 reward over the best result so far. A +10 jump from training-method
   changes alone inside one night is unlikely; the plan still screens the most plausible generic levers and reports whatever happens.
2. **Method vs architecture.** Recipe / auxiliary-loss changes are not architecture. Any method change that wins for Truck is also applied to
   the baseline in the finals (P9). The gate (vs. paper reference) can be passed by Truck + method; a "Truck beats the baseline" claim needs the
   baseline WITH the same method on the same seeds.
3. **Transition-prediction auxiliary (TP)** = the 4th idea's S_{t+1} = S_t + W_res[A_t] learned: the candidate-action token of the EXECUTED
   action predicts the observed obs delta and reward through one linear map. No leakage: targets are the executed transition already in the PPO
   batch (one action per state, P4), no extra env steps (P10), no game knowledge (P5); the head is trained (counted in params_total, not deployed).
   The baseline gets the same target through an MLP head on [features, one-hot(action)] (it has no per-action token). Prior evidence for aux
   losses here is weak (flat TF aux at 100k steps: no gain), so TP is a hypothesis, not an expected win.
4. Other screened levers: more PPO updates per env step (K1 64x32, K2 32x64; TASK-024 found larger batches worse), entropy annealing to 0 (EA).
   Training curves of E1/G4 are still rising steeply at 1M steps, so sample-efficiency levers are where a gain could come from.

## Seeds (all unused before; checked against every JSON under output/)
* Tuning (screening): 3000-3999, role `tune`. Platform: local RTX 3060 Ti (Docker dpod-local) and Kaggle T4 (pip craftax 1.6.1; versions
  recorded per file). A config is compared with E1 **on the same seeds and the same platform** (Kaggle E1 calibration runs: stage `Kkag`).
* Final block A: train 82-91 / test 464-473. Final block B (replication): train 92-101 / test 474-483. Role `final`, protocol EP-A, clean tree.

## Selection rule
* Screening: 4 tuning seeds per config. A config is a candidate if mean(reward + score) >= E1 (same seeds, platform) + 3.0.
  Combinations of candidates may be screened next (same rule vs E1). The final Truck config = best mean(reward + score) among candidates
  with >= 4 seeds; if none qualifies, the final Truck config is E1 (the TASK-024 config).
* Baseline for the finals: G4 (gru256 + ln + skip, 1.01M) with every method change of the chosen Truck config that applies to it
  (recipe knobs identically; TP through its MLP head), plus plain G4 when the chosen config differs from E1.
* Finals: block A for the chosen Truck config and its baseline(s); block B if time allows (replication). All seeds listed.
* Gate evaluation: G1.1 / G1.2 on block A exactly as defined (10 seeds, mean - 2 SE > reference on both metrics, params_total <= 4.0M for G1.1);
  block B and the pooled 20 seeds reported alongside; a gate is called "robustly passed" only if block B passes too.
* Truck vs baseline: difference of means with SE (independent seeds, same seed numbers), per block and pooled.
