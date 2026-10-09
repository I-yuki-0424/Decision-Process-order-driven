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

## Amendment 1 (2026-10-09 13:15 JST, after screening round 1, before round 2 and before any final run)
* Session interruption: the agent session stalled at a tool-permission prompt on 2026-10-08 ~19:10 JST; the local GPU was idle from 20:20 (end of
  stage K) to 13:05 on 10-09. The operator then extended the deadline to 2026-10-10 11:30 JST and waived all confirmations.
* Round 1 (tuning seeds 3000-3003; mean reward / score, r+s): local E1 43.50 / 14.80 (58.3); K1 43.96 / 15.42 (59.4); K2 44.07 / 14.39 (58.5);
  Kaggle E1 (2 seeds) 43.17 / 13.78 (57.0); EA 43.19 / 13.74 (56.9); **TP1 48.69 / 18.17 (66.9)**; TP03 46.54 / 17.24 (63.8);
  Kaggle G4 45.85 / 16.79 (62.6); **G4TP1 48.69 / 17.77 (66.5)**; G4TP03 48.10 / 17.58 (65.7).
* Only TP qualifies (>= E1 + 3 r+s on the same platform), and it lifts the baseline too. Consequences, fixed now:
  (1) the gate attempt uses Truck + TP; (2) the Truck-vs-baseline comparison is Truck+TP vs G4+TP at the SAME tp_coef choice rule
  (each arm's best coefficient on these tuning seeds, equal number of coefficients screened per arm: {0.3, 1, 3, 10});
  (3) plain E1 and plain G4 are run on final block A as well, so the TP effect is measured on fresh seeds for both arms;
  (4) attribution control TP1h (Truck, TP through an MLP head instead of the candidate tokens) is screened on the same seeds.
* Round 2: TP3, TP10 (Truck), G4TP3, G4TP10, TP1h on Kaggle; EV03, EV1 (local); TP1+EV0.3 (TP1EV) if EV qualifies.

## Amendment 2 (2026-10-09 19:40 JST, after screening round 2, before any TP final run)
* Round 2 (Kaggle, seeds 3000-3003; reward / score, r+s): TP3 47.57 / 15.90 (63.5); TP10 41.69 / 13.38 (55.1); TP1h 47.15 / 16.94 (64.1);
  G4TP3 46.82 / 16.86 (63.7); G4TP10 42.81 / 14.41 (57.2). EV (local, 1 seed each, stopped): EV03 32.7, EV1 30.7 vs E1 41.6 on seed 3000.
* Selection (rule of amendment 1): Truck final = **TP1** (tp_coef 1.0, candidate-token readout, 0.371M params_total, 0.283M deployed);
  baseline final = **G4TP1** (tp_coef 1.0, MLP head, 1.254M params_total, 0.880M deployed). Both arms screened the same 4 coefficients.
* Final runs: block A (82-91 / test 464-473) LOCAL for E1, G4, TP1, G4TP1 (one platform, clean commit); block B (92-101 / 474-483) on KAGGLE
  T4 for TP1, G4TP1, G4, E1 (one platform). Gate G1.1 / G1.2 evaluated on block A (TP1); block B = replication.

## Amendment 3 (2026-10-09 21:05 JST, after final block B of TP1 / G4TP1 / G4, BEFORE any block-A result of TP1 or G4TP1 exists)
* Block B (Kaggle, 92-101 / 474-483): G4TP1 48.14 +- 0.33 / 17.88 +- 0.55; G4 45.32 +- 0.50 / 15.27 +- 0.43; TP1 (Truck) 43.71 +- 0.97 / 14.63 +- 0.85.
  TP1 did not reproduce its tuning number (48.69, 4 seeds): winner's curse on a high-variance arm. TP on the GRU reproduced (+2.8 reward vs G4).
* Added gate candidate (config fixed in amendment 2, before any final): **G4TP1 = GRU baseline + transition-prediction auxiliary** ("GRU-TP").
  It was registered as the baseline arm; promoting it to a gate candidate is a decision taken after seeing block B, so block B is NOT used
  as its gate evidence. Its gate test is block A (local, pending), with block C as replication.
* Block C (train 102-111 / test 484-493, never used) on Kaggle for G4TP1, G4, TP1, E1 as time allows; report all blocks, never select blocks.
* Truck vs baseline (operator question) is reported per block for both pairs (E1 vs G4, TP1 vs G4TP1); no claim of superiority unless the pooled
  difference is > 2 SE in the same direction in every block.

## Amendment 4 (2026-10-09 22:30 JST, after block B for all four arms; block A TP arms 2/10 seeds done)
* Block B: E1 46.22 +- 0.91 / 16.21 +- 0.77 (vs G4 45.32 / 15.27, TP1 43.71 / 14.63). TP lowers Truck on block B (-2.5 reward) while it raises the GRU (+2.8).
* POST-HOC (labelled as such, never gate evidence for a claim decided before): hypothesis "TP conflicts with the logit readout of the candidate
  tokens" -> test TP1h (Truck, TP through a separate MLP head; config fixed in round 2) on block C (Kaggle) next to E1. TP1 is dropped from block C
  (its question is answered by blocks A + B).
* Block D (train 112-121 / test 494-503, never used) LOCAL for E1 and G4 after block A, to add power to the pre-registered Truck-vs-baseline
  comparison (pooled over every block of both TASK-024 and TASK-025, no block selection).

## Amendment 5 (2026-10-09 23:50 JST, after block C of G4TP1 / G4; block A TP arms 13/20 done)
* Block E (train 122-131 / test 504-513, never used) on Kaggle for E1 and G4: more power for the pre-registered E1-vs-G4 comparison
  (all blocks pooled, none selected). No further screening; no new configs.

## Amendment 6 (2026-10-10 00:45 JST, after block A of TP1 / G4TP1)
* Block A (pre-registered gate block, local): TP1 47.22 +- 1.53 / 16.55 +- 1.14 (G1.1 fail); G4TP1 48.40 +- 0.51 / 18.59 +- 0.71: reward - 2 SE
  = 47.38 < 47.40 -> **G1.1 NOT passed on its gate block** (by 0.02); blocks B and C (Kaggle) pass. Under the rules above G1.1 stays not passed.
* Block F (train 132-141 / test 514-523, never used) LOCAL for G4TP1 and TP1h after block D: one more fresh replication, reported together
  with every other block. It cannot by itself turn the block-A result into a pass; the report gives the count of passing blocks and the
  pooled estimate, all seeds listed.

## Amendment 7 (2026-10-10 01:50 JST, after block C of E1 / TP1h)
* Block C (Kaggle): TP1h 50.23 +- 1.70 / 19.84 +- 1.36 (post-hoc arm, amendment 4); E1 47.28 +- 1.12 / 17.09 +- 0.93.
* Block G (train 142-151 / test 524-533, never used) on Kaggle for TP1h and G4TP1: TP1h stays a post-hoc arm (chosen after block B); its
  evidence is blocks C, F, G together, all reported. No other new configs.

## Amendment 8 (2026-10-10 02:15 JST)
* Block H (train 152-161 / test 534-543, never used) on Kaggle for E1, G4, G4TP1, TP1h: more power for every comparison; reported with all
  other blocks. Last block of this task (runs must end by ~09:00 JST for the 11:30 report).
