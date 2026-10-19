# TASK-20261019-026: Phase-1 status check and G1.2 last-mile tuning (pre-registration, written 2026-10-19 14:45 UTC, before any run of this task)

Operator request (2026-10-19, unreachable for 15 h, no permission needed): (1) write a document explaining the Phase-1 results and comparing the arms,
(2) verify that Phase 1 is done; if so design Phase-2 candidates, if not "we should already be very close, so try tuning the parameters",
(3) record in the knowledge-DB, (4) update MASTER_GUIDANCE if the phase advances and add a note to use the knowledge-DB actively.

## 1. Verification (done before any run; sources: STATE.yaml TASK-025 `result`, RESULTS.md of 2026-10-08_phase1_push, MASTER_GUIDANCE roadmap)
* Roadmap rule `phase_advance`: a phase closes only when EVERY gate in it passes under EP-A. Phase-1 gates: G1.0, G1.1, G1.2 (G1.2 has `closes_phase="true"`), G1.S.
* G1.0 PASS (G4 45.42 / 15.48 >= 42.7 / 9.6). G1.1 passed on a pre-registered block at the second attempt (GRU-TP, block I 48.72 +- 0.36 / 18.82 +- 0.58; its
  first block A failed by 0.02): the operator has not yet ruled on whether that closes G1.1.
* **G1.2 FAIL for every arm in every block.** It needs reward_pct - 2 SE > 55.49 and score_pct - 2 SE > 16.77. Score is already met by the TP arms
  (pooled GRU-TP 18.51 - 2 x 0.21 = 18.10). Reward is not: best block mean 50.72 (TP1h, block F), best 10-seed block of a low-variance arm 48.72 (GRU-TP).
  At GRU-TP's SE (~0.4) the mean must reach ~56.3, i.e. **+7.6 reward over its pooled mean (+16 % relative)**; Truck-TP1h would need ~57.9 (+8.5) at its SE.
* "Phase 1 is done" is therefore **false**. Phase 2 is locked; the Phase-2 request ("if the conditions are met") is not triggered and nothing in Phase 2 is built.
* "Very close" does not hold for reward: the gap is 7.6 reward_pct = 1.7 achievements per episode, 19 SE of the best low-variance arm. It holds for score.

## 2. Critical review: can tuning close a +7.6 gap? (reasoning from the already recorded numbers; nothing run yet)
* Per-achievement ceiling: taking, for every achievement, the BEST unlock rate over all five TASK-025 arms (RESULTS.md section 5; arms differ in seeds, so this is
  an optimistic mix, not an achievable policy) gives (99.8+98.8+75.0+99.0+73.5+91.5+81.4+98.3+59.3+81.1+74.6+0.0+5.5+17.3+16.3+77.3+77.1+30.4+1.6+0+0+0)/22 = **52.6**.
  Even a hypothetical agent that matched the best arm on every achievement simultaneously stays below 55.49. G1.2 needs a larger unlock rate on the deep chain
  (stone tools 9-19 %, coal 14-31 %, iron <= 1.6 %, skeleton <= 5.5 %, eat_plant / diamond / iron tools 0 %), not a better mix of the shallow ones.
* Survival: zombies cause 70-82 % of deaths and evaluation episodes end after ~200-230 steps, so the deep chain is rarely reached at all.
* Past tuning: ~160 jobs of recipe screening for Truck (TASK-024), 16-trial x 2-seed x 1M-step random search (TASK-020), K1/K2/EA/gamma 0.99/width 256/3 layers (TASK-025)
  all moved reward by <= 3 (TP, the only lever that did, gave +3.1 on the GRU). Nothing recorded suggests a knob worth +7.6.
* Prior expectation, stated before running: no single-knob change of GRU-TP reaches +7.6. The run below is a falsification attempt of "we are very close", bounded by the
  compute of this session, not an expected win. If it shows no change, the honest conclusion is that G1.2 needs a new capability (survival / long-horizon credit assignment),
  not tuning.

## 3. Compute (measured)
Cloud session: 4 CPU cores, no GPU, no Kaggle credentials, no Docker. JAX 0.12.0 CPU, flax 0.12.10, optax 0.2.8, craftax 1.6.1, gymnax 0.0.9, Python 3.13.
Measured: GRU-TP (gru256+ln+skip+tp1), 65,536 steps = 110 s training (~600 steps/s) -> ~28 min per 1M-step run on all 4 cores (Truck arms are ~2x slower).
A third platform (CPU) next to the local RTX 3060 Ti and Kaggle T4: results are compared only within this task on the same platform (R0 reference on the same seeds).
Budget: ~11 h of job starts => ~20-24 GRU-class 1M-step runs. Truck arms and 10-seed final blocks of anything but a GRU-class arm do not fit.

## 4. Design (all existing knobs, no new code path in src/, no game knowledge, no extra env steps; EP-A-tuning, 999,424 steps, symbolic 1345-d, role `tune`)
Tuning seeds **4000-4999** (disjoint from every earlier seed: 0-9, 42-51, 52-171, 1000-1002, 2000-2005, 3000-3003, 3900; checked against STATE.yaml / README seed lists).
Evaluation on 256 fresh envs of the tuning seed (the same evaluator; `--eval-seed-offset` 0 for tuning). Final seeds are never touched in this stage.

Round R1 (2 tuning seeds 4000-4001 per config; one factor around R0 = G4TP1, the TASK-025 GRU-TP config; arm string for R5 differs only by width):
| id | change vs R0 | rationale |
|---|---|---|
| R0 | none (G4TP1) | reference on the same seeds and platform |
| R1 | lr 2e-3 | TASK-020 optimum was at the grid edge for some arms |
| R2 | lr 5e-4 | opposite side |
| R3 | entropy 0.02 | more exploration for the deep chain |
| R4 | gamma 0.99, lambda 0.8 | longer credit horizon (gamma 0.99 on Truck gave no gain; never tried on GRU-TP) |
| R5 | width 512 (~3.4M params) | capacity; G1.2 has no parameter cap (params_total only reported) |
| R6 | 32 envs x 128 steps | longer truncated-BPTT window at the same batch |
| R7 | 8 PPO epochs | more updates per env step |

Selection rule (fixed now): a config qualifies for round R2 if its 2-seed mean reward_pct >= R0 mean + 2.0 on the same seeds AND its score is not lower than R0.
R2 = the combination of the (up to) two best qualifying single changes, 3 seeds 4000-4002, plus R0 on the same seeds.
**Stop rule:** a 10-seed EP-A final block (train 172-181 / test 554-563, role `final`, clean tree, fixed config) is run ONLY IF a config reaches a tuning-seed mean
reward_pct >= 54.0 (G1.2 needs ~56.3 on final seeds; 2.3 cushion for the winner's curse seen in TASK-025). Otherwise no final block is run, no gate is claimed,
and the result is "G1.2 not reachable by tuning at this effort". No gate number is ever read from a tuning seed (P8). The Truck family is not screened (cost); this is
stated as a limitation, not a conclusion about Truck.

## 5. Not done, on purpose
* Phase 2 candidates: locked until G1.2 passes (roadmap `scope`); building them would be work on a locked phase.
* No edit of the roadmap numbers or `current_phase`.
* No reward shaping, no hand-coded achievement knowledge (P5), no new algorithm (strictness): a Dyna-style use of the TP head (RESULTS.md section 7) is an unconfirmed
  component and is only proposed in the report, not implemented.

## Amendment 1 (2026-10-19 ~15:30 UTC, after R1 had started, before any R1 result existed; arithmetic correction only, no design change)
* `scripts/adhoc/t026_gap_ceiling.py` recomputes the numbers of section 1/2: mean needed for GRU-TP = **56.2** (not ~56.3), gap to the pooled GRU-TP mean **+7.7** (not +7.6), i.e. ~19 block-SE
  (SE 0.36) or ~47 pooled-SE; Truck-TP1h needs **~58.0** at its pooled sd 3.9 (not ~57.9). The stop-rule threshold 54.0 is unchanged.
* Section 1 said the best 10-seed block of a low-variance arm is 48.72 (block I) -- correct; the best block of any arm is 50.72.
