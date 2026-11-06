# Router + goal-specific skill policies (BOTSYSTEM side track)

Status: **interface and tests only; no expert is trained, nothing is in a training queue** (owner 2026-11-06: "do not add it to the learning
queue yet"). This is a NEW architecture beside `integrations/botsystem/` and the existing learners (`scripts/botsystem_goal_ppo.py`,
`scripts/botsystem_skill_ppo.py`); none of those files was changed. Nothing here counts toward a roadmap gate (G1.x are Craftax-Classic, EP-A).

Source: owner's proposal, knowledge-DB `LLM-records/2026-11-06-dpod-goal-specific-expert-architecture-router-skill-policies-2026-11-06.md`.

## What it is

```
instruction --LLM (once, outside the loop)--> goal {goal_id, amount}
BOT mechanism: command sequences (gather, pit, ...) handle it                      -> done, no model is called
      | cannot (e.g. amount too large for its sequences)
      v   escalate {goal_id, amount, inventory}                       (escalation.py, JSONL like protocol.py)
   Router (TableRouter, rule-based v1)  -- decision {expert | refusal reason, optional set-up goals} -->
      v
   ExpertRegistry: loads ONLY that expert (LRU, `max_resident`), expert.act({goal_id, amount, inventory}) per decision
      env steps of every expert go into ONE StepBudget
```

| module | job |
|---|---|
| `goal_spec.py` | `GoalSpec`: target item, amount range, required items, the STANDARDISED start distribution (`initial`), `spec_hash`; `sample()` draws one start; `encode()` is the goal vector every expert and the router share |
| `router.py` | `TableRouter.route(goal_id, amount, inventory)`; optional, bounded, cycle-safe set-up delegation (off by default: open question) |
| `registry.py` | on-demand loading, refusal of an expert whose training distribution hash differs from the goal's, exact-keys input, one step budget |
| `escalation.py` | the two message types, closed validation of an untrusted request |

Terminology: this is hierarchical RL with options / a skill library with a router, not a mixture of experts in the strict sense (gated experts
trained jointly). "Staged inference" is not a standard term. (From memory, literature not checked here.)

## Critical review (CLAUDE.md "Critical review of proposed architectures")

The proposal can work as a model. It has four places where it could look better than it is; each is closed in the interface or turned into a
pre-registered control below. Recorded in STATE.yaml (TASK-20261106-040).

1. **Leakage through the start conditions.** The standardised preconditions are handed out by the environment (RCON `give`) during training.
   If the expert could see that (the seed, the spec, a "was handed out" flag) it would learn the training harness, not the task.
   Closed: `Instance.expert_input()` is exactly `{goal_id, amount, inventory}`; provenance stays in `Instance.meta`;
   `ExpertRegistry.act` refuses any other key (tests: `test_expert_input_carries_no_provenance`, `test_expert_input_is_exactly_the_declared_keys`).
2. **Evaluation on the training distribution is unfalsifiable.** An expert trained on `iron` starts and scored on the same start distribution
   can only look good. The claim "higher success on hard goals" is about DEPLOYMENT states: what the bot actually holds when its command
   sequences give up. Required evaluation (to be fixed in a pre-registration before any run): (a) held-out draws of the start distribution,
   (b) natural starts taken from real bot runs (the existing `normal` scenarios), (c) the escalation states logged by the BOT mechanism.
   (b) and (c) decide the claim; (a) alone never does.
3. **Confounded attribution.** Gains could come from the randomised preconditions, not from splitting into experts. Controls (same total
   env steps for every arm, seeds listed): M0 monolithic goal-conditioned policy on natural starts (today's baseline); M1 monolithic policy
   trained on the SAME standardised starts; E0 experts on natural starts; E1 experts on standardised starts. Only E1 vs M1 measures the split;
   M1 vs M0 measures the preconditions; E1 vs M0 is the owner's combined claim and is reported as such.
4. **Hidden cost.** "Call only the needed model" saves inference compute, not training steps. The step count is the sum over experts
   (`StepBudget`): an architecture with N experts that quotes one expert's steps is the unequal effort P7/P10 forbid. The router's latency and
   the experts' load time (`ExpertRegistry.loads`) are reported next to success rates.

Further limits kept visible: the router must not be trained on "which expert succeeds here", since the label needs counterfactual
branching of the environment (P3, and an oracle at decision time); v1 is a table, and a learned router has to keep `route`'s signature and
read only the request. Set-up delegation (`delegate_setup`) hard-codes a requirement graph (`requires`/`target`): that is hand-specified game
knowledge (P4), acceptable in this side track, never in a gate run. `EXAMPLE_GOALS` are placeholders to make the form testable
(`provisional=True`); their numbers are not a recommendation.

## Open questions (from the proposal, still open)

1. Training the router (supervised, RL, or the table).  2. The expert-to-expert interface beyond `encode()`.  3. Whether the router may order
set-up (`delegate_setup`).  4. The real start distribution per goal (initial items, amount range): owner.  5. How the DPOD evaluation rules
(leakage and oracle checks, pre-registration, positive control) are applied: a positive control for an expert is a scripted solver on the same
goal whose success the expert must approach; a negative control is a policy given a shuffled goal id, which must score near the no-op rate.

## Not done

No expert policy, no training script, no learner change, no registry entry in `MODEL_REGISTRY.yaml` (a model id is registered before the first
run), no wiring on the BOT side beyond the request builder in forge1201 `src/work/escalation.js`. To start training later: register the model
ids as provisional, write the pre-registration (point 2 and 3 above), and ask the owner to put it on the learning queue.

Run: `python -m unittest tests.test_botsystem_experts`
