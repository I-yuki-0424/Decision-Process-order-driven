# integrations/botsystem

Interface between DPOD and BOTSYSTEM (`forge1201-MODED-AIPlayer`, a Mineflayer bot for modded Minecraft 1.20.1).
DPOD is the policy and learner; BOTSYSTEM is the environment. Wire format: [PROTOCOL.md](PROTOCOL.md).

## Boundary with the research code

- **One direction only.** `integrations/` may import `src/`; `src/` must never import `integrations/`
  (`tests/test_botsystem_interface.py::TestBoundary`). The research code, the Kaggle kernels (`build_kaggle_kernel.py` embeds
  `src/` only) and the MLflow store do not know this directory exists.
- **Nothing from here counts toward a roadmap gate.** Gates G1.x are Craftax-Classic under EP-A. A BOTSYSTEM run is a
  different environment and protocol; its numbers are never quoted next to Craftax numbers (P-rules in
  `docs/core/MASTER_GUIDANCE.xml`). Craftax evidence does not transfer here automatically.
- The interface core is standard library only (checked by the same test), so BOTSYSTEM-side tooling and CI can import it
  without JAX. JAX policies belong in a separate module that imports `src/`, added when a policy is chosen.
- Run exchange data goes to `exchange/` (gitignored). Logs are the two JSONL files; do not duplicate them elsewhere.

## Run

```bash
python -m unittest tests.test_botsystem_interface                 # contract tests, fake environment
python -m integrations.botsystem --root exchange --run-id r1 \
       --policy random --seed 0 --expect-spec min                  # noop | random; stops on `bye` or 300 s without input
```

## Files

| file | role |
|---|---|
| `action_spec.py` | closed action-space definition: keys (`off`/`tap`/`hold`), axes, selects, rules; `nvec`, `encode`/`decode`, `key_schedule`, wire form, hash, the two proposals |
| `protocol.py` | line parsing and validation (size cap, no NaN, declared observation fields only) |
| `file_channel.py` | append-only JSONL writer and partial-line-safe reader |
| `serve.py` | `PolicyServer` (restartable, answers each `(episode, step)` once), `Policy` protocol, `NoopPolicy`, `UniformRandomPolicy` |

## Not done (needs a decision or a measurement first)

- **The policy.** Whether DPOD's Transformer+MDP model is the learner here is the owner's open decision (BOTSYSTEM
  task 013). `UniformRandomPolicy` is the control every learned policy has to beat; no learner is wired in.
- **The final action set.** `proposal_full` and `proposal_min` are proposals. Its ~3 million combinations rule out DPOD's
  flat 17-way softmax and candidate-token layout; a learner needs factored heads (one per component) or the small spec.
- **Training loop and reward.** The loop that turns the `obs`/`act` logs into rollouts and updates, resets, and
  vectorised environments. It must also respect the EP-style budget counting and the leakage rules in `CLAUDE.md`.
- **HTTP transport**, and any use across machines.
- **Environment speed.** Real-time Minecraft is 20 ticks/s; how many steps per second BOTSYSTEM can deliver is its
  task 007 and decides whether RL here is practical at all.
