# DPOD <-> BOTSYSTEM file API, v1

Status: **proposal** written on DPOD's side (2026-10-30). BOTSYSTEM (`forge1201-MODED-AIPlayer`) has not implemented or
agreed to it; its task `TASK-20261030-013` owns the final interface. `protocol.py` and `action_spec.py` enforce
exactly what is written here, and `tests/test_botsystem_interface.py` is the contract test.

## Roles

BOTSYSTEM is the environment (Forge server, Mineflayer bot, reward, resets). DPOD is the policy and learner. DPOD never
touches the game; BOTSYSTEM never runs the model.

## Transport

Two append-only JSON-lines files per run, each written by exactly one side:

```
<root>/<run_id>/env_to_policy.jsonl    BOTSYSTEM writes:  hello, obs, bye
<root>/<run_id>/policy_to_env.jsonl    DPOD writes:       hello_ack, act
```

- One message per line, UTF-8/ASCII JSON, ends with `\n`, at most 4 MiB, no `NaN`/`Infinity`.
- Write a line with one `write` call on an append-mode file. The reader only consumes up to the last `\n`, so a
  half-written line is harmless and is read on the next poll.
- A line that is not valid is counted and skipped (logged with its first 120 bytes), never fatal.
- Both files are the audit trail. Nothing is deleted or rewritten; `tail -f` and `jq` are the debugger.
- **Local filesystem only** (one machine; a Windows/WSL2 shared mount is untested). A synced
  folder (Nextcloud, SMB) does not guarantee append and read-after-write order. Between machines use HTTP with
  the same message bodies (POST one message, GET messages after `seq`). That transport is not written yet.
- A new environment process starts a **new `run_id`**. `(episode, step)` must not repeat inside one run, because DPOD
  skips pairs it already answered (that is what makes a DPOD restart safe).

## Envelope (every message)

| field | type | meaning |
|---|---|---|
| `v` | int | `1` |
| `type` | string | see below |
| `run_id` | string | must equal the directory name |
| `env_id` | int >= 0 | which of several vectorised environments; `0` for one |
| `seq` | int >= 0 | per-writer counter, 0, 1, 2, ... (gaps are visible in the log) |
| `t` | float | writer's unix time, seconds (informational) |

## Messages

### `hello` (BOTSYSTEM -> DPOD), first line

```json
{"v":1,"type":"hello","run_id":"r1","env_id":0,"seq":0,"t":1793369340.1,"env_version":"2026-10-30",
 "action_spec":{ ...see Action space... },
 "obs_spec":{"fields":[{"name":"pos","shape":[3],"dtype":"float"},{"name":"on_ground","shape":[],"dtype":"bool"}]}}
```

`dtype` is `float`, `int` or `bool`. Every `obs` must contain **exactly** these fields with these shapes. An undeclared
field is rejected, not ignored: that is the guard against answer leakage (DPOD's `CLAUDE.md`, "Critical review"). Declare
only what a player legitimately perceives (health, inventory, nearby blocks, own position). No hidden server state, no
ground-truth goal progress that the bot could not see.

### `hello_ack` (DPOD -> BOTSYSTEM)

`{"accepted": true, "action_spec_hash": "<sha256>"}` or `{"accepted": false, "reason": "..."}`. After a refusal DPOD
answers no `obs` until a new accepted `hello`. `action_spec_hash` is the sha256 of the spec's compact, key-sorted JSON in
which axis values are written as `%.6g` strings; compare it with your own to detect drift.

### `obs` (BOTSYSTEM -> DPOD)

```json
{"v":1,"type":"obs","run_id":"r1","env_id":0,"seq":7,"t":1793369341.0,"episode":0,"step":3,
 "obs":{"pos":[10.5,64.0,-3.2],"on_ground":true},
 "reward":0.0,"reward_terms":{"task":0.0,"shaping":0.0},"terminated":false,"truncated":false}
```

- `episode`, `step`: `step` starts at 0 in each episode. `reward` is the reward for the action taken in the previous
  step (0 on step 0). `reward_terms` is optional; log shaped terms separately from the task term.
- `terminated`: the goal succeeded or failed by the game's rules. `truncated`: a time or step limit ended it.
- After a terminal `obs` the environment resets and sends `step: 0` of the next `episode`. DPOD writes **no** `act`
  for a terminal `obs`.

### `act` (DPOD -> BOTSYSTEM)

```json
{"v":1,"type":"act","run_id":"r1","env_id":0,"seq":4,"t":1793369341.02,"episode":0,"step":3,"obs_seq":7,
 "action":{"forward":"hold","jump":"tap","yaw_delta_deg":15.0,"hotbar":0},"latency_ms":3.1}
```

Answers the `obs` with the same `(episode, step)`; `obs_seq` is that observation's `seq`. Actions are written by
component **name and label**, not by index, so a log line reads on its own. The environment must drop an `act`
whose `(episode, step)` is not the current one and log that it did. DPOD sends at most one `act` per `(episode, step)`.

### `bye` (BOTSYSTEM -> DPOD)

Ends the run; the DPOD server exits after handling the lines before it.

## Action space

An action holds one value per component for `ticks_per_step` game ticks.

| component | values | applied |
|---|---|---|
| key (`forward`, `jump`, `attack`, ...) | `off`, `tap`, `hold` | `off`: up all ticks. `tap`: down on tick 0 only. `hold`: down every tick |
| axis (`yaw_delta_deg`, `pitch_delta_deg`) | one of a listed set of numbers | once, at tick 0 |
| select (`hotbar`) | integer `0 .. n-1` | once, at tick 0 (`0` = leave unchanged) |

- "Pressed once" is `tap`; "held" is `hold`. Holding across several steps is `hold` in each of those steps, and
  releasing is `off`; there is no state carried between steps, so the policy's input fully describes what the key does.
- `tap` is refused when `ticks_per_step` is 1 (it would equal `hold`).
- Rules: `{"kind":"exclusive","a":"forward","b":"back"}` forbids both pressed (mode not `off`); `{"kind":"requires",
  "a":"sprint","b":"forward"}` forbids `a` pressed without `b`. DPOD never sends a rule-violating action.
- The wire form is `ActionSpec.to_wire()`; `from_wire` is strict and rejects unknown component kinds.
- `ActionSpec.key_schedule(action)` is the reference for what the environment must do on each tick (per tick, which keys
  are down). Whether Mineflayer's `setControlState` on the next physics tick reproduces the `tap` timing, and what a held
  `jump` does in Forge 1.20.1, are **unverified** and belong in BOTSYSTEM's task 007 measurement.
- `proposal_full()` (9 keys, yaw, pitch, hotbar: 2,952,450 combinations before rules) and `proposal_min()` (4 keys + yaw,
  90 valid combinations) are DPOD's proposals. `ticks_per_step = 4` is a placeholder until the speed is measured.
  BOTSYSTEM's `hello` is authoritative; DPOD refuses a spec only if started with `--expect-spec`.
