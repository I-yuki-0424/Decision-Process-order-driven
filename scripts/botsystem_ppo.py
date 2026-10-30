#!/usr/bin/env python3
"""Plain PPO baseline of DPOD's `transformer_branch` candidate on the BOTSYSTEM (Minecraft) environment.

Side track, NOT research code: nothing here counts toward a roadmap gate (G1.x), no MLflow, no EP-A/EP-B claims. It lives in
scripts/ (not integrations/) because integrations/ must stay standard-library only (tests/test_botsystem_interface.py).

What is DPOD's and unmodified: the actor (`src.model.candidates.transformer_branch`, called exactly as in
src/pipeline/candidate_experiment.py::_make_policy with a 16-row running history buffer) and the learner recipe of
src/pipeline/candidate_actor_critic.py (PPO clipped ratio, GAE gamma 0.99 / lambda 0.95, tanh-MLP critic, entropy 0.01,
value 0.5, clip 0.2, 4 epochs x 4 minibatches, global-norm clip 0.5, AdamW 3e-4 linearly decayed, advantage normalisation).
That recipe is re-implemented here because the original is fused with the Craftax adapters inside jit; formulas and defaults are
the same. Not tuned: d_model 64 instead of their default 256 is the only size change (CPU budget), rollout/updates are fixed.

What is new (the BOTSYSTEM-side mapping, all in this file): the candidate list is the flat enumeration of the environment's
action spec; ActionsData.features = fixed per-component one-hots, costs constant, resource_effects ZERO (no hand-coded
transition knowledge); SystemState.resource_levels = the declared observation fields, normalised, with the 9x4x9 block grid
one-hot; TransitionTarget.target_state = the goal vector. No field outside the declared observation is used.

Modes: train | eval (policy: learned, learned_greedy, random, scripted) | synthetic (learner positive control, no Minecraft).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from integrations.botsystem.action_spec import ActionSpec, proposal_min_pitch  # noqa: E402
from integrations.botsystem.serve import PolicyServer  # noqa: E402
from src.model.candidates import transformer_branch as tfb  # noqa: E402
from src.model.types import ActionHistory, ActionsData, InputContextN, SystemState, TransitionTarget  # noqa: E402

GAMMA, LAM = 0.99, 0.95
HIST = 16                   # running-history rows of the candidate (candidate_experiment.HIST_WINDOW)
NUM_COSTS, FEAT_DIM, GOAL_DIM, N_CAT = 4, 32, 22, 11
GRID_N = 324
SMALL_DIM = 4 + 3 + 1 + 1 + 1 + 1 + 16 + N_CAT + 1 + 16 + 1       # everything but the grid and the goal
STATE_DIM = SMALL_DIM + GRID_N * N_CAT
CRITIC_DIM = SMALL_DIM + GOAL_DIM + N_CAT                          # small fields + goal + grid category fractions


# --------------------------------------------------------------------------------------------------------------- observation
def encode_obs(o: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Declared observation -> (small float vector, grid ints, goal vector). Scalings are fixed constants."""
    small = np.concatenate([
        o["look"], np.asarray(o["vel"]) * 4.0, [float(o["on_ground"]), o["health"] / 20.0, o["food"] / 20.0, o["held_slot"] / 9.0],
        np.minimum(np.asarray(o["inventory"], dtype=np.float32), 8) / 8.0,
        np.eye(N_CAT)[int(o["cursor_block"])], [np.clip(o["cursor_dist"], -1.0, 6.0) / 5.0],
        np.asarray(o["entities"]) / np.tile([4.0, 8.0, 8.0, 8.0], 4), [o["step_frac"]],
    ]).astype(np.float32)
    assert small.shape == (SMALL_DIM,), small.shape
    return small, np.asarray(o["grid"], dtype=np.int8), np.asarray(o["goal"], dtype=np.float32)


def candidate_table(spec: ActionSpec):
    """Flat enumeration of the valid actions and their fixed feature rows (per-component one-hots)."""
    acts = list(spec.iter_valid())
    off = np.concatenate([[0], np.cumsum(spec.nvec)])
    feats = np.zeros((len(acts), FEAT_DIM), dtype=np.float32)
    assert off[-1] <= FEAT_DIM
    for i, a in enumerate(acts):
        for c, k in enumerate(spec.encode(a)):
            feats[i, off[c] + k] = 1.0
    return acts, feats


def make_input(feats, small, grid, goal, hist_act, hist_rew):
    """Traceable: build InputContextN for one step (batched by vmap over the leading axis of the per-step arguments)."""
    n = feats.shape[0]
    res = jnp.concatenate([small, jax.nn.one_hot(grid.astype(jnp.int32), N_CAT).reshape(-1)])
    actions = ActionsData(features=feats, costs=jnp.full((n, NUM_COSTS), 0.5), preconditions=jnp.zeros((n, 2), jnp.int32),
                          valid_mask=jnp.ones((n,), jnp.bool_), abstraction_scales=jnp.ones((n,)),
                          resource_effects=jnp.zeros((n, STATE_DIM)))
    state = SystemState(resource_levels=res, available_costs=jnp.array([100.0, 50.0, 30.0, 20.0]),
                        status_flags=jnp.zeros(4), progress_rate=jnp.zeros(()))
    history = ActionHistory(action_indices=hist_act / n, rewards=hist_rew, cost_changes=jnp.zeros((HIST, NUM_COSTS)),
                            noise_mask=jnp.zeros((HIST,), jnp.bool_), seq_len=jnp.array(HIST, jnp.int32))
    target = TransitionTarget(target_state=goal, conditions=jnp.zeros(4), deadline_step=jnp.array(150.0))
    return InputContextN(actions=actions, state=state, history=history, target=target)


def critic_vec(small, grid, goal):
    frac = jax.nn.one_hot(grid.astype(jnp.int32), N_CAT).mean(0)
    return jnp.concatenate([small, goal, frac])


def init_critic(key, in_dim, width=128):                           # candidate_actor_critic.init_critic
    k1, k2, k3 = jax.random.split(key, 3)
    s = lambda k, i, o, g: jax.random.normal(k, (i, o)) * (g / np.sqrt(i))
    return dict(w1=s(k1, in_dim, width, np.sqrt(2.0)), b1=jnp.zeros(width), w2=s(k2, width, width, np.sqrt(2.0)),
                b2=jnp.zeros(width), w3=s(k3, width, 1, 1.0), b3=jnp.zeros(1))


def critic_apply(cp, x):                                           # candidate_actor_critic.critic_apply
    h = jnp.tanh(x @ cp["w1"] + cp["b1"])
    h = jnp.tanh(h @ cp["w2"] + cp["b2"])
    return (h @ cp["w3"] + cp["b3"])[0]


# ------------------------------------------------------------------------------------------------------------------- learner
class Learner:
    def __init__(self, spec: ActionSpec, d_model: int, seed: int, lr: float, planned_opt_steps: int, ent_coef=0.01, vf_coef=0.5,
                 clip_eps=0.2, epochs=4, minibatches=4):
        self.spec, self.d_model = spec, d_model
        self.acts, feats = candidate_table(spec)
        self.n = len(self.acts)
        self.feats = jnp.asarray(feats)
        self.ent_coef, self.vf_coef, self.clip_eps, self.epochs, self.minibatches = ent_coef, vf_coef, clip_eps, epochs, minibatches
        k_actor, k_crit, self.key = jax.random.split(jax.random.PRNGKey(seed), 3)
        actor = tfb.init_transformer_branch_parameters(k_actor, d_model=d_model, num_actions=self.n, action_feat_dim=FEAT_DIM,
                                                       num_costs=NUM_COSTS, num_resources=STATE_DIM, target_dim=GOAL_DIM)
        self.params = dict(actor=actor, critic=init_critic(k_crit, CRITIC_DIM))
        self.n_actor = int(sum(x.size for x in jax.tree_util.tree_leaves(actor)))
        self.opt = optax.apply_if_finite(optax.chain(optax.clip_by_global_norm(0.5), optax.adamw(
            learning_rate=optax.linear_schedule(lr, 0.0, planned_opt_steps), eps=1e-5)), max_consecutive_errors=1000)
        self.opt_state = self.opt.init(self.params)
        self.updates_done = 0
        feats_c = self.feats

        def policy(actor_p, small, grid, goal, h_act, h_rew, hist):
            d, new_hist = tfb.forward_transformer_branch(actor_p, make_input(feats_c, small, grid, goal, h_act, h_rew), hist, num_heads=8)
            return d.action_logits, new_hist[-HIST:]

        @jax.jit
        def act_fn(params, small, grid, goal, h_act, h_rew, hist, key, greedy):
            logits, new_hist = policy(params["actor"], small, grid, goal, h_act, h_rew, hist)
            a = jnp.where(greedy, jnp.argmax(logits), jax.random.categorical(key, logits))
            logp = jax.nn.log_softmax(logits)
            return a, logp[a], critic_apply(params["critic"], critic_vec(small, grid, goal)), new_hist
        self.act_fn = act_fn

        def chunk_loss(params, ch):
            def one(small, grid, goal, h_act, h_rew, hist, act, old_logp, adv, ret):
                logits, _ = policy(params["actor"], small, grid, goal, h_act, h_rew, hist)
                logp_all = jax.nn.log_softmax(logits)
                ent = -jnp.sum(jnp.exp(logp_all) * logp_all)
                v = critic_apply(params["critic"], critic_vec(small, grid, goal))
                ratio = jnp.exp(logp_all[act] - old_logp)
                pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
                return jnp.stack([pg, 0.5 * (v - ret) ** 2, ent, old_logp - logp_all[act], (jnp.abs(ratio - 1.0) > clip_eps) * 1.0])
            return jax.vmap(one)(*ch).sum(0)

        def loss_fn(params, mb, chunk):
            b = mb["act"].shape[0]
            keys = ("small", "grid", "goal", "h_act", "h_rew", "hist", "act", "logp", "adv", "ret")
            chunks = tuple(mb[k].reshape((b // chunk, chunk) + mb[k].shape[1:]) for k in keys)
            parts = jax.lax.map(jax.checkpoint(lambda ch: chunk_loss(params, ch)), chunks).sum(0) / b
            return parts[0] + vf_coef * parts[1] - ent_coef * parts[2], parts

        @jax.jit
        def train_step(params, opt_state, mb):
            (loss, parts), g = jax.value_and_grad(loss_fn, has_aux=True)(params, mb, 64)
            upd, opt_state = self.opt.update(g, opt_state, params)
            return optax.apply_updates(params, upd), opt_state, parts
        self.train_step = train_step

    def act(self, small, grid, goal, h_act, h_rew, hist, greedy=False):
        self.key, k = jax.random.split(self.key)
        a, logp, v, new_hist = self.act_fn(self.params, small, grid, goal, h_act, h_rew, hist, k, greedy)
        return int(a), float(logp), float(v), new_hist

    def update(self, tr: dict, rng: np.random.Generator) -> dict:
        """tr: dict of arrays (B, ...) incl. adv, ret (already per-episode). B must be divisible by `minibatches`*64."""
        b = tr["act"].shape[0]
        mbs = b // self.minibatches
        adv = tr["adv"]
        tr = dict(tr, adv=((adv - adv.mean()) / (adv.std() + 1e-8)).astype(np.float32))
        parts = None
        for _ in range(self.epochs):
            perm = rng.permutation(b)
            for m in range(self.minibatches):
                idx = perm[m * mbs:(m + 1) * mbs]
                self.params, self.opt_state, parts = self.train_step(self.params, self.opt_state, {k: jnp.asarray(v[idx]) for k, v in tr.items()})
        self.updates_done += 1
        p = np.asarray(parts)
        return dict(pg=float(p[0]), vl=float(p[1]), entropy=float(p[2]), approx_kl=float(p[3]), clipfrac=float(p[4]))

    def save(self, path: Path, extra: dict) -> None:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(dict(params=jax.tree_util.tree_map(np.asarray, self.params), opt_state=jax.tree_util.tree_map(np.asarray, self.opt_state),
                             updates_done=self.updates_done, key=np.asarray(self.key), **extra), f)
        os.replace(tmp, path)

    def load(self, path: Path) -> dict:
        d = pickle.load(open(path, "rb"))
        self.params = jax.tree_util.tree_map(jnp.asarray, d["params"])
        self.opt_state = jax.tree_util.tree_map(jnp.asarray, d["opt_state"])
        self.updates_done, self.key = d["updates_done"], jnp.asarray(d["key"])
        return d


def gae_episode(rew: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """candidate_actor_critic.gae for one episode: the last step is terminal (time-limit truncation also gets next value 0)."""
    t_n, adv, last = len(rew), np.zeros(len(rew), np.float32), 0.0
    for t in reversed(range(t_n)):
        nv = val[t + 1] if t + 1 < t_n else 0.0
        last = rew[t] + GAMMA * nv - val[t] + (GAMMA * LAM * last if t + 1 < t_n else 0.0)
        adv[t] = last
    return adv, adv + val


# -------------------------------------------------------------------------------------------------------- policies for PolicyServer
class EnvPolicy:
    """DPOD `Policy` for one environment. kind: learned | learned_greedy | random | scripted. Records transitions when `sink` is set."""

    def __init__(self, learner: Learner | None, kind: str, env_name: str, sink=None, seed: int = 0):
        self.L, self.kind, self.name, self.sink = learner, kind, env_name, sink
        self.rng = np.random.default_rng(seed)
        self.pending = None
        self.episode = -1
        self.ep_return = 0.0
        self.ep_steps = 0
        self.log = None

    def bind(self, spec: ActionSpec) -> None:
        if self.L is not None and spec != self.L.spec:
            raise RuntimeError(f"environment spec {spec.name} differs from the learner's {self.L.spec.name}")
        self.spec = spec
        self.acts = self.L.acts if self.L is not None else list(spec.iter_valid())

    def _reset_episode(self, msg):
        self.episode, self.pending, self.ep_return, self.ep_steps = msg["episode"], None, 0.0, 0
        self.hist = jnp.zeros((HIST, self.L.d_model)) if self.L else None
        self.h_act, self.h_rew = np.zeros(HIST, np.float32), np.zeros(HIST, np.float32)
        self.traj = []

    def _close_pending(self, reward: float, done: bool):
        if self.pending is not None:
            self.traj.append(dict(self.pending, reward=np.float32(reward)))
            self.pending = None

    def act(self, msg):
        if msg["episode"] != self.episode:
            self._reset_episode(msg)
        else:
            self._close_pending(msg["reward"], False)
            self.ep_return += msg["reward"]
        self.ep_steps = msg["step"]
        if self.kind == "random":
            return self.acts[int(self.rng.integers(len(self.acts)))]
        if self.kind == "scripted":
            return scripted_action(self.spec, msg)
        small, grid, goal = encode_obs(msg["obs"])
        a, logp, v, new_hist = self.L.act(small, grid, goal, self.h_act.copy(), self.h_rew.copy(), self.hist, greedy=(self.kind == "learned_greedy"))
        self.pending = dict(small=small, grid=grid, goal=goal, h_act=self.h_act.copy(), h_rew=self.h_rew.copy(), hist=np.asarray(self.hist),
                            act=np.int32(a), logp=np.float32(logp), value=np.float32(v))
        self.hist = new_hist
        self.h_act = np.concatenate([self.h_act[1:], [a]]).astype(np.float32)
        self.h_rew = np.concatenate([self.h_rew[1:], [msg["reward"]]]).astype(np.float32)
        return self.acts[a]

    def end_episode(self, msg) -> None:
        if msg["episode"] != self.episode:
            return
        self._close_pending(msg["reward"], True)
        self.ep_return += msg["reward"]
        success = bool(msg["terminated"] and msg["reward_terms"].get("task", 0) > 0)
        outcome = dict(env=self.name, episode=msg["episode"], steps=msg["step"], ret=self.ep_return, success=success, truncated=bool(msg["truncated"]))
        if self.sink is not None:
            self.sink(self, outcome)
        self.episode = -1


def _nearest(values, want):
    return min(values, key=lambda v: abs(v - want))


def scripted_action(spec: ActionSpec, msg) -> dict:
    """Positive control for the collect scenario: reads only declared observation fields (log cells in the grid, crosshair,
    dropped-item entities). Shows the task is solvable in this action space; it is not a learner."""
    o, a = msg["obs"], spec.noop()
    yaw = math.atan2(o["look"][0], o["look"][1])
    pitch = math.atan2(o["look"][2], o["look"][3])
    p = o["pos"]
    fx, fy, fz = math.floor(p[0]), math.floor(p[1]), math.floor(p[2])
    logs, i = [], 0
    for dy in range(-1, 3):
        for dz in range(-4, 5):
            for dx in range(-4, 5):
                if o["grid"][i] == 3:
                    logs.append([fx + dx + 0.5, fy + dy + 0.5, fz + dz + 0.5])
                i += 1
    logs.sort(key=lambda c: math.dist(c, p))
    items = [[p[0] + o["entities"][4 * k + 1], p[1] + o["entities"][4 * k + 2], p[2] + o["entities"][4 * k + 3]] for k in range(4) if o["entities"][4 * k] == 1]
    tgt = logs[0] if logs else (items[0] if items else None)
    if tgt is None:
        a["forward"] = "hold"
        return a
    yerr = (math.degrees(math.atan2(-(tgt[0] - p[0]), -(tgt[2] - p[2]))) - math.degrees(yaw) + 180) % 360 - 180
    perr = math.degrees(math.atan2(tgt[1] - (p[1] + 1.62), max(math.hypot(tgt[0] - p[0], tgt[2] - p[2]), 0.3))) - math.degrees(pitch) if logs else 0.0
    yaws = [c for c in spec.components if c.name == "yaw_delta_deg"][0].values
    pitches = [c for c in spec.components if c.name == "pitch_delta_deg"][0].values
    if abs(yerr) >= 15:
        a["yaw_delta_deg"] = _nearest(yaws, -yerr)
    elif logs and abs(perr) > 8:
        a["pitch_delta_deg"] = _nearest(pitches, -perr)
    elif logs and o["cursor_block"] == 3 and 0 <= o["cursor_dist"] <= 4.4:
        a["attack"] = "hold"
    else:
        a["forward"] = "hold"
    return a


# ------------------------------------------------------------------------------------------------------------------ run loops
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    d = 1 + z * z / n
    c = ph + z * z / (2 * n)
    m = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


class Run:
    def __init__(self, a, learner, kind):
        self.a, self.L, self.kind = a, learner, kind
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.done_eps: list[dict] = []
        self.buffer: list[dict] = []          # finished episodes' transitions with adv/ret, arrival order
        self.buffer_steps = 0
        self.policies = [EnvPolicy(learner, kind, f"{a.run}-e{i}", sink=self.on_episode, seed=a.seed * 100 + i) for i in range(a.n_envs)]
        self.ep_log = open(self.out / "episodes.jsonl", "a")
        self.t0 = time.time()
        self.total_steps = 0
        self.dropped = 0

    def on_episode(self, pol: EnvPolicy, outcome: dict):
        outcome.update(update=self.L.updates_done if self.L else 0, t=round(time.time() - self.t0, 1))
        self.done_eps.append(outcome)
        self.ep_log.write(json.dumps(outcome) + "\n")
        self.ep_log.flush()
        self.total_steps += outcome["steps"]
        if self.kind == "learned" and pol.traj:
            rew = np.array([t["reward"] for t in pol.traj], np.float32)
            val = np.array([t["value"] for t in pol.traj], np.float32)
            adv, ret = gae_episode(rew, val)
            for t, ad, rt in zip(pol.traj, adv, ret):
                t["adv"], t["ret"] = np.float32(ad), np.float32(rt)
            self.buffer.extend(pol.traj)
            self.buffer_steps += len(pol.traj)
        pol.traj = []

    def servers(self):
        root = Path(self.a.root)
        for i, pol in enumerate(self.policies):
            rid = f"{self.a.run}-e{i}"
            (root / rid).mkdir(parents=True, exist_ok=True)
            yield PolicyServer(root, rid, pol, expected_spec=self.L.spec if self.L else proposal_min_pitch(self.a.ticks))

    def take_batch(self, b: int) -> dict:
        items, self.buffer = self.buffer[:b], self.buffer[b:]
        self.buffer_steps = len(self.buffer)
        keys = ("small", "grid", "goal", "h_act", "h_rew", "hist", "act", "logp", "adv", "ret")
        return {k: np.stack([t[k] for t in items]) for k in keys}


def drive(a, learner, kind, stop_fn, after_poll=None):
    """Poll every environment's PolicyServer until stop_fn(run) or the time cap."""
    run = Run(a, learner, kind)
    servers = list(run.servers())
    last_act = time.time()
    try:
        while not stop_fn(run):
            n = sum(s.poll_once() for s in servers)
            if after_poll:
                after_poll(run, servers)
            if n:
                last_act = time.time()
            else:
                if time.time() - last_act > a.idle_timeout:
                    print("idle timeout: no observation from any environment", flush=True)
                    break
                time.sleep(0.01)
    finally:
        for s in servers:
            s.close()
    return run


def summarize(eps: list[dict]) -> dict:
    n = len(eps)
    k = sum(e["success"] for e in eps)
    lo, hi = wilson(k, n)
    return dict(episodes=n, successes=k, success_rate=k / n if n else 0.0, wilson95=[round(lo, 4), round(hi, 4)],
                mean_return=float(np.mean([e["ret"] for e in eps])) if n else 0.0, mean_steps=float(np.mean([e["steps"] for e in eps])) if n else 0.0)


def mode_train(a):
    spec = proposal_min_pitch(a.ticks)
    L = Learner(spec, a.d_model, a.seed, a.lr, a.updates * 4 * 4)
    ck = Path(a.out) / "checkpoint.pkl"
    meta = {}
    if a.resume and ck.exists():
        meta = L.load(ck)
        print(f"resumed at update {L.updates_done}", flush=True)
    print(f"actor_params={L.n_actor} candidates={L.n} state_dim={STATE_DIM} backend={jax.default_backend()} d_model={a.d_model} rollout={a.rollout}", flush=True)
    rng = np.random.default_rng(a.seed + 7)
    log = open(Path(a.out) / "train_log.jsonl", "a")
    deadline = time.time() + a.max_minutes * 60
    state = dict(last_eps=0)

    def stop(run):
        return L.updates_done >= a.updates or time.time() > deadline

    def after(run, servers):
        if run.buffer_steps < a.rollout:
            return
        t_upd = time.time()
        batch = run.take_batch(a.rollout)
        dropped = 0
        for p in run.policies:           # strict on-policy: partial episodes recorded before the update are not mixed with the next policy
            dropped += len(p.traj)
            p.traj = []
        run.dropped += dropped
        stats = L.update(batch, rng)
        eps = run.done_eps[state["last_eps"]:]
        state["last_eps"] = len(run.done_eps)
        s = summarize(eps)
        rec = dict(update=L.updates_done, steps_total=run.total_steps, episodes_total=len(run.done_eps), batch_episodes=s["episodes"], success_rate=s["success_rate"],
                   successes=s["successes"], mean_return=s["mean_return"], mean_steps=s["mean_steps"], dropped_steps=dropped,
                   update_s=round(time.time() - t_upd, 1), wall_s=round(time.time() - run.t0, 1), **stats)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
        L.save(ck, dict(steps_total=run.total_steps))
        if L.updates_done in (1, 5) or L.updates_done % 10 == 0:
            L.save(Path(a.out) / f"checkpoint_u{L.updates_done:03d}.pkl", dict(steps_total=run.total_steps))

    run = drive(a, L, "learned", stop, after)
    L.save(Path(a.out) / "final.pkl", dict(steps_total=run.total_steps))
    final = dict(updates=L.updates_done, steps_total=run.total_steps, train_summary=summarize(run.done_eps), wall_s=round(time.time() - run.t0, 1), actor_params=L.n_actor)
    json.dump(final, open(Path(a.out) / "train_summary.json", "w"), indent=1)
    print("DONE", json.dumps(final), flush=True)


def mode_eval(a):
    spec = proposal_min_pitch(a.ticks)
    L = None
    if a.policy.startswith("learned"):
        L = Learner(spec, a.d_model, a.seed, a.lr, 1)
        L.load(Path(a.ckpt))
    run = drive(a, L, a.policy, lambda r: len(r.done_eps) >= a.episodes)
    res = dict(policy=a.policy, ckpt=a.ckpt, **summarize(run.done_eps), wall_s=round(time.time() - run.t0, 1))
    json.dump(res, open(Path(a.out) / f"eval_{a.policy}.json", "w"), indent=1)
    print("EVAL", json.dumps(res), flush=True)


# ------------------------------------------------------------------------------------- synthetic positive control (no Minecraft)
def mode_synthetic(a):
    """Learner positive control: a 3-step contextual task whose answer is in the goal vector (goal[2] > 0 -> yaw +30 is correct,
    else -30; yaw 0 is wrong). Chance level is 1/3 per step. If PPO + transformer_branch cannot learn this, a null result on
    Minecraft would say nothing about the environment. Uses the same EnvPolicy / Run / Learner code as the real run."""
    spec = proposal_min_pitch(4)
    a.n_envs = 1
    L = Learner(spec, a.d_model, a.seed, a.lr, a.updates * 16)
    run = Run(a, L, "learned")
    pol = run.policies[0]
    pol.bind(spec)
    rng, urng = np.random.default_rng(a.seed), np.random.default_rng(a.seed + 1)
    base = dict(look=[0, 1, 0, 1], vel=[0, 0, 0], on_ground=True, health=20, food=20, held_slot=0, inventory=[0] * 16, grid=[0] * GRID_N,
                cursor_block=0, cursor_dist=-1.0, entities=[0] * 16, step_frac=0.0)
    log = open(Path(a.out) / "synthetic_log.jsonl", "w")
    ep = 0

    def play(greedy_policy=None):
        nonlocal ep
        sign = 1.0 if rng.random() < 0.5 else -1.0
        goal = [0.0] * GOAL_DIM
        goal[2] = 0.5 * sign
        reward, correct = 0.0, 0
        for step in range(3):
            msg = dict(episode=ep, step=step, reward=reward, reward_terms={"task": reward}, terminated=False, truncated=False, obs=dict(base, goal=goal))
            act = pol.act(msg)
            ok = act["yaw_delta_deg"] == 30.0 * sign
            reward, correct = (1.0 if ok else 0.0), correct + int(ok)
        pol.end_episode(dict(episode=ep, step=3, reward=reward, reward_terms={"task": reward}, terminated=True, truncated=False, obs=dict(base, goal=goal)))
        ep += 1
        return correct

    for u in range(a.updates):
        correct = total = 0
        while run.buffer_steps < a.rollout:
            correct += play()
            total += 3
        stats = L.update(run.take_batch(a.rollout), urng)
        rec = dict(update=u + 1, step_accuracy=round(correct / total, 4), chance=0.3333, **stats)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
    pol.kind = "learned_greedy"
    greedy = sum(play() for _ in range(100)) / 300
    last = [json.loads(x)["step_accuracy"] for x in open(Path(a.out) / "synthetic_log.jsonl")][-3:]
    verdict = dict(final_sampled_accuracy_last3=float(np.mean(last)), greedy_accuracy=greedy, chance=1 / 3, passed=bool(np.mean(last) > 0.8))
    json.dump(verdict, open(Path(a.out) / "synthetic_result.json", "w"), indent=1)
    print("SYNTHETIC", json.dumps(verdict), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("train", "eval", "synthetic"), required=True)
    p.add_argument("--root", default="exchange")
    p.add_argument("--run", default="r")
    p.add_argument("--out", required=True)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--ticks", type=int, default=4)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--rollout", type=int, default=2048)
    p.add_argument("--updates", type=int, default=60)
    p.add_argument("--max-minutes", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--policy", choices=("learned", "learned_greedy", "random", "scripted"), default="learned")
    p.add_argument("--ckpt", default="")
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--idle-timeout", type=float, default=300.0)
    a = p.parse_args()
    {"train": mode_train, "eval": mode_eval, "synthetic": mode_synthetic}[a.mode](a)


if __name__ == "__main__":
    main()
