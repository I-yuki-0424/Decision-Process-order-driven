#!/usr/bin/env python3
"""On-demand server + asynchronous goal-conditioned PPO with a world model for the BOTSYSTEM skill-level environment
(forge1201-MODED-AIPlayer env >= v2.0; model scripts/botsystem_goal_wm.py). Side track (DPOD TASK-20261101-031): nothing here
counts toward a roadmap gate.

  python scripts/botsystem_goal_ppo.py --root <exchange> --base train [--learn 1] [--seed 42] [--init pretrained.pkl]

Serving, segments, semi-MDP GAE, staleness correction and run following are those of botsystem_skill_ppo.py (Server is
subclassed). What changes:

Goals (owner 2026-11-01: subdivide the objective; free values). A goal is 1-2 tokens over the goal space the player can read
from its own observation: 'item' (count held), 'kill' (kills this episode per entity type), 'adv' (an advancement flag),
'night' (nights survived). Each token asks for +delta (default 1) relative to the value at assignment; every other dimension
is free. A bot always has a goal; when it is reached the learner pays --goal-bonus and assigns the next one; after
--goal-horizon commands it is dropped (counted as a failure). Goals are proposed from the archive of dimensions some bot has
actually raised (nothing is listed in advance: no tech tree), weighted by the reachability head R(s, g) towards intermediate
difficulty, p(1-p): goals that are too early for this state (R ~ 0) and trivial ones (R ~ 1) are rarely proposed. With
probability --p-task the goal is one of the owner's task goals (a night survived, a diamond held, the Ender Dragon killed).

Goal bonus (owner 2026-11-02, after watching run 4: trivial results must be worth almost nothing, "even thousands of them
less than a single log"): bonus = --goal-bonus x (1 - p) x 0.5^n, p = max(R(s, g) at assignment, the dimension's past success
rate (wins + 1) / (tries + 2)) (the joint p of a two-token goal is the product), n = goals of the same dimension already reached
in this episode. A goal any bot reaches nearly always ("+1 dirt") pays nearly nothing, and repeating it decays geometrically
(a flat bonus of 1.0 had paid five logs for five "+1 dirt" goals; a place-then-mine loop could collect it).

Reward = the environment's terms under the --ext-stage weights of reward-v2 (default 'dragon': every term 1.0, i.e. the
Crafter-style task reward) + goal bonus. Episode return under the --ext-stage weights alone is logged as the task fitness.

Batch generation of a plan (owner 2026-11-02; DPOD 4th/5th idea): with --plan 1, at the start of an episode (and whenever a
plan is used up, fails or is judged infeasible) the plan decoder generates by beam search the whole sequence of subgoals toward
a target T (an owner task goal with probability --p-task-plan, else an archive dimension). The bot then pursues the plan's
steps in order: goal tokens = [current step, T]. Before a step becomes the goal, the reach head judges it from the present
state; R < --plan-tau means "physically too early" and the plan is regenerated (at most 3 times per episode, then an archive
goal). With probability --p-explore a goal comes from the archive instead (keeps exploring, feeds the archive). Hindsight plan
samples: at anchor states of an episode (its start and every --plan-anchor steps), the dimensions first raised afterwards, in
order, up to a later one taken as T (subsampled to the plan length, T last, END after it).

World model and hindsight (TASK-20261101-031): every completed transition gives world-model labels from the next
observation (success, duration, change of own state, inventory and kill counts); once its next --her-horizon steps are known
(or the episode ended) it also gets hindsight reach labels (dimensions raised in that window = positives; archive dimensions
that were not = negatives) and enters a replay ring. Each PPO minibatch adds --aux-coef x the supervised loss on a replay
minibatch, so the world model and R(s, g) keep learning off-policy without biasing the policy gradient (the policy itself is
trained on its own goals only; relabelling a policy-gradient sample would need importance weights).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from integrations.botsystem.action_spec import ActionSpec  # noqa: E402
from integrations.botsystem.protocol import ProtocolError, check_obs, parse_obs_spec  # noqa: E402
from botsystem_goal_wm import GK, GOAL_F, INT_KEYS, OBS_KEYS, GoalWM, plan_beam, sizes_from_hello  # noqa: E402
from botsystem_skill_ppo import SPEC_NAME, ReturnScale, Server, gae, masked_entropy, stage_reward  # noqa: E402

log = logging.getLogger("goal_ppo")
CKPT_VERSION = 1
GOAL_KEYS = ("goal_kind", "goal_id", "goal_feat", "goal_mask")
PLAN_LAB = ("task", "seq", "mask")
NH = 4                     # hindsight goals per transition (2 positives, 2 negatives at most)
TASK_GOALS = (("night", 0), ("item", "diamond"), ("adv", "minecraft:end/kill_dragon"))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a = p.add_argument
    a("--root", required=True), a("--base", required=True), a("--learn", type=int, default=1), a("--seed", type=int, default=42)
    a("--init", default=""), a("--population", type=int, default=1)
    a("--n-max", type=int, default=16), a("--seg", type=int, default=32), a("--batch-segs", type=int, default=16)
    a("--epochs", type=int, default=3), a("--minibatches", type=int, default=4), a("--lr", type=float, default=3e-4)
    a("--gamma-s", type=float, default=0.999), a("--lam", type=float, default=0.95), a("--clip", type=float, default=0.2)
    a("--ent", type=float, default=0.01), a("--vf", type=float, default=0.5), a("--max-grad-norm", type=float, default=0.5)
    a("--poll-s", type=float, default=0.01), a("--kind-prior", type=int, default=1)
    a("--goals", type=int, default=2), a("--goal-bonus", type=float, default=0.5), a("--goal-horizon", type=int, default=150)
    a("--p-task", type=float, default=0.2), a("--p-pair", type=float, default=0.15), a("--ext-stage", default="dragon")
    a("--her-horizon", type=int, default=32), a("--replay", type=int, default=12000), a("--aux-batch", type=int, default=64)
    a("--aux-coef", type=float, default=1.0), a("--aux-lr", type=float, default=1e-3), a("--aux-max-grad-norm", type=float, default=5.0)
    a("--plan", type=int, default=1), a("--p-explore", type=float, default=0.3), a("--p-task-plan", type=float, default=0.5)
    a("--plan-beam", type=int, default=4), a("--plan-tau", type=float, default=0.02), a("--plan-coef", type=float, default=1.0)
    a("--plan-replay", type=int, default=4000), a("--plan-batch", type=int, default=32), a("--plan-anchor", type=int, default=64)
    return p.parse_args(argv)


# ---------------------------------------------------------------- goal space (pure numpy, tested) ---------------------------
class GoalSpace:
    """Dimensions ('item', item id) / ('kill', entity id) / ('adv', progress index) / ('night', 0) and their values in a raw
    observation, the archive of dimensions seen raised, per-dimension assignment statistics, and goal encoding."""

    def __init__(self, sizes: dict, milestones: list[str], items: list[str] | None = None):
        self.s, self.n_mil, self.m = sizes, len(milestones), milestones
        self.items = items
        self.archive: dict[tuple, dict] = {}       # dim -> {"seen": raised count, "tries": assigned, "wins": reached}

    def values(self, o: dict) -> dict:
        """Sparse values of every non-zero dimension of one observation."""
        out: dict[tuple, float] = {}
        for i, c in zip(o["inv_item"], o["inv_count"]):
            if i > 0:
                out[("item", int(i))] = out.get(("item", int(i)), 0.0) + round(float(c) * 64)
        if "killed" in o:
            for j in np.flatnonzero(np.asarray(o["killed"]) > 0):
                out[("kill", int(j))] = round(float(o["killed"][j]) * 8)
        prog = np.asarray(o["progress"])
        for j in np.flatnonzero(prog[:self.n_mil] > 0.5):
            out[("adv", int(j))] = 1.0
        out[("night", 0)] = round(float(prog[-2]) * 10)
        return out

    def note_raised(self, before: dict, after: dict):
        for k, v in after.items():
            if v >= before.get(k, 0.0) + 1:
                self.archive.setdefault(k, {"seen": 0, "tries": 0, "wins": 0})["seen"] += 1

    def task_dims(self, item_ix: dict) -> list[tuple]:
        out = []
        for kind, x in TASK_GOALS:
            if kind == "night":
                out.append(("night", 0))
            elif kind == "item" and x in item_ix:
                out.append(("item", item_ix[x]))
            elif kind == "adv" and x in self.m:
                out.append(("adv", self.m.index(x)))
        return out

    def encode(self, goal: dict | None, vals: dict, step: int, horizon: int, G: int) -> dict:
        kind, gid, feat, mask = np.zeros(G, np.int32), np.zeros(G, np.int32), np.zeros((G, GOAL_F), np.float32), np.zeros(G, np.bool_)
        if goal:
            for j, (dim, base, delta) in enumerate((goal["tokens"] + goal.get("ctx", []))[:G]):
                kind[j], gid[j], mask[j] = GK[dim[0]], dim[1], True
                feat[j] = [np.log1p(delta), np.clip((vals.get(dim, 0.0) - base) / delta, 0, 1), min((step - goal["t0"]) / horizon, 1.0)]
        return {"goal_kind": kind, "goal_id": gid, "goal_feat": feat, "goal_mask": mask}

    @staticmethod
    def reached(goal: dict, vals: dict) -> bool:
        return all(vals.get(dim, 0.0) >= base + delta for dim, base, delta in goal["tokens"])


def dim_index(d: tuple, s: dict) -> int:
    """Goal-space dimension -> plan vocabulary index (0 = END): items, kills, progress/advancements, night."""
    k, i = d
    if k == "item":
        return 1 + i
    if k == "kill":
        return 1 + s["item"] + i
    if k == "adv":
        return 1 + s["item"] + s["entity"] + i
    return 1 + s["item"] + s["entity"] + s["progress"]


def index_dim(j: int, s: dict) -> tuple | None:
    V, E, P = s["item"], s["entity"], s["progress"]
    if j <= 0:
        return None
    if j <= V:
        return ("item", j - 1)
    if j <= V + E:
        return ("kill", j - 1 - V)
    if j <= V + E + P:
        return ("adv", j - 1 - V - E)
    return ("night", 0)


def plan_samples(firsts: list[tuple[int, tuple]], anchors: list[tuple[int, dict]], s: dict, L: int, rng) -> list[tuple[dict, dict]]:
    """Hindsight plans of one episode: firsts = [(step, dim)] in the order dimensions were first raised, anchors = [(step, obs
    arrays)]. For each anchor: T = a dimension first raised later (uniform), plan = the dimensions raised between (evenly
    subsampled to L - 1 in order) + T, then END (mask covers the plan and END)."""
    out = []
    for t_a, x in anchors:
        after = [d for t, d in firsts if t > t_a]
        if not after:
            continue
        j = int(rng.integers(len(after)))
        T, mid = after[j], after[:j]
        if len(mid) > L - 2:
            keep = np.linspace(0, len(mid) - 1, L - 2).round().astype(int)
            mid = [mid[i] for i in sorted(set(keep))]
        seq = [dim_index(d, s) for d in mid + [T]] + [0]
        q = np.zeros(L, np.int32)
        q[:len(seq)] = seq
        m = np.zeros(L, np.bool_)
        m[:len(seq)] = True
        out.append((x, {"task": np.int32(dim_index(T, s)), "seq": q, "mask": m}))
    return out


def goal_value(b0: float, p: float, n_prev: int) -> float:
    """Bonus of a reached goal: b0 x (1 - p) x 0.5^n_prev (p = predicted/observed chance of reaching it, n_prev = goals of the
    same dimension already reached this episode)."""
    return float(b0 * (1.0 - min(max(p, 0.0), 1.0)) * 0.5 ** n_prev)


def hindsight(dims_now: dict, future: list[dict], archive: list[tuple], rng) -> dict:
    """Reach labels for one step: dims_now = values at t, future = values at t+1..t+H. Positives = dims raised by >= 1 at some
    point of the window (up to 2), negatives = archive dims (up to NH - #pos), each with its true label."""
    best: dict[tuple, float] = {}
    for f in future:
        for k, v in f.items():
            if v > best.get(k, -1.0):
                best[k] = v
    raised = [k for k, v in best.items() if v >= dims_now.get(k, 0.0) + 1]
    pos = [raised[i] for i in rng.permutation(len(raised))[:2]]
    pool = [d for d in archive if d not in pos]
    neg = [pool[i] for i in rng.permutation(len(pool))[:NH - len(pos)]] if pool else []
    kind, gid, lab, mask = np.zeros(NH, np.int32), np.zeros(NH, np.int32), np.zeros(NH, np.float32), np.zeros(NH, np.float32)
    for j, d in enumerate(pos + neg):
        kind[j], gid[j], mask[j] = GK[d[0]], d[1], 1.0
        lab[j] = float(best.get(d, -1.0) >= dims_now.get(d, 0.0) + 1)
    return {"her_kind": kind, "her_id": gid, "her_delta": np.ones(NH, np.float32), "her_label": lab, "her_mask": mask}


def to_arrays(obs: dict, sizes: dict) -> dict:
    """Raw observation -> model arrays; fields a v1.3 environment did not send are zeros, the view masked off."""
    out = {}
    vr = sizes["view_shape"][0] * sizes["view_shape"][1]
    fill = {"clock": (sizes["clock"],), "killed": (sizes["entity"],), "view_cat": (vr,), "view_item": (vr,), "view_dist": (vr,)}
    for k in OBS_KEYS:
        dt = np.int32 if k in INT_KEYS else (np.bool_ if k == "cand_mask" else np.float32)
        out[k] = np.asarray(obs[k], dt) if k in obs else np.zeros(fill[k], dt)
    out["view_valid"] = np.bool_("view_cat" in obs)
    return out


def wm_labels(x_next: dict, raw_next: dict, terminated: bool, act: int, has_killed: bool) -> dict:
    return {"act": np.int32(act), "ok": np.float32(raw_next["last_reason"] == 0), "dt": np.float32(raw_next["dt_ticks"]),
            "n_inv_item": x_next["inv_item"], "n_inv_count": x_next["inv_count"], "n_self": x_next["self"], "n_killed": x_next["killed"],
            "wm_valid": np.float32(not terminated), "kill_valid": np.float32(has_killed)}


class Replay:
    """Fixed-capacity ring of labelled transitions (observation arrays + world-model + hindsight labels)."""

    def __init__(self, cap: int):
        self.cap, self.n, self.i, self.buf = cap, 0, 0, None
        self.lock = threading.Lock()

    def add(self, x: dict, lab: dict):
        row = {**{"o/" + k: v for k, v in x.items()}, **{"a/" + k: v for k, v in lab.items()}}
        with self.lock:
            if self.buf is None:
                self.buf = {k: np.zeros((self.cap,) + np.shape(v), np.asarray(v).dtype) for k, v in row.items()}
            for k, v in row.items():
                self.buf[k][self.i] = v
            self.i, self.n = (self.i + 1) % self.cap, min(self.n + 1, self.cap)

    def sample(self, n: int, rng):
        with self.lock:
            idx = rng.integers(0, self.n, n)
            obs = {k[2:]: v[idx] for k, v in self.buf.items() if k.startswith("o/")}
            lab = {k[2:]: v[idx] for k, v in self.buf.items() if k.startswith("a/")}
        return obs, lab


def dummy_plan(x: dict, n: int, L: int):
    obs = {k: np.stack([v] * n) for k, v in x.items() if k not in GOAL_KEYS}
    return obs, {"task": np.zeros(n, np.int32), "seq": np.zeros((n, L), np.int32), "mask": np.zeros((n, L), np.bool_)}


def dummy_aux(x: dict, n: int):
    """A zero-weight auxiliary batch shaped like one observation (before anything is labelled)."""
    obs = {k: np.stack([v] * n) for k, v in x.items() if k not in GOAL_KEYS}
    lab = {**wm_labels(x, {"last_reason": 0, "dt_ticks": 0.0}, True, 0, False),
           **{k: np.zeros(NH, np.int32 if k in ("her_kind", "her_id") else np.float32) for k in ("her_kind", "her_id", "her_delta", "her_label", "her_mask")}}
    return obs, {k: np.stack([np.asarray(v)] * n) for k, v in lab.items()}


# ---------------------------------------------------------------- learner ---------------------------------------------------
class GoalLearner:
    def __init__(self, a, model: GoalWM, params, opt_state=None):
        self.a, self.model = a, model
        self.opt = optax.chain(optax.clip_by_global_norm(a.max_grad_norm), optax.adam(a.lr))
        # The supervised world-model/reach loss has its own optimiser state and clip: its gradient (a sum over every item) under
        # the policy's 0.5 global-norm clip shrank every step to nothing (the world model never left the marginal).
        self.aux_opt = optax.chain(optax.clip_by_global_norm(a.aux_max_grad_norm), optax.adam(a.aux_lr))
        if opt_state is None:
            opt_state = (self.opt.init(params), self.aux_opt.init(params))
        self.params, self.opt_state = params, opt_state
        self.updates = 0
        self._update = jax.jit(self._update_fn)

    def _aux(self, params, aux_obs, aux_lab, p_obs, p_lab):
        aux, terms, _ = self.model.aux_loss(params, aux_obs, aux_lab)
        pl, acc = self.model.plan_loss(params, p_obs, p_lab["task"], p_lab["seq"], p_lab["mask"])
        on = jnp.any(p_lab["mask"]).astype(jnp.float32)
        return self.a.aux_coef * aux + self.a.plan_coef * on * pl, {**terms, "plan": pl, "plan_acc": acc}

    def _loss(self, params, b, ent_coef):
        def body(h, o):
            logits, value, h2, _ = self.model.step(params, h, o)
            return h2, (logits, value)
        obs_t = {k: jnp.swapaxes(v, 0, 1) for k, v in b["obs"].items()}
        _, (logits, value) = jax.lax.scan(body, b["h0"], obs_t)
        logits, value = jnp.swapaxes(logits, 0, 1), jnp.swapaxes(value, 0, 1)
        lp = jax.nn.log_softmax(logits, -1)
        nlp = jnp.take_along_axis(lp, b["act"][..., None], -1)[..., 0]
        ratio = jnp.exp(nlp - b["logp"])
        adv = b["adv"]
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - self.a.clip, 1 + self.a.clip) * adv)
        vl = 0.5 * (value - b["ret"]) ** 2
        entropy = masked_entropy(logits, b["obs"]["cand_mask"])
        m = b["valid"]
        den = jnp.maximum(m.sum(), 1.0)
        mean = lambda x: (x * m).sum() / den
        loss = mean(pg) + self.a.vf * mean(vl) - ent_coef * mean(entropy)
        return loss, {"pg": mean(pg), "vf": mean(vl), "ent": mean(entropy), "kl": mean(b["logp"] - nlp),
                      "clipfrac": mean((jnp.abs(ratio - 1) > self.a.clip).astype(jnp.float32)),
                      "u_ok": params["pi"]["u"][0], "u_dt": params["pi"]["u"][1], "u_gain": params["pi"]["u"][2]}

    def _update_fn(self, params, opt_state, b, aux_obs, aux_lab, p_obs, p_lab, ent_coef):
        o_pi, o_aux = opt_state
        (loss, mets), g = jax.value_and_grad(self._loss, has_aux=True)(params, b, ent_coef)
        up, o_pi = self.opt.update(g, o_pi, params)
        params = optax.apply_updates(params, up)
        (aux, terms), g2 = jax.value_and_grad(self._aux, has_aux=True)(params, aux_obs, aux_lab, p_obs, p_lab)
        up2, o_aux = self.aux_opt.update(g2, o_aux, params)
        return optax.apply_updates(params, up2), (o_pi, o_aux), {**mets, "loss": loss, "aux": aux, **{"aux_" + k: v for k, v in terms.items()}}

    def update(self, segs, scale: ReturnScale, replay: Replay, rng, plan_replay: Replay | None = None):
        a, T = self.a, self.a.seg
        sd = scale.std(0)
        advs, rets = [], []
        for s in segs:
            ad, rt = gae(np.asarray(s["rew"], np.float32) / sd, s["val"], s["boot"], s["dt"], s["done"], a.gamma_s, a.lam)
            advs.append(ad), rets.append(rt)
        scale.update(0, np.concatenate(rets) * sd)
        flat = np.concatenate(advs)
        mu, sg = flat.mean(), flat.std() + 1e-8
        B = len(segs)
        pad = lambda x: np.concatenate([x, np.zeros((T - len(x),) + np.shape(x)[1:], np.asarray(x).dtype)]) if len(x) < T else np.asarray(x)
        keys = list(segs[0]["obs"][0].keys())
        batch = {"obs": {k: np.stack([pad(np.stack([o[k] for o in s["obs"]])) for s in segs]) for k in keys},
                 "h0": np.stack([s["h0"] for s in segs]), "act": np.stack([pad(np.asarray(s["act"], np.int32)) for s in segs]),
                 "logp": np.stack([pad(np.asarray(s["logp"], np.float32)) for s in segs]),
                 "adv": np.stack([pad((ad - mu) / sg) for ad in advs]), "ret": np.stack([pad(rt) for rt in rets]),
                 "valid": np.stack([pad(np.ones(len(s["act"]), np.float32)) for s in segs])}
        batch["obs"]["cand_mask"][..., 0] |= batch["valid"] == 0
        mets, params, opt_state = [], self.params, self.opt_state
        mb = max(1, B // a.minibatches)
        for _ in range(a.epochs):
            perm = rng.permutation(B)
            for i in range(0, B, mb):
                sub = jax.tree_util.tree_map(lambda x: x[perm[i:i + mb]], batch)
                ao, al = replay.sample(a.aux_batch, rng) if replay.n >= a.aux_batch else dummy_aux(segs[0]["obs"][0], a.aux_batch)
                po, pla = (plan_replay.sample(a.plan_batch, rng) if plan_replay is not None and plan_replay.n >= a.plan_batch
                           else dummy_plan(segs[0]["obs"][0], a.plan_batch, self.model.L))
                params, opt_state, m = self._update(params, opt_state, sub, ao, al, po, pla, jnp.float32(a.ent))
                mets.append({k: float(v) for k, v in m.items()})
        self.params, self.opt_state = params, opt_state
        self.updates += 1
        out = {k: float(np.mean([m[k] for m in mets])) for k in mets[0]}
        out.update(segments=B, steps=int(batch["valid"].sum()), ret_std=sd, adv_raw_mean=float(mu),
                   reward_mean=float(np.mean(np.concatenate([s["rew"] for s in segs]))), replay=replay.n,
                   plan_replay=plan_replay.n if plan_replay is not None else 0)
        return out


# ---------------------------------------------------------------- server ----------------------------------------------------
class GoalServer(Server):
    def __init__(self, a):
        super().__init__(a)
        self.replay = Replay(a.replay)
        self.plan_replay = Replay(a.plan_replay)
        self.goals: GoalSpace | None = None
        self.scale = ReturnScale()
        self.queue: list[dict] = []
        self.glearner: GoalLearner | None = None
        self.stats = {"assigned": 0, "reached": 0, "task_assigned": 0, "task_reached": 0, "plans": 0, "plan_steps_reached": 0,
                      "plans_completed": 0, "replans_infeasible": 0, "plan_steps_expired": 0}

    @property
    def learner(self):
        return self.glearner

    def on_hello(self, msg, run=None):
        run = run or next(iter(self.runs))
        try:
            spec = ActionSpec.from_wire(msg["action_spec"])
            if spec.name != SPEC_NAME:
                raise ValueError(f"action spec {spec.name}, expected {SPEC_NAME}")
            obs_spec = parse_obs_spec(msg["obs_spec"])
            sizes = sizes_from_hello(msg, self.a.goals)
            if not (sizes["has_view"] and sizes["has_killed"]):
                raise ValueError("environment older than skills v2.0 (no view_*/killed/clock fields)")
            self.ensure_model(sizes, msg)
        except (ValueError, KeyError, ProtocolError) as e:
            log.error("hello refused: %s", e)
            self.send(msg["env_id"], "hello_ack", run=run, accepted=False, reason=str(e)[:300])
            return
        self.spec, self.obs_spec = spec, obs_spec
        self.envs[msg["env_id"]] = {**self.fresh_env(), "run": run}
        self.send(msg["env_id"], "hello_ack", accepted=True, action_spec_hash=spec.spec_hash())

    def ensure_model(self, sizes, hello):
        if self.model is not None:
            if sizes != self.sizes:
                raise ValueError(f"environment sizes {sizes} differ from the model's {self.sizes}")
            return
        if self.a.population != 1:
            raise ValueError("goal learner: --population 1 only")
        self.sizes, self.hello = sizes, hello
        self.model = GoalWM(sizes, kind_prior=bool(self.a.kind_prior))
        self.goals = GoalSpace(sizes, hello["milestones"])
        self.item_ix = {n: i for i, n in enumerate(hello.get("vocab", {}).get("items", []))}
        params = self.model.init(jax.random.PRNGKey(self.a.seed))
        if self.ckpt.exists():
            with open(self.ckpt, "rb") as f:
                ck = pickle.load(f)
            if ck.get("stats"):
                self.stats.update(ck["stats"])
                ck["stats"] = self.stats
            if ck["version"] != CKPT_VERSION or ck["model"] != self.model.name or ck["sizes"] != sizes:
                raise ValueError("checkpoint does not fit (version, model or sizes); move it away to start fresh")
            self.glearner = GoalLearner(self.a, self.model, ck["params"], ck["opt_state"])
            self.glearner.updates = ck["updates"]
            self.scale.__dict__.update(ck["scale"])
            self.goals.archive = ck["archive"]
            self.total_steps, self.stats = ck["total_steps"], ck["stats"]
            log.info("resumed from %s: update %d, %d steps, archive %d dims", self.ckpt, ck["updates"], self.total_steps, len(self.goals.archive))
        else:
            if self.a.init:
                params = load_pretrained(params, self.a.init)
                log.info("initialised encoder/world model/reach from %s", self.a.init)
            self.glearner = GoalLearner(self.a, self.model, params)
        self.rng = np.random.default_rng(self.a.seed + self.glearner.updates)
        self._serve = jax.jit(self.model.step)
        self._reach = jax.jit(jax.vmap(self.model.reach_logit, in_axes=(None, None, 0, 0, 0)))
        self._penc = jax.jit(lambda P, o: self.model.encode(P["enc"], o)[:2])
        self._pdec = jax.jit(jax.vmap(self.model.plan_logits, in_axes=(None, None, None, None, 0)))
        v = hello.get("vocab", {})
        self.names = {"item": v.get("items", []), "kill": v.get("entities", []), "adv": hello["milestones"] + [f"notable:{n}" for n in hello.get("notable", [])] + ["nights", "slept"]}
        tot, dep = self.model.param_counts(self.glearner.params)
        self.tlog(event="model", name=self.model.name, sizes={k: v for k, v in sizes.items()}, params_total=tot, params_deployed=dep,
                  updates=self.glearner.updates, learn=bool(self.a.learn), init=self.a.init or None,
                  args={k: v for k, v in vars(self.a).items() if k not in ("root", "base")})
        if self.a.learn:
            self.learn_thread = threading.Thread(target=self.learn_loop, daemon=True)
            self.learn_thread.start()

    def fresh_env(self):
        return {"h": np.zeros(self.model.width, np.float32), "pending": None, "seg": None, "ep": None, "goal": None, "cls": None,
                "hist": [], "vals": None, "reached_dims": {}, "plan": None, "firsts": [], "anchors": [], "base": {}, "replans": 0}

    # -- goals --
    def reach_p(self, e, dims: list) -> np.ndarray:
        """max(R(s, g), past success rate) per dimension; R only once the bot has an encoded state."""
        rate = np.array([(self.goals.archive.get(d, {}).get("wins", 0) + 1) / (self.goals.archive.get(d, {}).get("tries", 0) + 2)
                         for d in dims], np.float32)
        if e["cls"] is None or self.glearner is None:
            return rate
        k = np.array([GK[d[0]] for d in dims], np.int32)
        g = np.array([d[1] for d in dims], np.int32)
        r = 1 / (1 + np.exp(-np.asarray(self._reach(self.glearner.params, jnp.asarray(e["cls"]), k[:, None], g[:, None],
                                                     np.ones((len(dims), 1), np.float32)))[:, 0]))
        return np.maximum(r, rate)

    def dim_name(self, d) -> str:
        k, i = d
        if k == "night":
            return "night"
        names = self.names.get(k, [])
        return f"{k}:{names[i] if i < len(names) else i}"

    def make_plan(self, e, x, vals, step) -> dict | None:
        """Batch-generate a plan from this observation: target T (owner task goal or a hard archive dimension), beam search."""
        task, arch = self.goals.task_dims(self.item_ix), list(self.goals.archive)
        if task and (not arch or self.rng.random() < self.a.p_task_plan):
            T = task[self.rng.integers(len(task))]
        elif arch:
            rate = np.array([(self.goals.archive[d]["wins"] + 1) / (self.goals.archive[d]["tries"] + 2) for d in arch])
            w = (1 - rate) + 1e-3
            T = arch[self.rng.choice(len(arch), p=w / w.sum())]
        else:
            return None
        o = {k: v for k, v in x.items() if k not in GOAL_KEYS}
        beams = plan_beam(self.model, self.glearner.params, self._penc, self._pdec, o, dim_index(T, self.sizes), self.a.plan_beam)
        steps = [index_dim(j, self.sizes) for j in (beams[0][0] if beams else [])]
        steps = [d for d in steps if d is not None and d != T][:self.model.L - 1] + [T]
        self.stats["plans"] += 1
        plan = {"T": T, "steps": steps, "k": 0, "t0": step, "logp": round(beams[0][1], 3) if beams else None}
        if self.stats["plans"] % 25 == 1:
            self.tlog(event="plan", n=self.stats["plans"], target=self.dim_name(T), steps=[self.dim_name(d) for d in steps],
                      logp=plan["logp"], updates=self.glearner.updates)
        return plan

    def plan_goal(self, e, x, vals, step) -> dict | None:
        """The next step of the bot's plan as its goal (tokens [step], context [T]); regenerates a plan when there is none, it
        is used up, or the reach head judges its next step physically too early (R < --plan-tau)."""
        for _ in range(2):
            if e["plan"] is None or e["plan"]["k"] >= len(e["plan"]["steps"]):
                if e["replans"] >= 4:
                    return None
                e["plan"], e["replans"] = self.make_plan(e, x, vals, step), e["replans"] + 1
                if e["plan"] is None:
                    return None
            pl = e["plan"]
            d = pl["steps"][pl["k"]]
            p = float(self.reach_p(e, [d])[0])
            if e["cls"] is not None and p < self.a.plan_tau:
                self.stats["replans_infeasible"] += 1
                e["plan"] = None
                continue
            if d in self.goals.archive:
                self.goals.archive[d]["tries"] += 1
            self.stats["assigned"] += 1
            ctx = [(pl["T"], vals.get(pl["T"], 0.0), 1.0)] if d != pl["T"] else []
            return {"tokens": [(d, vals.get(d, 0.0), 1.0)], "ctx": ctx, "t0": step, "task": pl["T"] in self.goals.task_dims(self.item_ix) and d == pl["T"],
                    "p": p, "plan": True}
        return None

    def propose(self, e, vals, step) -> dict:
        task = self.goals.task_dims(self.item_ix)
        arch = list(self.goals.archive)
        is_task = bool(task) and (not arch or self.rng.random() < self.a.p_task)
        if is_task:
            dims = [task[self.rng.integers(len(task))]]
            p_sel = self.reach_p(e, dims)
        else:
            cand = [arch[i] for i in self.rng.permutation(len(arch))[:64]]
            p = self.reach_p(e, cand)
            tries = np.array([self.goals.archive[d]["tries"] for d in cand], np.float32)
            w = p * (1 - p) + 0.05 / (1 + tries)
            n = 2 if (len(cand) > 1 and self.rng.random() < self.a.p_pair) else 1
            idx = self.rng.choice(len(cand), size=n, replace=False, p=w / w.sum())
            dims, p_sel = [cand[i] for i in idx], p[idx]
        for d in dims:
            if d in self.goals.archive:
                self.goals.archive[d]["tries"] += 1
        self.stats["assigned"] += 1
        self.stats["task_assigned"] += int(is_task)
        return {"tokens": [(d, vals.get(d, 0.0), 1.0) for d in dims], "t0": step, "task": is_task, "p": float(np.prod(p_sel))}

    # -- observations --
    def on_obs_batch(self, msgs):
        rows = []
        for m in msgs:
            e = self.envs.get(m["env_id"])
            if e is None:
                continue
            try:
                check_obs(m, self.obs_spec)
            except ProtocolError as err:
                log.warning("obs rejected: %s", err)
                continue
            rows.append((m, e, to_arrays(m["obs"], self.sizes)))
        # A bot can have two observations in one poll (the last of an episode and the first of the next): bookkeeping of the
        # second must follow the handling of the first, so the poll is processed in rounds with each bot at most once.
        rounds: list[list] = []
        for r in rows:
            for rd in rounds:
                if all(q[0]["env_id"] != r[0]["env_id"] for q in rd):
                    rd.append(r)
                    break
            else:
                rounds.append([r])
        for rd in rounds:
            self.serve_round(rd)

    def serve_round(self, rows):
        prepared = [(m, e, x, self.goal_step(m, e, x)) for m, e, x in rows]
        n = self.a.n_max
        for i in range(0, len(prepared), n):
            chunk = prepared[i:i + n]
            arrs = [{**x, **ge} for _, _, x, (ge, _) in chunk]
            batch = {k: np.stack([x[k] for x in arrs] + [arrs[0][k]] * (n - len(arrs))) for k in arrs[0]}
            h = np.stack([e["h"] for _, e, _, _ in chunk] + [chunk[0][1]["h"]] * (n - len(chunk)))
            logits, value, h2, cls = (np.asarray(v) for v in self._serve(self.glearner.params, h, batch))
            for j, (m, e, x, (ge, bonus)) in enumerate(chunk):
                e["cls"] = cls[j]
                self.handle(m, e, arrs[j], logits[j], float(value[j]), h2[j], bonus)

    def goal_step(self, m, e, x):
        """Goal bookkeeping on arrival of an observation: archive, hindsight labels, reached/expired goals, the next goal.
        Returns (goal arrays for this observation, goal bonus earned by the step that led here)."""
        vals = self.goals.values(m["obs"])
        if m["step"] == 0 or e["vals"] is None:
            e["goal"], e["hist"], e["reached_dims"] = None, [], {}
            e["plan"], e["replans"], e["firsts"], e["base"] = None, 0, [], dict(vals)
            e["anchors"] = [(m["step"], {k: v for k, v in x.items()})]
        else:
            self.goals.note_raised(e["vals"], vals)
            seen = {d for _, d in e["firsts"]}
            for d, v in vals.items():
                if v >= e["base"].get(d, 0.0) + 1 and d not in seen:
                    e["firsts"].append((m["step"], d))
            if m["step"] % self.a.plan_anchor == 0 and len(e["anchors"]) < 6:
                e["anchors"].append((m["step"], {k: v for k, v in x.items()}))
        bonus = 0.0
        g = e["goal"]
        if g is not None and self.goals.reached(g, vals):
            n_prev = max(e["reached_dims"].get(d, 0) for d, _, _ in g["tokens"])
            bonus = goal_value(self.a.goal_bonus, g.get("p", 0.5), n_prev)
            for d, _, _ in g["tokens"]:
                e["reached_dims"][d] = e["reached_dims"].get(d, 0) + 1
            self.stats["reached"] += 1
            self.stats["task_reached"] += int(g["task"])
            for d, _, _ in g["tokens"]:
                if d in self.goals.archive:
                    self.goals.archive[d]["wins"] += 1
            if e["ep"]:
                e["ep"]["goals_reached"] += 1
            if g.get("plan") and e["plan"] is not None:
                self.stats["plan_steps_reached"] += 1
                e["plan"]["k"] += 1
                if e["plan"]["k"] >= len(e["plan"]["steps"]):
                    self.stats["plans_completed"] += 1
                    e["plan"], e["replans"] = None, 0
            g = None
        if g is not None and m["step"] - g["t0"] >= self.a.goal_horizon:
            if g.get("plan"):
                self.stats["plan_steps_expired"] += 1
                e["plan"] = None
            g = None
        if m["terminated"] or m["truncated"]:
            if self.a.learn:
                for xo, lab in plan_samples(e["firsts"], e["anchors"], self.sizes, self.model.L, self.rng):
                    self.plan_replay.add(xo, lab)
        elif g is None:
            if self.a.plan and self.rng.random() >= self.a.p_explore:
                g = self.plan_goal(e, x, vals, m["step"])
            if g is None:
                g = self.propose(e, vals, m["step"])
        e["goal"], e["vals"] = g, vals
        return self.goals.encode(g, vals, m["step"], self.a.goal_horizon, self.a.goals), bonus

    def handle(self, m, e, x, logits, value, h2, bonus):
        terms, term, trunc = m.get("reward_terms", {}), m["terminated"], m["truncated"]
        if e["ep"] is None or m["step"] == 0:
            e["ep"] = {"terms": {}, "steps": 0, "episode": m["episode"], "t0": time.time(), "goals_reached": 0, "bonus": 0.0}
        if m["step"] > 0:
            for k, v in terms.items():
                e["ep"]["terms"][k] = e["ep"]["terms"].get(k, 0.0) + float(v)
            e["ep"]["bonus"] += bonus
        p = e["pending"]
        if p is not None and m["step"] == p["step"] + 1:
            s = e["seg"]
            s["obs"].append(p["x"]), s["act"].append(p["a"]), s["logp"].append(p["logp"]), s["val"].append(p["v"])
            s["rew"].append(stage_reward(terms, self.hello["stages"][self.a.ext_stage]) + bonus)
            s["dt"].append(float(m["obs"]["dt_ticks"])), s["done"].append(bool(term)), s["u"].append(p["u"])
            e["hist"].append({"x": {k: v for k, v in p["x"].items() if k not in GOAL_KEYS},
                              "lab": wm_labels(x, m["obs"], term, p["a"], self.sizes["has_killed"]), "vals": e["hist_vals"]})
            if term or trunc or len(s["act"]) >= self.a.seg:
                s["boot"] = 0.0 if term else value
                self.emit(s)
                e["seg"] = None
        e["pending"] = None
        self.flush_hindsight(e, final=term or trunc)
        if term or trunc:
            self.end_episode(m, e)
            e["h"] = np.zeros_like(e["h"])
            return
        if e["seg"] is None:
            e["seg"] = {"h0": e["h"].copy(), "obs": [], "act": [], "logp": [], "val": [], "rew": [], "dt": [], "done": [], "u": []}
        mask = x["cand_mask"]
        lg = np.where(mask, logits, -np.inf)
        pr = np.exp(lg - lg.max())
        pr /= pr.sum()
        a = int(self.rng.choice(len(pr), p=pr))
        e["pending"] = {"step": m["step"], "x": x, "a": a, "logp": float(np.log(pr[a] + 1e-12)), "v": value, "u": self.glearner.updates}
        e["hist_vals"] = e["vals"]
        e["h"] = h2
        e["ep"]["steps"] += 1
        self.total_steps += 1
        self.send(m["env_id"], "act", episode=m["episode"], step=m["step"], obs_seq=m["seq"], action={"candidate": a}, latency_ms=0.0)

    def flush_hindsight(self, e, final: bool):
        """Label every history entry whose next --her-horizon values are known (all of them at the end of an episode) and
        move it into the replay ring."""
        H, hist = self.a.her_horizon, e["hist"]
        if not hist:
            return
        vals_seq = [h["vals"] for h in hist[1:]] + [e["vals"]]
        if not self.a.learn:
            e["hist"] = []
            return
        n_ready = len(hist) if final else max(0, len(hist) - H)
        arch = list(self.goals.archive)
        for t in range(n_ready):
            her = hindsight(hist[t]["vals"], vals_seq[t:t + H], arch, self.rng)
            self.replay.add(hist[t]["x"], {**hist[t]["lab"], **her})
        e["hist"] = [] if final else hist[n_ready:]

    def emit(self, seg):
        if self.a.learn and seg["act"]:
            with self.qlock:
                self.queue.append(seg)

    def end_episode(self, m, e):
        ep = e["ep"] or {"terms": {}, "steps": 0, "t0": time.time(), "goals_reached": 0, "bonus": 0.0}
        prog = m["obs"]["progress"]
        rec = {"t": round(time.time(), 3), "run_id": self.runs[e["run"]]["rid"] if e.get("run") in self.runs else self.run,
               "env_id": m["env_id"], "episode": m["episode"], "steps": ep["steps"], "terminated": m["terminated"],
               "nights": round(prog[-2] * 10, 3), "slept": round(prog[-1] * 10, 3), "terms": {k: round(v, 4) for k, v in ep["terms"].items()},
               "return_task": round(stage_reward(ep["terms"], self.hello["stages"][self.a.ext_stage]), 4),
               "goals_reached": ep["goals_reached"], "goal_bonus": round(ep["bonus"], 3),
               "achievements": int(sum(prog[:len(self.hello["milestones"])])), "wall_s": round(time.time() - ep["t0"], 1)}
        self.ep_log.write(json.dumps(rec) + "\n"), self.ep_log.flush()
        e["ep"], e["goal"], e["vals"] = None, None, None

    # -- learning --
    def learn_loop(self):
        while not self.stop.is_set():
            with self.qlock:
                segs = self.queue[:self.a.batch_segs] if len(self.queue) >= self.a.batch_segs else None
                if segs:
                    self.queue = self.queue[self.a.batch_segs:]
            if not segs:
                time.sleep(0.2)
                continue
            t0 = time.time()
            rep_n = self.replay.n
            mets = self.glearner.update(segs, self.scale, self.replay, self.rng, self.plan_replay)
            top = sorted(self.goals.archive.items(), key=lambda kv: -kv[1]["tries"])[:5]
            self.tlog(event="update", update=self.glearner.updates, seconds=round(time.time() - t0, 2), total_steps=self.total_steps,
                      queue=len(self.queue), staleness_mean=round(float(np.mean([self.glearner.updates - 1 - u for sg in segs for u in sg["u"]])), 3),
                      archive=len(self.goals.archive), goals=dict(self.stats), replay_at_start=rep_n,
                      top_goals=[[k[0], k[1], v["tries"], v["wins"]] for k, v in top], **mets)
            self.save()

    def save(self):
        ck = {"version": CKPT_VERSION, "sizes": self.sizes, "model": self.model.name, "params": jax.device_get(self.glearner.params),
              "opt_state": jax.device_get(self.glearner.opt_state), "updates": self.glearner.updates, "scale": dict(self.scale.__dict__),
              "archive": self.goals.archive, "total_steps": self.total_steps, "stats": self.stats, "t": time.time()}
        tmp = self.ckpt.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(ck, f)
        os.replace(tmp, self.ckpt)


def load_pretrained(params, path):
    """Copy the encoder, world model, reach head and goal embeddings of an offline-pretrained parameter file (scripts/
    botsystem_wm_pretrain.py) into fresh parameters; shapes must match, the policy part stays fresh."""
    with open(path, "rb") as f:
        pre = pickle.load(f)["params"]
    out = dict(params)
    for part in ("enc", "wm", "reach", "goal"):
        if part in pre:
            ok = jax.tree_util.tree_all(jax.tree_util.tree_map(lambda x, y: np.shape(x) == np.shape(y), params[part], pre[part]))
            if not ok:
                raise ValueError(f"pretrained part '{part}' does not fit the model")
            out[part] = jax.tree_util.tree_map(jnp.asarray, pre[part])
    return out


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    a = parse_args(argv)
    s = GoalServer(a)
    signal.signal(signal.SIGTERM, lambda *_: s.stop.set())
    try:
        s.loop()
    finally:
        if a.learn and s.glearner is not None:
            s.save()
        s.tlog(event="stop")


if __name__ == "__main__":
    main()
