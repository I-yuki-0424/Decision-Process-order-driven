#!/usr/bin/env python3
"""On-demand policy server + asynchronous PPO learner for the BOTSYSTEM skill-level environment (forge1201-MODED-AIPlayer
docs/proposed/RL-TRAINING-v2.md), with the owner's two-stage curriculum. Side track: nothing here counts toward a roadmap gate.

  python scripts/botsystem_skill_ppo.py --root <exchange> --base train [--learn 1] [--seed 42]

Serving (main thread): follows <root>/<base>.current (the environment's live run; a restart of the environment is a new run),
answers every observation as it arrives (bots step at their own pace: inference on demand, not batch generation of actions),
one forward pass for whatever arrived in the same poll. Learning (second thread, --learn 1): segments of SEG transitions per
bot are cut as they complete; every BATCH segments one PPO update (clipped ratio, GAE, entropy bonus, global-norm clip);
the server switches to the new parameters between polls. Behaviour log-probs are stored with each transition, so an action
taken with parameters one update old is still corrected by the ratio (staleness <= 1 update, measured in train_log).

Semi-MDP: a step lasts as long as its command (dt ticks); the discount per step is gamma_s ** (dt / 20), gamma per second.

Curriculum (owner's proposal, 2026-11-01): stage 'survive' (target T = ten nights survived, all slept; reward weights from the
environment's table: survival 1, other terms 0.1) until survival is stable, then stage 'dragon' (T = the Ender Dragon is dead;
all weights 1). Stable = the last W-episode block means of nights survived differ from their mean by <= tol (relative) for
`windows` blocks and that mean >= min_nights; or the stage-1 step budget is spent (logged as the reason). Each stage has its own
value head and return scale. The target is the learner's: the environment sends progress only, never a goal.

Population (--population M, default 1; owner 2026-11-01: keep the greedy learner and add cautious ones, combine like a
genetic algorithm): M members with their own parameters, optimiser state, return scale, entropy coefficient and per-term
reward multipliers (member 0 'greedy' = the stage weights as they are; member 1 'cautious' = invalid x3, survival x2, kills
x0.5; further members drawn around 1). Bot env_id % M acts for member env_id % M. Every --pbt-interval updates of every
member, once each has --pbt-window finished episodes, members are ranked by the mean return under the unmodified stage
weights (fitness never uses a member's own multipliers, so members are comparable); the worst takes a copy of the best's
parameters and optimiser state and the best's settings, each multiplied by 0.8 or 1.25 (exploit/explore of population-based
training, Jaderberg et al. 2017). With M = 1 this is the single learner of before.

Outputs in <root>/<base>/train/: checkpoint.pkl (every update, atomic; resumed on start), train_log.jsonl (updates, stage
switches, run changes, pbt), episodes.jsonl (one line per finished episode: return per term, nights, slept, steps, stage,
member).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
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
from integrations.botsystem.file_channel import LineReader, LineWriter  # noqa: E402
from integrations.botsystem.protocol import ProtocolError, check_obs, parse_obs_spec  # noqa: E402
from botsystem_skill_truck import INT_KEYS, OBS_KEYS, SkillTruck, sizes_from_hello  # noqa: E402

log = logging.getLogger("skill_ppo")
STAGES = ("survive", "dragon")
SPEC_NAME = "skills_v1"
CKPT_VERSION = 2
HP_TERMS = ("survival", "kills", "notable", "mined", "achievements", "invalid")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a = p.add_argument
    a("--root", required=True), a("--base", required=True), a("--learn", type=int, default=1), a("--seed", type=int, default=42)
    a("--n-max", type=int, default=16), a("--seg", type=int, default=32), a("--batch-segs", type=int, default=16)
    a("--epochs", type=int, default=3), a("--minibatches", type=int, default=4), a("--lr", type=float, default=3e-4)
    a("--gamma-s", type=float, default=0.999), a("--lam", type=float, default=0.95), a("--clip", type=float, default=0.2)
    a("--ent", type=float, default=0.01), a("--vf", type=float, default=0.5), a("--max-grad-norm", type=float, default=0.5)
    a("--stage-window", type=int, default=100), a("--stage-windows", type=int, default=3), a("--stage-tol", type=float, default=0.1)
    a("--stage-min-nights", type=float, default=1.0), a("--stage1-max-steps", type=int, default=600_000)
    a("--poll-s", type=float, default=0.01)
    a("--population", type=int, default=1), a("--pbt-interval", type=int, default=20), a("--pbt-window", type=int, default=20)
    a("--kind-prior", type=int, default=1)
    return p.parse_args(argv)


# ---------------------------------------------------------------- curriculum ------------------------------------------------
class Curriculum:
    def __init__(self, a, milestones: list[str], notable: list[str]):
        self.a, self.stage, self.nights, self.steps, self.switched = a, 0, [], [0, 0], None
        self.m, self.n = milestones, notable

    def target(self) -> np.ndarray:
        """T in the environment's progress space (milestones, notable kill flags, nights/10, slept/10)."""
        v = np.zeros(len(self.m) + len(self.n) + 2, np.float32)
        if STAGES[self.stage] == "survive":
            v[-2:] = 1.0
        else:
            v[self.m.index("minecraft:end/kill_dragon")] = 1.0
            v[len(self.m) + self.n.index("ender_dragon")] = 1.0
        return v

    def end_episode(self, nights: float) -> str | None:
        """Record one finished episode; returns the switch reason when the stage changes."""
        if self.stage:
            return None
        self.nights.append(nights)
        W, k = self.a.stage_window, self.a.stage_windows
        why = None
        if len(self.nights) >= W * k:
            means = [float(np.mean(self.nights[len(self.nights) - (i + 1) * W:len(self.nights) - i * W])) for i in range(k)]
            m = float(np.mean(means))
            if m >= self.a.stage_min_nights and all(abs(x - m) <= self.a.stage_tol * m for x in means):
                why = f"stable: last {k} blocks of {W} episodes {['%.3f' % x for x in reversed(means)]} nights"
        if why is None and self.steps[0] >= self.a.stage1_max_steps:
            why = f"stage-1 budget spent ({self.steps[0]} steps), survival not stable"
        if why:
            self.stage, self.switched = 1, {"t": time.time(), "steps": self.steps[0], "episodes": len(self.nights), "reason": why}
        return why

    def state(self):
        return {"stage": self.stage, "nights": self.nights, "steps": self.steps, "switched": self.switched}

    def load(self, s):
        self.stage, self.nights, self.steps, self.switched = s["stage"], list(s["nights"]), list(s["steps"]), s["switched"]


class ReturnScale:
    """Running std of discounted returns per stage (rewards are divided by it; the mean is left alone)."""

    def __init__(self):
        self.n, self.m, self.s2 = [1e-4] * 2, [0.0] * 2, [1.0] * 2

    def update(self, stage, x: np.ndarray):
        if not x.size:
            return
        bm, bv, bn = float(x.mean()), float(x.var()), x.size
        n, m = self.n[stage], self.m[stage]
        d, tot = bm - m, n + bn
        self.m[stage] = m + d * bn / tot
        self.s2[stage] = (self.s2[stage] * n + bv * bn + d * d * n * bn / tot) / tot
        self.n[stage] = tot

    def std(self, stage):
        return max(math.sqrt(self.s2[stage]), 1e-3)


# ---------------------------------------------------------------- math (pure, tested) ---------------------------------------
def gae(rew, val, boot, dt, done, gamma_s, lam):
    """Semi-MDP GAE over one segment: discount per step gamma_s ** (dt / 20). Returns (advantages, returns)."""
    T = len(rew)
    adv = np.zeros(T, np.float32)
    nxt, a = boot, 0.0
    for t in reversed(range(T)):
        g = gamma_s ** (dt[t] / 20.0)
        nd = 0.0 if done[t] else 1.0
        delta = rew[t] + g * nxt * nd - val[t]
        a = delta + g * lam * nd * a
        adv[t], nxt = a, val[t]
    return adv, adv + np.asarray(val, np.float32)


def stage_reward(terms: dict, weights: dict) -> float:
    return float(sum(weights.get(k, 0.0) * float(v) for k, v in terms.items()))


def masked_entropy(logits, mask):
    lp = jax.nn.log_softmax(logits, -1)
    return -jnp.sum(jnp.where(mask, jnp.exp(lp) * lp, 0.0), -1)


# ---------------------------------------------------------------- learner ---------------------------------------------------
class Learner:
    def __init__(self, a, model: SkillTruck, params, opt_state=None):
        self.a, self.model = a, model
        self.opt = optax.chain(optax.clip_by_global_norm(a.max_grad_norm), optax.adam(a.lr))
        self.params, self.opt_state = params, opt_state if opt_state is not None else self.opt.init(params)
        self.updates = 0
        self._update = jax.jit(self._update_fn)

    def _loss(self, params, b, ent_coef):
        def body(h, x):
            o, st = x
            logits, value, h2 = self.model.step(params, h, o, st)
            return h2, (logits, value)
        obs_t = {k: jnp.swapaxes(v, 0, 1) for k, v in b["obs"].items()}       # (T, B, ...)
        stage_t = jnp.broadcast_to(b["stage"][None], obs_t["cand_mask"].shape[:2])
        _, (logits, value) = jax.lax.scan(body, b["h0"], (obs_t, stage_t))
        logits, value = jnp.swapaxes(logits, 0, 1), jnp.swapaxes(value, 0, 1)  # (B, T, K), (B, T)
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
                      "clipfrac": mean((jnp.abs(ratio - 1) > self.a.clip).astype(jnp.float32))}

    def _update_fn(self, params, opt_state, b, ent_coef):
        (loss, aux), g = jax.value_and_grad(self._loss, has_aux=True)(params, b, ent_coef)
        up, opt_state = self.opt.update(g, opt_state, params)
        return optax.apply_updates(params, up), opt_state, {**aux, "loss": loss}

    def update(self, segs, scale: ReturnScale, ent_coef: float | None = None):
        """segs: list of dicts from Server._emit; all of one stage. Returns metrics."""
        ent_coef = jnp.float32(self.a.ent if ent_coef is None else ent_coef)
        a = self.a
        T = a.seg
        stage = segs[0]["stage"]
        sd = scale.std(stage)
        advs, rets = [], []
        for s in segs:
            r = np.asarray(s["rew"], np.float32) / sd
            ad, rt = gae(r, s["val"], s["boot"], s["dt"], s["done"], a.gamma_s, a.lam)
            advs.append(ad), rets.append(rt)
        scale.update(stage, np.concatenate(rets) * sd)
        flat = np.concatenate(advs)
        mu, sg = flat.mean(), flat.std() + 1e-8
        B = len(segs)
        pad = lambda x, fill=0: np.concatenate([x, np.full((T - len(x),) + np.shape(x)[1:], fill, np.asarray(x).dtype)]) if len(x) < T else np.asarray(x)
        batch = {
            "obs": {k: np.stack([pad(np.stack([o[k] for o in s["obs"]])) for s in segs]) for k in OBS_KEYS + ("target",)},
            "h0": np.stack([s["h0"] for s in segs]), "stage": np.full((B,), stage, np.int32),
            "act": np.stack([pad(np.asarray(s["act"], np.int32)) for s in segs]),
            "logp": np.stack([pad(np.asarray(s["logp"], np.float32)) for s in segs]),
            "adv": np.stack([pad((ad - mu) / sg) for ad in advs]), "ret": np.stack([pad(rt) for rt in rets]),
            "valid": np.stack([pad(np.ones(len(s["act"]), np.float32)) for s in segs]),
        }
        batch["obs"]["cand_mask"][..., 0] |= batch["valid"] == 0   # padded rows: keep one candidate valid
        mets = []
        rng = np.random.default_rng(self.updates)
        mb = max(1, B // a.minibatches)
        params, opt_state = self.params, self.opt_state
        for _ in range(a.epochs):
            perm = rng.permutation(B)
            for i in range(0, B, mb):
                idx = perm[i:i + mb]
                sub = jax.tree_util.tree_map(lambda x: x[idx], batch)
                params, opt_state, m = self._update(params, opt_state, sub, ent_coef)
                mets.append({k: float(v) for k, v in m.items()})
        self.params, self.opt_state = params, opt_state
        self.updates += 1
        out = {k: float(np.mean([m[k] for m in mets])) for k in mets[0]}
        out.update(segments=B, steps=int(batch["valid"].sum()), ret_std=sd, stage=STAGES[stage],
                   adv_raw_mean=float(mu), reward_mean=float(np.mean(np.concatenate([s["rew"] for s in segs]))))
        return out


# ---------------------------------------------------------------- population ------------------------------------------------
class Member:
    """One learner of the population: its own parameters (in .learner), return scale, segment queue, settings, fitness."""

    def __init__(self, idx: int, name: str, ent: float, w: dict):
        self.idx, self.name, self.ent, self.w = idx, name, float(ent), {k: float(v) for k, v in w.items()}
        self.learner: Learner | None = None
        self.scale = ReturnScale()
        self.queue: list[dict] = []
        self.fit: list[float] = []       # returns of finished episodes under the unmodified stage weights
        self.since = 0                   # updates since the last population step

    @property
    def params(self):
        return self.learner.params

    def weights(self, base: dict) -> dict:
        return {k: v * self.w.get(k, 1.0) for k, v in base.items()}

    def hp(self) -> dict:
        return {"member": self.idx, "name": self.name, "ent": round(self.ent, 5), "w": {k: round(v, 3) for k, v in self.w.items()}}


def initial_members(a, rng) -> list[Member]:
    out = [Member(0, "greedy", a.ent, {}), Member(1, "cautious", a.ent, {"invalid": 3.0, "survival": 2.0, "kills": 0.5})]
    for i in range(2, a.population):
        out.append(Member(i, f"m{i}", float(np.exp(rng.uniform(np.log(a.ent / 3), np.log(a.ent * 3)))),
                          {k: float(np.exp(rng.uniform(np.log(0.5), np.log(2.0)))) for k in HP_TERMS}))
    return out[:a.population]


def pbt_step(members: list[Member], a, rng) -> dict | None:
    """Exploit/explore once every member has had --pbt-interval updates and --pbt-window episodes: the worst member (mean
    fitness over its last window) takes the best's parameters, optimiser state and return scale, and the best's settings
    perturbed by x0.8 or x1.25 each. Returns the record logged as event 'pbt', or None when it is not time yet."""
    if len(members) < 2 or any(m.since < a.pbt_interval or len(m.fit) < a.pbt_window for m in members):
        return None
    fit = {m.idx: float(np.mean(m.fit[-a.pbt_window:])) for m in members}
    best, worst = max(members, key=lambda m: fit[m.idx]), min(members, key=lambda m: fit[m.idx])
    for m in members:
        m.since = 0
    rec = {"fitness": {str(k): round(v, 4) for k, v in fit.items()}, "exploit": None}
    if best is worst or fit[best.idx] <= fit[worst.idx]:
        return rec
    f = lambda: float(rng.choice([0.8, 1.25]))
    worst.learner.params, worst.learner.opt_state = best.learner.params, best.learner.opt_state   # immutable pytrees
    worst.scale.__dict__.update({k: list(v) for k, v in best.scale.__dict__.items()})
    worst.ent = float(np.clip(best.ent * f(), 1e-3, 0.1))
    worst.w = {k: float(np.clip(best.w.get(k, 1.0) * f(), 0.25, 4.0)) for k in HP_TERMS}
    worst.name, worst.fit, worst.queue = f"{best.name}>{worst.idx}", [], []
    rec["exploit"] = {"from": best.idx, "to": worst.idx, "new": worst.hp()}
    return rec


# ---------------------------------------------------------------- server ----------------------------------------------------
def to_arrays(obs: dict, target: np.ndarray) -> dict:
    out = {k: np.asarray(obs[k], np.int32 if k in INT_KEYS else (np.bool_ if k == "cand_mask" else np.float32)) for k in OBS_KEYS}
    out["target"] = target
    return out


class Server:
    def __init__(self, a):
        self.a, self.root, self.base = a, Path(a.root), a.base
        self.out_dir = self.root / self.base / "train"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.train_log = open(self.out_dir / "train_log.jsonl", "a")
        self.ep_log = open(self.out_dir / "episodes.jsonl", "a")
        self.run = self.reader = None      # the run followed last (logs, tests)
        self.runs: dict[str, dict] = {}     # current file -> {rid, reader, writer, seq}: one per environment shard
        self.envs: dict[int, dict] = {}
        self.obs_spec = self.spec = None
        self.model = self.curr = self.hello = None
        self.members: list[Member] = []
        self.rng = np.random.default_rng(a.seed)
        self.qlock = threading.Lock()
        self.stop = threading.Event()
        self.total_steps = 0
        self.ckpt = self.out_dir / "checkpoint.pkl"

    def tlog(self, **kw):
        self.train_log.write(json.dumps({"t": round(time.time(), 3), **kw}) + "\n"), self.train_log.flush()

    # -- run following --
    def follow(self):
        """Follow every environment shard of this base: <base>.current and <base>.s<i>.current each name the live run of one
        environment process (a restart of a shard is a new run; its bots' state is dropped and they say hello again)."""
        for cur in [self.root / f"{self.base}.current", *sorted(self.root.glob(f"{self.base}.s*.current"))]:
            try:
                rid = cur.read_text().strip()
            except OSError:
                continue
            key = cur.name
            if (key in self.runs and self.runs[key]["rid"] == rid) or not (self.root / rid / "env_to_policy.jsonl").exists():
                continue
            d = self.root / rid
            self.runs[key] = {"rid": rid, "reader": LineReader(d / "env_to_policy.jsonl", run_id=rid, allowed=("hello", "obs", "bye")),
                              "writer": LineWriter(d / "policy_to_env.jsonl"), "seq": 0}
            for e in [k for k, v in self.envs.items() if v["run"] == key]:
                del self.envs[e]
            self.run, self.reader = rid, self.runs[key]["reader"]
            self.tlog(event="run", run_id=rid, shard=key)
            log.info("following run %s (%s)", rid, key)

    def send(self, env_id, type_, run=None, **kw):
        r = self.runs[run or self.envs[env_id]["run"]]
        r["writer"].send({"v": 1, "type": type_, "run_id": r["rid"], "env_id": env_id, "seq": r["seq"], "t": round(time.time(), 3), **kw})
        r["seq"] += 1

    # -- hello --
    def on_hello(self, msg, run=None):
        run = run or next(iter(self.runs))
        try:
            spec = ActionSpec.from_wire(msg["action_spec"])
            if spec.name != SPEC_NAME:
                raise ValueError(f"action spec {spec.name}, expected {SPEC_NAME}")
            obs_spec = parse_obs_spec(msg["obs_spec"])
            sizes = sizes_from_hello(msg)
            self.ensure_model(sizes, msg)
        except (ValueError, KeyError, ProtocolError) as e:
            log.error("hello refused: %s", e)
            self.send(msg["env_id"], "hello_ack", run=run, accepted=False, reason=str(e)[:300])
            return
        self.spec, self.obs_spec = spec, obs_spec
        self.envs[msg["env_id"]] = {**self.fresh_env(), "run": run}
        self.send(msg["env_id"], "hello_ack", accepted=True, action_spec_hash=spec.spec_hash())

    @property
    def learner(self):
        """Member 0's learner (the only one with --population 1)."""
        return self.members[0].learner if self.members else None

    def member(self, env_id: int) -> Member:
        return self.members[env_id % len(self.members)]

    def ensure_model(self, sizes, hello):
        if self.model is not None:
            if sizes != self.sizes:
                raise ValueError(f"environment sizes {sizes} differ from the model's {self.sizes}")
            return
        self.sizes, self.hello = sizes, hello
        self.model = SkillTruck(sizes, kind_prior=bool(self.a.kind_prior))
        self.curr = Curriculum(self.a, hello["milestones"], hello["notable"])
        self.members = initial_members(self.a, self.rng)
        for m in self.members:
            m.learner = Learner(self.a, self.model, self.model.init(jax.random.PRNGKey(self.a.seed + m.idx)))
        if self.ckpt.exists():
            with open(self.ckpt, "rb") as f:
                ck = pickle.load(f)
            if ck["version"] != CKPT_VERSION or ck["sizes"] != sizes or len(ck["members"]) != len(self.members) or ck["model"] != self.model.name:
                raise ValueError("checkpoint does not fit (version, sizes, population or model); move it away to start fresh")
            for m, c in zip(self.members, ck["members"]):
                m.name, m.ent, m.w, m.fit = c["name"], c["ent"], c["w"], list(c["fit"])
                m.learner.params, m.learner.opt_state, m.learner.updates = c["params"], c["opt_state"], c["updates"]
                m.scale.__dict__.update(c["scale"])
            self.curr.load(ck["curriculum"])
            self.total_steps = ck["total_steps"]
            self.rng = np.random.default_rng(self.a.seed + sum(m.learner.updates for m in self.members))
            log.info("resumed from %s: updates %s, stage %s, %d steps", self.ckpt, [m.learner.updates for m in self.members],
                     STAGES[self.curr.stage], self.total_steps)
        self._serve = jax.jit(self.model.step)
        tot, dep = self.model.param_counts(self.members[0].params)
        self.tlog(event="model", name=self.model.name, sizes=sizes, params_total=tot, params_deployed=dep, stage=STAGES[self.curr.stage],
                  updates=[m.learner.updates for m in self.members], learn=bool(self.a.learn), population=[m.hp() for m in self.members],
                  args={k: v for k, v in vars(self.a).items() if k not in ("root", "base")})
        if self.a.learn:
            self.learn_thread = threading.Thread(target=self.learn_loop, daemon=True)
            self.learn_thread.start()

    def fresh_env(self):
        return {"h": np.zeros(self.model.width, np.float32), "pending": None, "seg": None, "ep": None}

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
            rows.append((m, e))
        if not rows:
            return
        target, stage = self.curr.target(), self.curr.stage
        n = self.a.n_max
        for mem in self.members:                    # one forward pass per member over its bots that arrived in this poll
            mine = [r for r in rows if self.member(r[0]["env_id"]) is mem]
            for i in range(0, len(mine), n):
                chunk = mine[i:i + n]
                arrs = [to_arrays(m["obs"], target) for m, _ in chunk]
                batch = {k: np.stack([x[k] for x in arrs] + [arrs[0][k]] * (n - len(arrs))) for k in arrs[0]}
                h = np.stack([e["h"] for _, e in chunk] + [chunk[0][1]["h"]] * (n - len(chunk)))
                logits, value, h2 = self._serve(mem.params, h, batch, np.full((n,), stage, np.int32))
                logits, value, h2 = np.asarray(logits), np.asarray(value), np.asarray(h2)
                for j, (m, e) in enumerate(chunk):
                    self.handle(m, e, arrs[j], logits[j], float(value[j]), h2[j], stage)

    def handle(self, m, e, x, logits, value, h2, stage):
        terms, term, trunc = m.get("reward_terms", {}), m["terminated"], m["truncated"]
        if e["ep"] is None or m["step"] == 0:
            e["ep"] = {"terms": {}, "steps": 0, "episode": m["episode"], "stage": stage, "t0": time.time()}
        if m["step"] > 0:
            for k, v in terms.items():
                e["ep"]["terms"][k] = e["ep"]["terms"].get(k, 0.0) + float(v)
        p = e["pending"]
        if p is not None and m["step"] == p["step"] + 1:
            s = e["seg"]
            s["obs"].append(p["x"]), s["act"].append(p["a"]), s["logp"].append(p["logp"]), s["val"].append(p["v"])
            s["rew"].append(stage_reward(terms, self.member(m["env_id"]).weights(self.hello["stages"][STAGES[s["stage"]]])))
            s["dt"].append(float(m["obs"]["dt_ticks"])), s["done"].append(bool(term)), s["u"].append(p["u"])
            if term or trunc or len(s["act"]) >= self.a.seg:
                s["boot"] = 0.0 if term or s["stage"] != stage else value
                self.emit(s, self.member(m["env_id"]))
                e["seg"] = None
        e["pending"] = None
        if term or trunc:
            self.end_episode(m, e)
            e["h"] = np.zeros_like(e["h"])
            return
        if e["seg"] is None or e["seg"]["stage"] != stage:
            e["seg"] = {"h0": e["h"].copy(), "stage": stage, "obs": [], "act": [], "logp": [], "val": [], "rew": [], "dt": [], "done": [], "u": []}
        mask = x["cand_mask"]
        lg = np.where(mask, logits, -np.inf)
        pr = np.exp(lg - lg.max())
        pr /= pr.sum()
        a = int(self.rng.choice(len(pr), p=pr))
        e["pending"] = {"step": m["step"], "x": x, "a": a, "logp": float(np.log(pr[a] + 1e-12)), "v": value,
                        "u": self.member(m["env_id"]).learner.updates}
        e["h"] = h2
        e["ep"]["steps"] += 1
        self.total_steps += 1
        self.curr.steps[stage] += 1
        self.send(m["env_id"], "act", episode=m["episode"], step=m["step"], obs_seq=m["seq"], action={"candidate": a},
                  latency_ms=0.0)

    def emit(self, seg, mem: Member):
        if not self.a.learn or not seg["act"]:
            return
        with self.qlock:
            mem.queue.append(seg)

    def end_episode(self, m, e):
        ep = e["ep"] or {"terms": {}, "steps": 0, "stage": self.curr.stage, "t0": time.time()}
        prog = m["obs"]["progress"]
        nights, slept = round(prog[-2] * 10, 3), round(prog[-1] * 10, 3)
        rec = {"t": round(time.time(), 3), "run_id": self.runs[e["run"]]["rid"] if e.get("run") in self.runs else self.run, "env_id": m["env_id"], "episode": m["episode"], "steps": ep["steps"],
               "terminated": m["terminated"], "stage": STAGES[ep["stage"]], "nights": nights, "slept": slept,
               "terms": {k: round(v, 4) for k, v in ep["terms"].items()},
               "return_stage": round(stage_reward(ep["terms"], self.hello["stages"][STAGES[ep["stage"]]]), 4),
               "achievements": int(sum(prog[:len(self.hello["milestones"])])), "wall_s": round(time.time() - ep["t0"], 1),
               "member": self.member(m["env_id"]).idx}
        self.ep_log.write(json.dumps(rec) + "\n"), self.ep_log.flush()
        with self.qlock:
            self.member(m["env_id"]).fit.append(rec["return_stage"])
        if self.a.learn:
            why = self.curr.end_episode(nights)
            if why:
                self.tlog(event="stage_switch", to=STAGES[self.curr.stage], **self.curr.switched)
                log.info("curriculum: stage -> %s (%s)", STAGES[self.curr.stage], why)
        e["ep"] = None

    # -- learning thread --
    def learn_loop(self):
        while not self.stop.is_set():
            did = False
            for mem in self.members:
                with self.qlock:
                    stage = self.curr.stage
                    mem.queue = [s for s in mem.queue if s["stage"] == stage]   # a stage switch drops older segments
                    segs = mem.queue[:self.a.batch_segs] if len(mem.queue) >= self.a.batch_segs else None
                    if segs:
                        mem.queue = mem.queue[self.a.batch_segs:]
                if not segs:
                    continue
                t0 = time.time()
                mets = mem.learner.update(segs, mem.scale, mem.ent)   # the server reads mem.params from its next poll on
                mem.since += 1
                did = True
                self.tlog(event="update", update=mem.learner.updates, member=mem.idx, seconds=round(time.time() - t0, 2),
                          total_steps=self.total_steps, queue=len(mem.queue), ent_coef=round(mem.ent, 5),
                          staleness_mean=round(float(np.mean([mem.learner.updates - 1 - u for sg in segs for u in sg["u"]])), 3), **mets)
                with self.qlock:
                    rec = pbt_step(self.members, self.a, self.rng)
                if rec:
                    self.tlog(event="pbt", **rec)
                    if rec["exploit"]:
                        log.info("population: member %s <- %s %s", rec["exploit"]["to"], rec["exploit"]["from"], rec["fitness"])
            if did:
                self.save()
            else:
                time.sleep(0.2)

    def save(self):
        mem = [{"idx": m.idx, "name": m.name, "ent": m.ent, "w": m.w, "fit": m.fit[-self.a.pbt_window:],
                "params": jax.device_get(m.learner.params), "opt_state": jax.device_get(m.learner.opt_state),
                "updates": m.learner.updates, "scale": dict(m.scale.__dict__)} for m in self.members]
        ck = {"version": CKPT_VERSION, "sizes": self.sizes, "model": self.model.name, "members": mem, "curriculum": self.curr.state(),
              "total_steps": self.total_steps, "t": time.time()}
        tmp = self.ckpt.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(ck, f)
        os.replace(tmp, self.ckpt)

    def loop(self):
        last_follow = 0.0
        while not self.stop.is_set():
            if time.time() - last_follow > 2.0:
                self.follow()
                last_follow = time.time()
            if not self.runs:
                time.sleep(0.5)
                continue
            obs = []
            for key, r in list(self.runs.items()):
                for m in r["reader"].poll():
                    if m["type"] == "hello":
                        self.on_hello(m, key)
                    elif m["type"] == "obs":
                        if self.envs.get(m["env_id"], {}).get("run") == key:
                            obs.append(m)
                    elif m["type"] == "bye" and self.envs.get(m["env_id"], {}).get("run") == key:
                        self.envs.pop(m["env_id"], None)
            if obs:
                self.on_obs_batch(obs)
            else:
                time.sleep(self.a.poll_s)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    a = parse_args(argv)
    s = Server(a)
    signal.signal(signal.SIGTERM, lambda *_: s.stop.set())
    try:
        s.loop()
    finally:
        if a.learn and s.learner is not None:
            s.save()
        s.tlog(event="stop")


if __name__ == "__main__":
    main()
