"""Shared harness for the Phase-1 model-family comparison (roadmap v10, EP-A rules with a reduced training budget = "EP-A-mini").

Identical for every arm (only the `Arm` policy module differs, see src/model/epa_policies.py):
  * environment: CraftaxClassicSymbolicEnv, default parameters (episode limit 10000 = the env default, no T cap), the raw
    1345-d symbolic observation, reward unchanged. Env auto-resets on `done`, so metrics use masked_achievements().
  * learner: PPO (GAE), same optimiser/clipping/entropy for all arms; recurrent arms use sequence minibatches over envs.
  * evaluation: sampled (stochastic) policy, FIRST episode of >= 256 fresh envs per seed, final parameters only.
        reward_pct = mean over episodes of (distinct achievements / 22) x 100   (NOT the env return)
        score_pct  = Crafter score exp(mean_i ln(1 + s_i)) - 1, s_i = % of episodes that unlocked achievement i
  * budget: every environment step of every env copy is counted in `env_steps_total`.
Nothing here reads simulator state at decision time: the policy sees only the observation (and its own past
observations/actions through the carry).

Recipe options of the 1M-step reference baselines (Dedieu et al. 2025 Table 3; Moon et al. 2023), default off so earlier
configs run unchanged: `value_norm` = EMA decay of the value-target mean/std (the critic then predicts standardized
targets, GAE runs on de-standardized values), `adv_norm="batch"` standardizes the GAE once over the whole batch.
"""
import hashlib
import json
import os
import platform
import subprocess
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv

from src.environment.craftax_env_adapter import NUM_ACHIEVEMENTS, calculate_crafter_score, masked_achievements
from src.model.epa_policies import Arm, ev_loss, make_optimizer, tp_loss

_ENV = CraftaxClassicSymbolicEnv()
_PARAMS = _ENV.default_params
EPISODE_LIMIT = int(_PARAMS.max_timesteps)   # 10000, the environment default


def env_reset(keys):
    obs, st = jax.vmap(_ENV.reset, in_axes=(0, None))(keys, _PARAMS)
    return obs.astype(jnp.float32), st


def env_step(keys, st, act):
    obs, st2, rew, done, _ = jax.vmap(_ENV.step, in_axes=(0, 0, 0, None))(keys, st, act, _PARAMS)
    return obs.astype(jnp.float32), st2, rew.astype(jnp.float32), done


class PPOConfig(NamedTuple):
    total_steps: int = 100_000
    num_envs: int = 16
    num_steps: int = 32
    epochs: int = 4
    minibatches: int = 4
    lr: float = 1e-3
    anneal_lr: bool = True
    gamma: float = 0.99
    lam: float = 0.8
    clip: float = 0.2
    ent: float = 0.01
    vf: float = 0.5
    max_grad_norm: float = 1.0
    value_norm: float = 0.0        # EMA decay of the value-target mean/std; 0 = off (raw targets)
    adv_norm: str = "minibatch"    # GAE standardization: "minibatch" (per minibatch) | "batch" (once per update)
    warmup: float = 0.0            # fraction of optimiser steps with a linear lr ramp-up from 0 (0 = off)
    remat: bool = False            # recompute the per-step policy forward in the backward pass (recurrent arms): same maths, less VRAM
    ent_final: float = -1.0        # >= 0: entropy coefficient annealed linearly from `ent` (first update) to `ent_final` (last); < 0 = constant


def _vn_stats(vn):
    """(mean, std) of the value targets from the EMA accumulators vn = [E[R], E[R^2], debias weight]; (0, 1) before the first update."""
    seen = vn[2] > 0
    w = jnp.where(seen, vn[2], 1.0)
    mean = jnp.where(seen, vn[0] / w, 0.0)
    var = jnp.where(seen, vn[1] / w - mean ** 2, 1.0)
    return mean, jnp.sqrt(jnp.maximum(var, 1e-2))


def _vn_update(vn, ret, decay):
    return decay * vn + (1.0 - decay) * jnp.stack([ret.mean(), (ret ** 2).mean(), jnp.ones(())])


# ----------------------------------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------------------------------
def summarize_achievements(ach: np.ndarray) -> dict:
    """ach: (episodes, 22) 0/1 achievements unlocked during each episode."""
    per_ep = ach.sum(1) / NUM_ACHIEVEMENTS * 100.0
    rates = ach.mean(0) * 100.0
    return {"reward_pct": float(per_ep.mean()), "score_pct": calculate_crafter_score(list(rates)),
            "achievement_rates_pct": [float(x) for x in rates]}


def make_eval_chunk(arm: Arm, chunk: int):
    @jax.jit
    def eval_chunk(params, carry, env_st, obs, ach, alive, length, key):
        def body(c, k):
            carry, env_st, obs, ach, alive, length = c
            k1, k2 = jax.random.split(k)
            logits, _, caux = arm.step(params, carry, obs)
            a = jax.random.categorical(k1, logits)
            nobs, env_st2, _, d = env_step(jax.random.split(k2, obs.shape[0]), env_st, a)
            ach = jnp.maximum(ach, masked_achievements(env_st2.achievements, d[:, None], alive[:, None]))
            length = length + alive.astype(jnp.int32)
            alive = alive & ~d
            return (arm.advance(caux, obs, a, d), env_st2, nobs, ach, alive, length), None

        c, _ = jax.lax.scan(body, (carry, env_st, obs, ach, alive, length), jax.random.split(key, chunk))
        return c

    return eval_chunk


def evaluate_first_episodes(arm: Arm, params, key, n_envs=256, chunk=250, max_steps=EPISODE_LIMIT + 1, eval_fn=None):
    """First episode of every env, sampled policy. Returns achievements (n_envs, 22), lengths, censored count.
    Eval steps are not part of the training budget and never feed back into training or selection."""
    eval_fn = eval_fn or make_eval_chunk(arm, chunk)
    k0, k1 = jax.random.split(key)
    obs, env_st = env_reset(jax.random.split(k0, n_envs))
    carry = arm.init_carry(n_envs)
    ach = jnp.zeros((n_envs, NUM_ACHIEVEMENTS), jnp.float32)
    alive = jnp.ones((n_envs,), jnp.bool_)
    length = jnp.zeros((n_envs,), jnp.int32)
    steps = 0
    while steps < max_steps and bool(alive.any()):
        k1, sub = jax.random.split(k1)
        carry, env_st, obs, ach, alive, length = eval_fn(params, carry, env_st, obs, ach, alive, length, sub)
        steps += chunk
    return np.asarray(ach), np.asarray(length), int(np.asarray(alive).sum())


# ----------------------------------------------------------------------------------------------------------------------
# PPO trainer
# ----------------------------------------------------------------------------------------------------------------------
class Trainer:
    def __init__(self, arm: Arm, cfg: PPOConfig):
        self.arm, self.cfg = arm, cfg
        N, T = cfg.num_envs, cfg.num_steps
        if (N * T) % cfg.minibatches or (arm.recurrent and N % cfg.minibatches):
            raise ValueError("num_envs*num_steps (flat) / num_envs (recurrent) must be divisible by minibatches")
        if cfg.adv_norm not in ("minibatch", "batch") or not 0.0 <= cfg.value_norm < 1.0:
            raise ValueError("adv_norm must be 'minibatch' or 'batch' and value_norm an EMA decay in [0, 1)")
        self.steps_per_update = N * T
        self.n_updates = cfg.total_steps // self.steps_per_update
        self.env_steps_total = self.n_updates * self.steps_per_update
        n_opt = max(self.n_updates * cfg.epochs * cfg.minibatches, 1)
        lr = optax.linear_schedule(cfg.lr, 0.0, n_opt) if cfg.anneal_lr else cfg.lr
        if cfg.warmup:
            nw = max(int(cfg.warmup * n_opt), 1)
            decay = lr if callable(lr) else (lambda c, v=lr: v)
            lr = lambda c, decay=decay, nw=nw: decay(c) * jnp.minimum((c + 1) / nw, 1.0)
        self.opt = make_optimizer(arm, lr, cfg.max_grad_norm)
        self.update = jax.jit(self._update)
        self.eval_fn = make_eval_chunk(arm, 250)

    def init(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        params = self.arm.init(k1)
        obs, env_st = env_reset(jax.random.split(k2, self.cfg.num_envs))
        return dict(params=params, opt=self.opt.init(params), env=env_st, obs=obs,
                    carry=self.arm.init_carry(self.cfg.num_envs), ep_ret=jnp.zeros((self.cfg.num_envs,)), key=k3,
                    vn=jnp.zeros((3,)), u=jnp.zeros((), jnp.int32))

    # -- rollout -------------------------------------------------------------------------------------------------------
    def _rollout(self, params, st, key):
        arm, N = self.arm, self.cfg.num_envs

        def body(c, k):
            env_st, obs, carry, ep_ret = c
            k1, k2 = jax.random.split(k)
            logits, value, caux = arm.step(params, carry, obs)
            a = jax.random.categorical(k1, logits)
            logp = jnp.take_along_axis(jax.nn.log_softmax(logits), a[:, None], 1)[:, 0]
            nobs, env_st2, r, d = env_step(jax.random.split(k2, N), env_st, a)
            ep2 = ep_ret + r
            tr = dict(obs=obs, carry=carry, act=a, logp=logp, val=value, rew=r, done=d, nobs=nobs,
                      fin_ret=jnp.where(d, ep2, 0.0))
            return (env_st2, nobs, arm.advance(caux, obs, a, d), jnp.where(d, 0.0, ep2)), tr

        (env_st, obs, carry, ep_ret), traj = jax.lax.scan(
            body, (st["env"], st["obs"], st["carry"], st["ep_ret"]), jax.random.split(key, self.cfg.num_steps))
        _, last_v, _ = arm.step(params, carry, obs)
        return env_st, obs, carry, ep_ret, traj, last_v

    def _gae(self, traj, last_v):
        cfg = self.cfg

        def body(c, x):
            adv_n, v_n = c
            r, d, v = x
            nd = 1.0 - d.astype(jnp.float32)
            adv = r + cfg.gamma * v_n * nd - v + cfg.gamma * cfg.lam * nd * adv_n
            return (adv, v), adv

        _, adv = jax.lax.scan(body, (jnp.zeros_like(last_v), last_v), (traj["rew"], traj["done"], traj["val"]),
                              reverse=True)
        return adv, adv + traj["val"]

    # -- loss ----------------------------------------------------------------------------------------------------------
    def _ppo_terms(self, logits, value, mb, ent_coef=None):
        cfg = self.cfg
        ent_coef = cfg.ent if ent_coef is None else ent_coef
        logp_all = jax.nn.log_softmax(logits)
        logp = jnp.take_along_axis(logp_all, mb["act"][..., None], -1)[..., 0]
        ratio = jnp.exp(logp - mb["logp"])
        adv = mb["adv"] if cfg.adv_norm == "batch" else (mb["adv"] - mb["adv"].mean()) / (mb["adv"].std() + 1e-8)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
        vc = mb["val"] + jnp.clip(value - mb["val"], -cfg.clip, cfg.clip)
        vl = 0.5 * jnp.maximum((value - mb["ret"]) ** 2, (vc - mb["ret"]) ** 2).mean()
        ent = -(jnp.exp(logp_all) * logp_all).sum(-1).mean()
        kl = ((ratio - 1.0) - (logp - mb["logp"])).mean()   # k3 estimate of KL(old || new)
        clipfrac = (jnp.abs(ratio - 1.0) > cfg.clip).mean()
        return pg + cfg.vf * vl - ent_coef * ent, jnp.stack([pg, vl, ent]), jnp.stack([kl, clipfrac])

    def _loss_flat(self, params, mb, ent_coef=None):
        logits, value, wl = self.arm.train_forward(params, mb)
        loss, aux, ex = self._ppo_terms(logits, value, mb, ent_coef)
        return loss + wl, jnp.concatenate([aux, wl[None], ex])

    def _loss_rec(self, params, mb, ent_coef=None):
        tp, evc = self.arm.tp_coef, self.arm.ev_coef

        def body(h, x):
            obs, act, done = x
            if tp or evc:   # auxiliaries: transition prediction [S_{t+1} - S_t, r_t] of the executed action / head features
                logits, value, h2, pred, feat = self.arm.step_aux(params, h, obs, act)
                return self.arm.advance(h2, obs, act, done), (logits, value, pred, feat)
            logits, value, h2 = self.arm.step(params, h, obs)
            return self.arm.advance(h2, obs, act, done), (logits, value)

        _, outs = jax.lax.scan(jax.checkpoint(body) if self.cfg.remat else body, mb["carry"][0],
                               (mb["obs"], mb["act"], mb["done"]))
        logits, value = outs[0], outs[1]
        loss, aux, ex = self._ppo_terms(logits, value, mb, ent_coef)
        flat = {k: mb[k].reshape((-1,) + mb[k].shape[2:]) for k in ("obs", "act", "rew", "nobs", "done")}
        wl = self.arm.aux_loss(params, flat)   # world-model loss of arms that have one (zeros otherwise)
        if tp:
            wl = wl + tp * tp_loss(outs[2], mb["obs"], mb["nobs"], mb["rew"], mb["done"])
        if evc:
            wl = wl + evc * ev_loss(params["pi"]["ev"], outs[3], mb["act"], mb["rew"], mb["done"])
        return loss + wl, jnp.concatenate([aux, wl[None], ex])

    # -- one PPO update ------------------------------------------------------------------------------------------------
    def _update(self, st):
        cfg, arm = self.cfg, self.arm
        key, kr, ku = jax.random.split(st["key"], 3)
        env_st, obs, carry, ep_ret, traj, last_v = self._rollout(st["params"], st, kr)
        vn = st["vn"]
        if cfg.value_norm:   # the critic predicts standardized targets: de-standardize for GAE, re-standardize with the updated stats
            mu, sd = _vn_stats(vn)
            traj, last_v = dict(traj, val=traj["val"] * sd + mu), last_v * sd + mu
        adv, ret = self._gae(traj, last_v)
        if cfg.value_norm:
            vn = _vn_update(vn, ret, cfg.value_norm)
            mu, sd = _vn_stats(vn)
            traj, ret = dict(traj, val=(traj["val"] - mu) / sd), (ret - mu) / sd   # old predictions in the new scale (value clip)
        if cfg.adv_norm == "batch":
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        batch = dict(traj, adv=adv, ret=ret)
        nmb = cfg.minibatches
        ent_coef = None   # constant cfg.ent (unchanged maths) unless an annealing target is set
        if cfg.ent_final >= 0:
            ent_coef = cfg.ent + (cfg.ent_final - cfg.ent) * st["u"].astype(jnp.float32) / max(self.n_updates - 1, 1)
        if arm.recurrent:
            batch["carry"] = traj["carry"][0:1]   # rollout-start GRU state (recurrent arms only use carry[0])
            loss_fn = lambda p, mb: self._loss_rec(p, mb, ent_coef)

            def split(x, perm):
                n = perm.shape[0]
                if x.shape[0] == 1 and x.ndim == 3:            # carry: (1, N, W)
                    return x[:, perm].reshape(1, nmb, n // nmb, -1).transpose(1, 0, 2, 3)
                return x[:, perm].reshape((x.shape[0], nmb, n // nmb) + x.shape[2:]).swapaxes(0, 1)

            size = cfg.num_envs
        else:
            batch = jax.tree_util.tree_map(lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:]), batch)
            loss_fn = lambda p, mb: self._loss_flat(p, mb, ent_coef)
            split = lambda x, perm: x[perm].reshape((nmb, perm.shape[0] // nmb) + x.shape[1:])
            size = self.steps_per_update

        def epoch(carry_, kk):
            perm = jax.random.permutation(kk, size)
            mbs = jax.tree_util.tree_map(lambda x: split(x, perm), batch)

            def mb_step(c, mb):
                p, o = c
                (_, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, mb)
                upd, o = self.opt.update(g, o, p)
                return (optax.apply_updates(p, upd), o), jnp.concatenate([aux, optax.global_norm(g)[None]])

            return jax.lax.scan(mb_step, carry_, mbs)

        (params, opt), aux = jax.lax.scan(epoch, (st["params"], st["opt"]), jax.random.split(ku, cfg.epochs))
        done = traj["done"]
        m = aux.mean((0, 1))   # [pg, vl, ent, wm, kl, clipfrac, grad_norm]; stats keeps the old layout [pg, vl, ent, wm, fin_ret, n_done] + extras
        stats = jnp.concatenate([m[:4], jnp.stack([traj["fin_ret"].sum(), done.sum().astype(jnp.float32)]), m[4:]])
        return dict(params=params, opt=opt, env=env_st, obs=obs, carry=carry, ep_ret=ep_ret, key=key, vn=vn, u=st["u"] + 1), stats

    def train(self, seed: int, log_every: int = 0, log_fn=print):
        st = self.init(jax.random.PRNGKey(seed))
        curve, t0 = [], time.time()
        for u in range(self.n_updates):
            st, s = self.update(st)
            curve.append(s)
            if log_every and (u + 1) % log_every == 0:
                s = np.asarray(s)
                log_fn(f"  update {u + 1}/{self.n_updates} steps={(u + 1) * self.steps_per_update} "
                       f"ep_return={s[4] / max(s[5], 1):.2f} episodes={int(s[5])} ent={s[2]:.3f} wm={s[3]:.3f}")
        curve = np.asarray(jnp.stack(curve)) if curve else np.zeros((0, 9))
        return st["params"], curve, time.time() - t0

    def evaluate(self, params, seed: int, n_envs: int = 256):
        # eval env keys come from a stream disjoint from the training key (seed) -> fresh envs
        key = jax.random.fold_in(jax.random.PRNGKey(seed), 0xE7A1)
        ach, length, censored = evaluate_first_episodes(self.arm, params, key, n_envs, eval_fn=self.eval_fn)
        out = summarize_achievements(ach)
        out.update(eval_episodes=int(n_envs), eval_censored=censored, eval_mean_length=float(length.mean()))
        return out


def curve_summary(curve: np.ndarray, n_points: int = 10) -> list:
    """Training return curve (mean finished-episode return per update, binned)."""
    if len(curve) == 0:
        return []
    bins = np.array_split(np.arange(len(curve)), min(n_points, len(curve)))
    return [float(curve[b, 4].sum() / max(curve[b, 5].sum(), 1)) for b in bins]


def diag_summary(curve: np.ndarray, n_points: int = 10) -> dict:
    """Binned training diagnostics: policy entropy, value loss, approx. KL(old||new), clip fraction, gradient norm (pre-clip)."""
    if len(curve) == 0:
        return {}
    bins = np.array_split(np.arange(len(curve)), min(n_points, len(curve)))
    cols = {"entropy": 2, "value_loss": 1, "approx_kl": 6, "clip_frac": 7, "grad_norm": 8}
    return {k: [float(curve[b, c].mean()) for b in bins] for k, c in cols.items()}


def mean_se(xs) -> tuple:
    xs = np.asarray(xs, dtype=float)
    return float(xs.mean()), float(xs.std(ddof=1) / np.sqrt(len(xs))) if len(xs) > 1 else float("nan")


EVALUATOR_FILES = ("src/environment/craftax_env_adapter.py", "src/pipeline/epa_harness.py")
_ENV_PACKAGES = ("jax", "jaxlib", "jax-cuda12-plugin", "flax", "optax", "craftax", "gymnax", "chex", "numpy")


def _git(*args):
    """git output, or None. `safe.directory` lets it run inside the Docker container, where /workspace has another owner."""
    try:
        return subprocess.check_output(["git", "-c", "safe.directory=*", *args], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def code_sha256(paths=("src",), extra_files=()):
    """Content hash of every .py file under `paths` plus `extra_files`, as imported by this process. Identifies the code
    even when the tree had uncommitted changes, which `git_commit` alone cannot (TASK-20261004-022 audit finding)."""
    h = hashlib.sha256()
    files = sorted({os.path.join(r, f) for p in paths for r, _, fs in os.walk(p) for f in fs if f.endswith(".py")} | set(extra_files))
    for f in files:
        h.update(f.replace(os.sep, "/").encode())
        with open(f, "rb") as fh:
            h.update(hashlib.sha256(fh.read()).digest())
    return h.hexdigest()


def run_provenance(extra_files=()):
    """Code and software identity of a run (roadmap <reporting>: git commit and evaluator commit). Host-side values passed
    by the sweep drivers (GIT_COMMIT, GIT_DIRTY, EVALUATOR_COMMIT) win over what git reports inside a container."""
    from importlib import metadata
    versions = {}
    for pkg in _ENV_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except Exception:
            pass
    dirty_env = os.environ.get("GIT_DIRTY")
    status = _git("status", "--porcelain", "--", "src", "scripts", "tests", "docker", "requirements.txt", "requirements-cuda.txt")
    dirty = (dirty_env == "1") if dirty_env in ("0", "1") else (None if status is None else bool(status))
    dev = jax.devices()[0]
    return dict(git_commit=os.environ.get("GIT_COMMIT") or _git("rev-parse", "HEAD") or "unknown", git_dirty=dirty,
                evaluator_commit=os.environ.get("EVALUATOR_COMMIT") or _git("log", "-1", "--format=%H", "--", *EVALUATOR_FILES) or "unknown",
                code_sha256=code_sha256(extra_files=extra_files), python=platform.python_version(), packages=versions,
                backend=jax.default_backend(), device=getattr(dev, "device_kind", str(dev)), xla_flags=os.environ.get("XLA_FLAGS", ""))


def make_record(arm_name, arm_key, arm_kwargs, cfg_dict, per_seed, role, protocol, env_steps_total, params_total,
                params_deployed, tuning_budget, git_commit, provenance=None):
    """Result-file fields required by roadmap <reporting>. Everything numeric is passed in from executed runs.
    `provenance` (run_provenance()) adds the evaluator commit, a dirty-tree flag, a code hash and package versions."""
    rm, rs = mean_se([r["reward_pct"] for r in per_seed])
    sm, ss = mean_se([r["score_pct"] for r in per_seed])
    rec = dict(phase=1, gate_id=None, protocol=protocol, role=role, arm=arm_name, arm_key=arm_key,
               arm_kwargs=arm_kwargs, config=cfg_dict, git_commit=git_commit,
               evaluator="masked_achievements (post 2026-09-29 auto-reset fix)", observation="symbolic_1345",
               episode_limit=EPISODE_LIMIT, env_steps_total=env_steps_total, env_steps_auxiliary=0,
               seeds=[r["seed"] for r in per_seed], per_seed=per_seed, reward_pct_mean=rm, reward_pct_se=rs,
               score_pct_mean=sm, score_pct_se=ss, params_total=params_total, params_deployed=params_deployed,
               tuning_budget=tuning_budget, policy="sampled, first episode of each eval env, final params")
    if provenance is not None:
        rec.update(evaluator_commit=provenance["evaluator_commit"], provenance=provenance)
    return rec


def write_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
