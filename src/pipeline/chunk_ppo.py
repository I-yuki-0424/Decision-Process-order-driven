"""Chunk-level PPO on the latency decision process: the learner for the return-vs-latency experiment.

Macro-step MDP (src/environment/latency_envs.py): state (o_c, u_c), action = chunk P_c (k tokens, all executed:
P_c[0:k-delta] this cycle, P_c[k-delta:k] as the next committed prefix), reward R_c = sum_i gamma^i r_{t_c+i},
discount gamma^k between cycles. PPO uses token-level ratios with one chunk-level advantage (as in RLHF-style PPO).

Arms -- ONLY the actor's state input differs; critic, optimiser, budget and architecture are identical:
  ignore     : o_c                         (plans as if the chunk started now)
  augment    : o_c + committed prefix u_c  (information-complete Markov baseline, Katsikopoulos & Engelbrecht 2003)
  wm         : factored-WM forecast of s(t_c + delta) from (o_c, u_c); co-trained on rollout ticks
               (+ optional reference-action pretraining, whose env ticks are charged to the budget)
  wm_passive : forecast with f only (the committed actions' effects are ignored), WM trained ONLY on reference-action
               data -- the original "no-op world model" proposal, tested in the setting where it could matter
  oracle     : exact s(t_c + delta) from the simulator (perfect foresight = zero-latency information; upper bound only)
Critic (training-only, asymmetric): o_c, one-hot u_c, goal, tick/max_ticks for EVERY arm, so learner power is equal
across arms and the time-limit truncation is Markov for the critic.

Design constraints recorded in docs/experiments/2026-09-28_latency_chunking/DESIGN.md:
  * H = k: tokens beyond the commit horizon would never execute, so PPO gives them no signal.
  * Hindsight goal relabelling is used ONLY for the auxiliary temporal-distance head (supervised on behaviour-policy
    time-to-goal), never for the policy gradient: relabelled goals make PPO's on-policy samples off-policy.
"""
import json
import os
import time
from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.environment.latency_envs import CycleOutput, LatencyEnv, make_core_env
from src.model import chunk_policy as cp
from src.model import factored_world_model as fwm
from src.model.checkpoint import AsyncCheckpointManager

ARMS = ("ignore", "augment", "wm", "wm_passive", "oracle")
FORECAST_ARMS = ("wm", "wm_passive", "oracle")


class ChunkPPOConfig(NamedTuple):
    env: str = "intercept"
    env_kwargs: tuple = ()               # hashable (("max_ticks", 200),)
    arm: str = "augment"
    delta: int = 4
    k: int = 8
    hist_len: int = 4
    actor: str = "transformer"           # "transformer" | "mlp" (calibration, k = 1)
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    mlp_width: int = 512
    mlp_layers: int = 2
    critic_width: int = 256
    critic_layers: int = 2
    critic_time_feature: bool = True
    num_envs: int = 64
    cycles_per_update: int = 32
    total_env_ticks: int = 1_000_000
    update_epochs: int = 4
    num_minibatches: int = 4
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    dist_coef: float = 0.1
    dist_max_gap: int = 8
    wm_hidden: int = 256
    wm_window: int = 8
    wm_lr: float = 1e-3
    wm_steps_per_update: int = 4
    wm_batch: int = 256
    wm_freeze_f: bool = False
    passive_envs: int = 0
    passive_ticks_per_env: int = 0
    passive_pretrain_steps: int = 0
    equal_env_ticks: bool = True
    eval_every: int = 10
    eval_cycles: int = 64
    eval_windows: int = 8                # MAX sequential eval_cycles-sized windows; evaluation stops early once every
                                         # env has finished its first episode (see evaluate()'s docstring)
    checkpoint_ticks: int = 100_000
    time_limit_s: float = 0.0            # 0 = none; otherwise checkpoint and stop (ADR-002: 24 h per config)
    seed: int = 0


def craftax_calibration_config(**overrides) -> ChunkPPOConfig:
    """Plain PPO (k = 1, delta = 0, MLP actor) on Craftax-Classic for comparison with published PPO numbers.
    Hyper-parameters verified against github.com/MichaelTMatthews/Craftax_Baselines ppo.py + models/actor_critic.py
    (2026-09-28): tanh MLP, 3 hidden layers x 512 (actor and critic), lr 2e-4 with linear-to-0 anneal, adam eps 1e-5,
    gamma 0.99, GAE lambda 0.8, clip 0.2, ent 0.01, vf 0.5, grad-norm 1.0 (not stated in the paper table, kept as a
    standard PPO default), 4 epochs x 8 minibatches, num_envs 1024, 64 steps/rollout. No published Craftax-CLASSIC
    (vs. full Craftax) PPO number exists at this exact 1e6-tick budget (README only tables Craftax-1B/1M for full
    Craftax; the Craftax-1M row is PPO ~2.2% of max reward). num_envs was corrected from an earlier guess of 64 to
    1024 specifically so the update count/batch-size regime matches that published Craftax-1M setup as closely as
    possible -- pass/fail for --calibrate should be judged against "clearly above a random/no-op policy, no NaNs,
    order-of-magnitude consistent with the ~2.2%-of-max full-Craftax-1M number", not an exact target Craftax-Classic
    reproduces even less well documented.
    eval_cycles left at the default (64, matching cycles_per_update): an earlier version of this preset raised it to
    512 to get a larger evaluation sample, but at num_envs=1024 that made a single eval rollout 8x the size of a
    training rollout, which OOM'd the 8GB local GPU mid-run (found during TASK-20260928-013's first real run,
    update 10's scheduled eval).
    eval_windows=4 (not the default 1): [SUPERSEDED, see below] a follow-up diagnostic (this task, scripts/adhoc/diag_eval_gap.py) found the
    "several hundred eval episodes" reasoning above was wrong in practice -- a single 64-cycle eval window only let
    ~8% of the 1024 envs (not "several hundred") actually finish an episode before the window ended, because eval
    always starts from a synchronised fresh reset (unlike training, whose env state persists and keeps a steady
    stream of completions). That undercounts and biases mean_return toward whichever episodes died fastest.
    eval_windows=4 runs 4 sequential 64-cycle windows (same per-window memory footprint that already fits, just
    more wall time) with env state carried forward between them, giving episodes 4x longer to finish -- confirmed
    to raise both the done-fraction and mean_return substantially on an already-trained checkpoint with no policy
    change, i.e. the earlier single-window number really was an undercount, not a true performance reading.
    UPDATE (TASK-20260928-015, items 8/9): evaluate() no longer averages "whatever finished in a fixed window". It scores the
    FIRST episode of every env (an unbiased per-policy estimate) and keeps stepping, up to eval_windows=16 windows of 64 ticks
    (early exit once all envs finished; the measured trained/untrained MLP policies all finished within 384-640 ticks), and
    reports the censored fraction. Headline eval_return is the STOCHASTIC policy (what PPO optimises); the greedy return is
    logged separately as eval_return_greedy -- on the two trained MLP checkpoints greedy is ~4.6x worse (0.80/0.85 vs
    3.67/3.88), a real behavioural gap, not a window artefact."""
    base = ChunkPPOConfig(env="craftax", arm="ignore", delta=0, k=1, hist_len=0, actor="mlp", mlp_width=512,
                          mlp_layers=3, critic_width=512, critic_layers=3, critic_time_feature=False, num_envs=1024,
                          cycles_per_update=64, lr=2e-4, gae_lambda=0.8, max_grad_norm=1.0, num_minibatches=8,
                          dist_coef=0.0, eval_windows=16)
    return base._replace(**overrides)


class Transition(NamedTuple):
    ai: cp.ActorInput
    ci: jnp.ndarray          # critic input
    chunk: jnp.ndarray       # (k,)
    logp: jnp.ndarray        # (k,)
    value: jnp.ndarray       # ()
    out: CycleOutput
    achieved: jnp.ndarray    # (G,) achieved goal of o_c (hindsight relabelling), (0,) if unused


class MiniBatch(NamedTuple):
    ai: cp.ActorInput
    ci: jnp.ndarray
    chunk: jnp.ndarray
    logp: jnp.ndarray
    value: jnp.ndarray
    adv: jnp.ndarray
    ret: jnp.ndarray
    dist_goal: jnp.ndarray
    dist_target: jnp.ndarray
    dist_valid: jnp.ndarray


def validate_config(cfg: ChunkPPOConfig):
    if cfg.arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    if not (cfg.k >= 1 and 0 <= cfg.delta <= cfg.k):
        raise ValueError("need k >= 1 and 0 <= delta <= k")
    if cfg.actor not in ("transformer", "mlp"):
        raise ValueError("actor must be 'transformer' or 'mlp'")
    if cfg.actor == "mlp" and cfg.k != 1:
        raise ValueError("the MLP actor emits one action; chunks need the autoregressive transformer "
                         "(independent per-step heads mix modes across steps)")
    passive = cfg.passive_envs * cfg.passive_ticks_per_env > 0 and cfg.passive_pretrain_steps > 0
    if cfg.arm == "wm_passive" and not passive:
        raise ValueError("wm_passive needs reference-action data (passive_envs, passive_ticks_per_env, "
                         "passive_pretrain_steps > 0)")
    if cfg.wm_freeze_f and not passive:
        raise ValueError("wm_freeze_f freezes a pretrained f; enable passive pretraining")
    if passive and cfg.passive_ticks_per_env < cfg.wm_window:
        raise ValueError("passive_ticks_per_env must be >= wm_window")
    if cfg.arm == "wm" and cfg.cycles_per_update * cfg.k < cfg.wm_window:
        raise ValueError("cycles_per_update * k must be >= wm_window")
    if (cfg.num_envs * cfg.cycles_per_update) % cfg.num_minibatches:
        raise ValueError("num_envs * cycles_per_update must be divisible by num_minibatches")


class ChunkPPO:
    def __init__(self, cfg: ChunkPPOConfig):
        validate_config(cfg)
        self.cfg = cfg
        core = make_core_env(cfg.env, **dict(cfg.env_kwargs))
        self.core, self.env = core, LatencyEnv(core, cfg.delta, cfg.k, cfg.hist_len)
        self.P = cfg.delta if cfg.arm == "augment" else 0
        self.uses_wm = cfg.arm in ("wm", "wm_passive")
        self.use_effect = cfg.arm == "wm"
        self.use_passive = self.uses_wm and cfg.passive_envs * cfg.passive_ticks_per_env > 0 \
            and cfg.passive_pretrain_steps > 0
        self.use_dist = cfg.actor == "transformer" and core.achieved_goal is not None and cfg.dist_coef > 0
        self.pcfg = cp.ChunkPolicyConfig(core.obs_dim, core.goal_dim, core.num_actions, cfg.k, core.dt,
                                         cfg.d_model, cfg.n_layers, cfg.n_heads)
        if cfg.actor == "transformer":
            self.logits_fn = lambda p, ai, a: cp.forward_chunk_policy(p, self.pcfg, ai, a)
        else:
            self.logits_fn = lambda p, ai, a: cp.forward_mlp_actor(p, ai, a, core.num_actions)
        self.critic_dim = core.obs_dim + cfg.delta * core.num_actions + core.goal_dim + int(cfg.critic_time_feature)
        self.ticks_per_update = cfg.num_envs * cfg.cycles_per_update * cfg.k
        self.passive_ticks = cfg.passive_envs * cfg.passive_ticks_per_env if self.use_passive else 0
        budget = cfg.total_env_ticks - (self.passive_ticks if cfg.equal_env_ticks else 0)
        self.n_updates = max(budget // self.ticks_per_update, 0)
        n_opt = max(self.n_updates * cfg.update_epochs * cfg.num_minibatches, 1)
        lr = optax.linear_schedule(cfg.lr, 0.0, n_opt) if cfg.anneal_lr else cfg.lr
        self.opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(lr, eps=1e-5))
        self.wm_opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(cfg.wm_lr))
        self.collect = jax.jit(self._collect, static_argnames=("n_cycles", "greedy"))
        self.update = jax.jit(self._update)
        self.wm_train = jax.jit(self._wm_train, static_argnames=("n_steps", "train_f"))
        self.streams = jax.jit(self._streams)
        self.collect_passive = jax.jit(self._collect_passive)
        self.forecast_skill = jax.jit(self._forecast_skill)

    # -- parameters ----------------------------------------------------------------------------------------------------
    def init(self, key):
        cfg, core = self.cfg, self.core
        ka, kc, kw = jax.random.split(key, 3)
        if cfg.actor == "transformer":
            actor = cp.init_chunk_policy_parameters(ka, self.pcfg)
        else:
            d_in = cp.mlp_actor_input_dim(core.obs_dim, core.goal_dim, self.P, core.num_actions)
            actor = cp.init_mlp_actor_parameters(ka, d_in, core.num_actions, cfg.mlp_width, cfg.mlp_layers)
        pp = {"actor": actor, "critic": cp.init_critic_parameters(kc, self.critic_dim, cfg.critic_width,
                                                                   cfg.critic_layers)}
        wm = fwm.init_world_model_parameters(kw, core.obs_dim, core.num_actions, cfg.wm_hidden) if self.uses_wm else {}
        return {"pp": pp, "opt": self.opt.init(pp), "wm": wm, "wm_opt": self.wm_opt.init(wm),
                "stats": fwm.identity_stats(core.obs_dim)}

    # -- per-env pieces ------------------------------------------------------------------------------------------------
    def _forecast(self, wm, stats, key, ws, obs_c):
        if self.cfg.arm == "oracle":
            return self.env.oracle_forecast(key, ws)
        if self.uses_wm:
            return fwm.forecast_state(wm, stats, obs_c, ws.prefix, self.core.ref_action, self.core.dt,
                                      self.use_effect)
        return obs_c

    def actor_input(self, ws, obs_c, forecast) -> cp.ActorInput:
        arm = self.cfg.arm
        state = forecast if arm in FORECAST_ARMS else obs_c
        prefix = ws.prefix if arm == "augment" else jnp.zeros((0,), jnp.int32)
        lag = self.cfg.delta * self.core.dt
        state_age = lag if arm == "augment" else 0.0        # forecast arms: state is AT plan start
        obs_age = 0.0 if arm == "ignore" else lag            # ignore: plans as if no latency existed
        return cp.ActorInput(state, self.env.goal(ws), ws.hist_obs, ws.hist_act, ws.hist_valid, prefix,
                             jnp.asarray(state_age, jnp.float32), jnp.asarray(obs_age, jnp.float32))

    def critic_input(self, ws, obs_c):
        parts = [obs_c, jax.nn.one_hot(ws.prefix, self.core.num_actions).reshape(-1), self.env.goal(ws)]
        if self.cfg.critic_time_feature:
            parts.append((ws.tick / self.core.max_ticks)[None].astype(jnp.float32))
        return jnp.concatenate(parts)

    def _env_cycle(self, pp, wm, stats, ws, key, greedy):
        k_cycle, k_act = jax.random.split(key)
        obs_c = self.env.observe(ws)
        ai = self.actor_input(ws, obs_c, self._forecast(wm, stats, k_cycle, ws, obs_c))
        ci = self.critic_input(ws, obs_c)
        chunk, logp = cp.sample_chunk(self.logits_fn, pp["actor"], ai, k_act, self.cfg.k, greedy)
        value = cp.forward_critic(pp["critic"], ci)
        ach = self.core.achieved_goal(obs_c) if self.use_dist else jnp.zeros((0,), jnp.float32)
        ws2, out = self.env.cycle(k_cycle, ws, chunk)
        return ws2, Transition(ai, ci, chunk, logp, value, out, ach)

    # -- rollout -------------------------------------------------------------------------------------------------------
    def reset_envs(self, key):
        return jax.vmap(self.env.reset)(jax.random.split(key, self.cfg.num_envs))

    def _collect(self, pp, wm, stats, ws, key, n_cycles, greedy=False):
        step = jax.vmap(partial(self._env_cycle, greedy=greedy), in_axes=(None, None, None, 0, 0))

        def body(carry, _):
            s, kk = carry
            kk, sub = jax.random.split(kk)
            s2, tr = step(pp, wm, stats, s, jax.random.split(sub, self.cfg.num_envs))
            return (s2, kk), tr

        (ws, key), traj = jax.lax.scan(body, (ws, key), None, length=n_cycles)
        last_v = jax.vmap(lambda s: cp.forward_critic(pp["critic"], self.critic_input(s, self.env.observe(s))))(ws)
        return ws, key, traj, last_v   # traj leaves: (C, N, ...)

    # -- advantages ----------------------------------------------------------------------------------------------------
    def gae(self, traj, last_v):
        cfg = self.cfg
        R = (traj.out.tick_reward * cfg.gamma ** jnp.arange(cfg.k)).sum(-1)
        done = traj.out.done.astype(jnp.float32)
        gk = cfg.gamma ** cfg.k

        def body(carry, x):
            adv_next, v_next = carry
            r, d, v = x
            td = r + gk * v_next * (1.0 - d) - v
            adv = td + gk * cfg.gae_lambda * (1.0 - d) * adv_next
            return (adv, v), adv

        _, adv = jax.lax.scan(body, (jnp.zeros_like(last_v), last_v), (R, done, traj.value), reverse=True)
        return adv, adv + traj.value

    def hindsight(self, key, traj):
        """Relabelled goal = achieved goal of the snapshot j cycles later (same episode). Target = log(1 + ticks from
        the actor's state time to that snapshot). Behaviour-policy time, not optimal distance: a diagnostic/auxiliary."""
        cfg = self.cfg
        ach, ends = traj.achieved, traj.out.done.astype(jnp.int32)
        C = ends.shape[0]
        j = jax.random.randint(key, ends.shape, 1, cfg.dist_max_gap + 1)
        tgt = jnp.arange(C)[:, None] + j
        tgt_c = jnp.clip(tgt, 0, C - 1)
        cum = jnp.cumsum(ends, axis=0)
        ends_through = jnp.take_along_axis(cum, jnp.clip(tgt - 1, 0, C - 1), axis=0)
        valid = (tgt < C) & (ends_through - (cum - ends) == 0)
        goal = jnp.take_along_axis(ach, tgt_c[..., None], axis=0)
        shift = cfg.delta if cfg.arm in FORECAST_ARMS else 0
        return goal, jnp.log1p((j * cfg.k - shift).astype(jnp.float32)), valid

    # -- losses and updates --------------------------------------------------------------------------------------------
    def ppo_loss(self, pp, mb: MiniBatch):
        cfg = self.cfg
        logp, ent = jax.vmap(lambda ai, a: cp.chunk_log_probs(self.logits_fn, pp["actor"], ai, a))(mb.ai, mb.chunk)
        ratio = jnp.exp(logp - mb.logp)
        adv = ((mb.adv - mb.adv.mean()) / (mb.adv.std() + 1e-8))[:, None]
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv).mean()
        v = jax.vmap(lambda x: cp.forward_critic(pp["critic"], x))(mb.ci)
        v_clip = mb.value + jnp.clip(v - mb.value, -cfg.clip_eps, cfg.clip_eps)
        vl = 0.5 * jnp.maximum((v - mb.ret) ** 2, (v_clip - mb.ret) ** 2).mean()
        ent_m = ent.mean()
        loss = pg + cfg.vf_coef * vl - cfg.ent_coef * ent_m
        dl = jnp.array(0.0)
        if self.use_dist:
            pred = jax.vmap(lambda ai: cp.forward_temporal_distance(pp["actor"], self.pcfg, ai))(
                mb.ai._replace(goal=mb.dist_goal))
            w = mb.dist_valid.astype(jnp.float32)
            dl = (w * (pred - mb.dist_target) ** 2).sum() / jnp.maximum(w.sum(), 1.0)
            loss = loss + cfg.dist_coef * dl
        kl = ((ratio - 1.0) - jnp.log(ratio)).mean()
        clipfrac = (jnp.abs(ratio - 1.0) > cfg.clip_eps).mean()
        return loss, jnp.stack([pg, vl, ent_m, dl, kl, clipfrac])

    def make_batch(self, traj, last_v, key):
        adv, ret = self.gae(traj, last_v)
        C, N = adv.shape
        if self.use_dist:
            dg, dtgt, dv = self.hindsight(key, traj)
        else:
            dg = jnp.zeros((C, N, self.core.goal_dim)); dtgt = jnp.zeros((C, N)); dv = jnp.zeros((C, N), jnp.bool_)
        mb = MiniBatch(traj.ai, traj.ci, traj.chunk, traj.logp, traj.value, adv, ret, dg, dtgt, dv)
        return jax.tree_util.tree_map(lambda x: x.reshape((C * N,) + x.shape[2:]), mb)

    def _update(self, pp, opt_state, traj, last_v, key):
        cfg = self.cfg
        kb, kp = jax.random.split(key)
        batch = self.make_batch(traj, last_v, kb)
        B, nmb = batch.adv.shape[0], cfg.num_minibatches

        def epoch(carry, kk):
            perm = jax.random.permutation(kk, B)
            mbs = jax.tree_util.tree_map(lambda x: x[perm].reshape((nmb, B // nmb) + x.shape[1:]), batch)

            def step(c, mb):
                p, o = c
                (_, aux), g = jax.value_and_grad(self.ppo_loss, has_aux=True)(p, mb)
                upd, o = self.opt.update(g, o, p)
                return (optax.apply_updates(p, upd), o), aux

            return jax.lax.scan(step, carry, mbs)

        (pp, opt_state), aux = jax.lax.scan(epoch, (pp, opt_state), jax.random.split(kp, cfg.update_epochs))
        return pp, opt_state, aux.mean((0, 1))

    # -- world model ---------------------------------------------------------------------------------------------------
    def _streams(self, traj):
        o, k, C, N = traj.out, self.cfg.k, traj.out.done.shape[0], traj.out.done.shape[1]
        sw = lambda x: jnp.swapaxes(x, 0, 1).reshape((N, C * k) + x.shape[3:])
        return sw(o.tick_obs[:, :, :k]), sw(o.tick_obs[:, :, 1:]), sw(o.tick_act), sw(o.tick_alive), sw(o.tick_end)

    def _collect_passive(self, key):
        """Reference-action rollouts from the env's own reset distribution (no wider coverage is assumed available)."""
        cfg, core = self.cfg, self.core
        penv = LatencyEnv(core, 0, cfg.passive_ticks_per_env, 0)
        k1, k2 = jax.random.split(key)
        ws = jax.vmap(penv.reset)(jax.random.split(k1, cfg.passive_envs))
        chunk = jnp.full((cfg.passive_ticks_per_env,), core.ref_action, jnp.int32)
        _, out = jax.vmap(penv.cycle, in_axes=(0, 0, None))(jax.random.split(k2, cfg.passive_envs), ws, chunk)
        return out.tick_obs[:, :-1], out.tick_obs[:, 1:], out.tick_act, out.tick_alive, out.tick_end

    def _wm_train(self, wm, wm_os, stats, key, obs_b, nxt, act, alive, end, n_steps, train_f):
        cfg, core = self.cfg, self.core

        def step(carry, kk):
            p, o = carry
            ow, aw, v = fwm.extract_windows(kk, obs_b, nxt, act, alive, end, cfg.wm_window, cfg.wm_batch)
            loss, g = jax.value_and_grad(fwm.world_model_loss)(p, stats, ow, aw, v, core.ref_action, core.dt,
                                                               self.use_effect, train_f)
            upd, o = self.wm_opt.update(g, o, p)
            return (optax.apply_updates(p, upd), o), loss

        (wm, wm_os), losses = jax.lax.scan(step, (wm, wm_os), jax.random.split(key, n_steps))
        return wm, wm_os, losses.mean()

    def _forecast_skill(self, traj):
        """Skill of the actor's state input as a forecast of the true observation at t_c + delta, on the policy's own
        state distribution: 1 - err(input) / err(persistence). 0 for ignore/augment by construction, 1 for oracle."""
        d = self.cfg.delta
        if d == 0:
            return jnp.array(jnp.nan)
        o = traj.out
        true, obs_c, valid = o.tick_obs[:, :, d], o.tick_obs[:, :, 0], o.tick_alive[:, :, d - 1]
        err = lambda p: (((p - true) ** 2).sum(-1) * valid).sum()
        return 1.0 - err(traj.ai.state) / jnp.maximum(err(obs_c), 1e-12)

    # -- evaluation ----------------------------------------------------------------------------------------------------
    def evaluate(self, st, key, greedy: bool = False):
        """Unbiased per-policy return: the FIRST episode of every env, all started together from a fresh reset.

        Why not "mean over the episodes that finished inside the window" (the old estimator and the training metric): with
        synchronised starts only the fastest deaths are counted, and a policy that dies sooner is counted more often
        (TASK-20260928-013/014 diagnostics; TASK-20260928-015: an UNTRAINED policy scored 0.30 / 1.16 with 1 / 4 windows vs a true
        first-episode mean of 1.36). Here every env contributes exactly once, envs are stepped until all have finished or
        `eval_windows` windows of `eval_cycles` cycles have elapsed (peak memory = one window), and the censored fraction is
        reported. Envs that never finish are NOT silently dropped: `mean_return_incl_censored` adds their running return.
        Default is the stochastic policy (what PPO optimises); pass greedy=True for the argmax policy."""
        cfg = self.cfg
        k1, k2 = jax.random.split(key)
        ws = self.reset_envs(k1)
        N = cfg.num_envs
        seen = np.zeros(N, bool)
        ret = np.zeros(N)
        term = np.zeros(N, bool)
        for kk in jax.random.split(k2, cfg.eval_windows):
            ws, _, traj, _ = self.collect(st["pp"], st["wm"], st["stats"], ws, kk,
                                          n_cycles=cfg.eval_cycles, greedy=greedy)
            done = np.asarray(traj.out.done)
            first = done & ((np.cumsum(done, 0) - done) == 0) & ~seen[None, :]   # first completion of a still-unseen env
            fin = first.any(0)
            ret[fin] = (np.asarray(traj.out.finished_return) * first).sum(0)[fin]
            term[fin] = (np.asarray(traj.out.terminated) & first).any(0)[fin]
            seen |= fin
            if seen.all():
                break
        partial = np.asarray(ws.ep_return)[~seen]
        n = int(seen.sum())
        return dict(mean_return=float(ret[seen].mean()) if n else float("nan"), episodes=n,
                    terminal_rate=float(term[seen].mean()) if n else float("nan"),
                    censored_frac=float(1.0 - n / N),
                    mean_return_incl_censored=float((ret[seen].sum() + partial.sum()) / N))

    # -- dry run (wiring check: one rollout + loss/gradient evaluation, NO parameter update) --------------------------
    def dry_run(self, key=None):
        key = jax.random.PRNGKey(self.cfg.seed) if key is None else key
        k1, k2, k3, k4 = jax.random.split(key, 4)
        st = self.init(k1)
        _, _, traj, last_v = self.collect(st["pp"], st["wm"], st["stats"], self.reset_envs(k2), k3,
                                          n_cycles=self.cfg.cycles_per_update)
        batch = self.make_batch(traj, last_v, k4)
        (loss, aux), g = jax.jit(jax.value_and_grad(self.ppo_loss, has_aux=True))(st["pp"], batch)
        gnorm = jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g)))
        info = dict(loss=float(loss), grad_norm=float(gnorm), pg=float(aux[0]), vl=float(aux[1]),
                    entropy=float(aux[2]), dist=float(aux[3]), forecast_skill=float(self.forecast_skill(traj)),
                    chunk_shape=list(traj.chunk.shape), state_shape=list(traj.ai.state.shape),
                    sequential_policy_passes=self.cfg.k, forecast_model_steps=self.cfg.delta if self.uses_wm else 0,
                    n_updates_planned=self.n_updates, ticks_per_update=self.ticks_per_update,
                    passive_ticks=self.passive_ticks)
        if self.uses_wm:
            streams = self.streams(traj)
            ow, aw, v = fwm.extract_windows(k4, *streams, self.cfg.wm_window, self.cfg.wm_batch)
            info["wm_loss"] = float(fwm.world_model_loss(st["wm"], st["stats"], ow, aw, v, self.core.ref_action,
                                                         self.core.dt, self.use_effect))
        return info

    # -- training loop (not executed in this task) ---------------------------------------------------------------------
    def run(self, out_dir: Optional[str] = None, log_fn=print, max_updates: Optional[int] = None):
        cfg, core = self.cfg, self.core
        key = jax.random.PRNGKey(cfg.seed)
        key, k_init, k_env = jax.random.split(key, 3)
        st, start, records = self.init(k_init), 0, []
        ckpt = None
        if out_dir:
            ckpt = AsyncCheckpointManager(os.path.join(out_dir, "checkpoints"),
                                          save_every=max(1, cfg.checkpoint_ticks // self.ticks_per_update))
            resumed = ckpt.resume()
            if resumed is not None:
                start, meta, payload = resumed
                st, key, records = payload["st"], payload["key"], list(meta.get("records", []))
                log_fn(f"resumed at update {start} (env states re-initialised)")
        if start == 0 and self.use_passive:
            key, kp, kt = jax.random.split(key, 3)
            streams = self.collect_passive(kp)
            st["stats"] = fwm.compute_world_model_stats(streams[0], streams[1], streams[3], core.dt)
            st["wm"], st["wm_opt"], l = self.wm_train(st["wm"], st["wm_opt"], st["stats"], kt, *streams,
                                                      n_steps=cfg.passive_pretrain_steps, train_f=True)
            log_fn(f"passive pretraining: {self.passive_ticks} reference-action ticks, loss {float(l):.4f}")
        ws = self.reset_envs(k_env)
        completed = np.zeros(cfg.num_envs, np.int64)   # episodes each env has finished since (re)start; envs are fresh here
        n_updates = self.n_updates if max_updates is None else min(self.n_updates, max_updates)
        t0 = time.time()
        for u in range(start, n_updates):
            key, kc, ku, kw, ke = jax.random.split(key, 5)
            ws, _, traj, last_v = self.collect(st["pp"], st["wm"], st["stats"], ws, kc,
                                               n_cycles=cfg.cycles_per_update)
            wm_loss = float("nan")
            if cfg.arm == "wm":
                streams = self.streams(traj)
                if u == 0 and not self.use_passive:   # statistics fixed from the first rollout the agent saw
                    st["stats"] = fwm.compute_world_model_stats(streams[0], streams[1], streams[3], core.dt)
                st["wm"], st["wm_opt"], l = self.wm_train(st["wm"], st["wm_opt"], st["stats"], kw, *streams,
                                                          n_steps=cfg.wm_steps_per_update,
                                                          train_f=not cfg.wm_freeze_f)
                wm_loss = float(l)
            st["pp"], st["opt"], aux = self.update(st["pp"], st["opt"], traj, last_v, ku)
            aux = np.asarray(aux)
            ep, completed = train_episode_summary(traj, completed)
            rec = dict(update=u + 1, env_ticks=self.passive_ticks + (u + 1) * self.ticks_per_update,
                       train_return=ep["mean_return"], train_episodes=ep["episodes"],
                       train_terminal_rate=ep["terminal_rate"], train_settled=ep["settled"],
                       train_return_all=ep["mean_return_all"], train_episodes_all=ep["episodes_all"],
                       pg=float(aux[0]), vl=float(aux[1]),
                       entropy=float(aux[2]), dist_loss=float(aux[3]), approx_kl=float(aux[4]),
                       clipfrac=float(aux[5]), forecast_skill=float(self.forecast_skill(traj)), wm_loss=wm_loss,
                       wall_s=time.time() - t0)
            if cfg.eval_every and ((u + 1) % cfg.eval_every == 0 or u + 1 == n_updates):
                ev = self.evaluate(st, ke, greedy=False)
                evg = self.evaluate(st, ke, greedy=True)
                rec.update(eval_return=ev["mean_return"], eval_episodes=ev["episodes"],
                           eval_terminal_rate=ev["terminal_rate"], eval_censored_frac=ev["censored_frac"],
                           eval_return_greedy=evg["mean_return"], eval_episodes_greedy=evg["episodes"],
                           eval_censored_frac_greedy=evg["censored_frac"])
            records.append(rec)
            log_fn(" ".join(f"{a}={b:.4g}" if isinstance(b, float) else f"{a}={b}" for a, b in rec.items()))
            over_time = cfg.time_limit_s and time.time() - t0 > cfg.time_limit_s
            if ckpt is not None:
                ckpt.maybe_save({"st": st, "key": key}, u + 1, config={"cfg": cfg._asdict(), "records": records},
                                force=bool(over_time) or u + 1 == n_updates)
            if over_time:
                log_fn(f"time limit reached after update {u + 1}; checkpointed, stopping")
                break
        if ckpt is not None:
            ckpt.close()
        settled = [r["train_return"] for r in records if r.get("train_settled") and r["train_return"] == r["train_return"]]
        summary = dict(cfg=cfg._asdict(), n_updates=n_updates, passive_ticks=self.passive_ticks,
                       ticks_per_update=self.ticks_per_update, backend=jax.default_backend(), metric_protocol=METRIC_PROTOCOL,
                       n_settled_updates=len(settled),
                       train_return_settled_mean=float(np.mean(settled)) if settled else float("nan"), records=records)
        if out_dir:
            with open(os.path.join(out_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=1)
        return summary


METRIC_PROTOCOL = ("v2 (TASK-20260928-015): eval_return = mean return of the FIRST episode of every env (fresh start, stochastic "
                   "policy, censoring reported; eval_return_greedy = argmax policy); train_return = mean over finished episodes "
                   "EXCLUDING each env's first, valid only where train_settled (every env had finished >= 2 episodes before the "
                   "window); train_return_all = old definition (biased low early because all envs start synchronised). "
                   "Files written before this protocol have eval_return = greedy, 1-window and train_return = train_return_all.")


def train_episode_summary(traj, completed_before):
    """Training-rollout episode statistics without the synchronised-start bias.

    All envs start at tick 0 together, so during the first updates only the fastest deaths complete and the plain mean of
    finished returns is biased low (an UNTRAINED policy read 0.39 -> 1.4 over 4 updates; its true mean is 1.36). The bias decays
    as episode start times spread out. Here each env's first completion is excluded from `mean_return`, and `settled` is True
    only once every env had already finished >= 2 episodes before this window (measured: from then the excluded and plain
    means coincide). Use only settled updates for learning curves; `mean_return_all` keeps the old definition.
    Returns (summary dict, completed_after) with completed_after = completed_before + per-env completions in this window."""
    done = np.asarray(traj.out.done)                              # (C, N)
    fr = np.asarray(traj.out.finished_return)
    term = np.asarray(traj.out.terminated)
    before = completed_before[None, :] + np.cumsum(done, 0) - done  # completions of that env strictly before each cycle
    keep = done & (before >= 1)
    n, n_all = int(keep.sum()), int(done.sum())
    out = dict(mean_return=float((fr * keep).sum() / n) if n else float("nan"), episodes=n,
               terminal_rate=float((term & keep).sum() / n) if n else float("nan"),
               mean_return_all=float((fr * done).sum() / n_all) if n_all else float("nan"), episodes_all=n_all,
               settled=bool(completed_before.min() >= 2))
    return out, completed_before + done.sum(0)


def episode_summary(traj):
    done = np.asarray(traj.out.done)
    n = int(done.sum())
    ret = float((np.asarray(traj.out.finished_return) * done).sum() / max(n, 1)) if n else float("nan")
    term = float((np.asarray(traj.out.terminated) & done).sum() / max(n, 1)) if n else float("nan")
    return dict(mean_return=ret, episodes=n, terminal_rate=term)
