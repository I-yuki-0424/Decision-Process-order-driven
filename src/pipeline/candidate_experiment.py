"""Fast, fully-jitted convergence experiment for the DPOD candidate models.

Differs from src/pipeline/candidate_benchmark.py (kept untouched) in the
following DOCUMENTED ways (see docs/experiments/2026-09-25_candidate_convergence/README.md):

1. Whole episodes run inside one jitted `lax.scan` (vmapped over B parallel
   envs) instead of a Python loop that retraces every step. The stateful
   candidates' history buffer grows by one row per step (shared_stages.
   compress_step_to_history), which is un-jittable in a scan and O(T) per
   step; here it is truncated to a static sliding window of HIST_WINDOW rows.
2. Training uses an on-policy REINFORCE update once per batch of episodes
   with discounted return-to-go, per-episode-normalised advantages, sampled
   actions and an entropy bonus (`mode="reinforce"`). `mode="argmax_legacy"`
   reproduces the original loss shape (greedy action, reward-weighted
   cross-entropy, no exploration) as the A/B control.
3. Metrics per update (episode return, achievements unlocked) are logged to
   JSONL so convergence can be read off a learning curve; a uniform-random
   policy is evaluated with the identical evaluator as the reference.

All numbers come from real CraftaxEnvAdapter steps and real model forward
passes. No synthetic scores.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.environment.craftax_env_adapter import (
    ACHIEVEMENT_NAMES,
    NUM_ACHIEVEMENTS,
    CraftaxEnvAdapter,
    calculate_crafter_score,
)
from src.model.checkpoint import AsyncCheckpointManager
from src.pipeline.candidate_benchmark import CANDIDATE_REGISTRY

HIST_WINDOW = 16
GAMMA = 0.99


def _init_hist(stateful: bool, d_model: int):
    return jnp.zeros((HIST_WINDOW, d_model)) if stateful else jnp.zeros((1, 1))


def _normalise_logits(logits):
    """A/B fix (mode 'reinforce_normlogits'): MDP-search candidates emit raw Q-values (|x| up to 1e7)
    with -1e9 for masked actions, so softmax is exactly one-hot and log_softmax*0 -> NaN. Replace
    masked entries by (min finite - 10) then standardise to zero-mean / unit-std."""
    masked = logits < -1e8
    lo = jnp.min(jnp.where(masked, jnp.inf, logits))
    lo = jnp.where(jnp.isfinite(lo), lo, 0.0)
    x = jnp.where(masked, lo - 10.0, logits)
    return (x - x.mean()) / (x.std() + 1e-6)


def _make_policy(spec, stateful: bool, norm_logits: bool = False):
    """(params, input_n, hist) -> (logits, new_hist, decision_d) with a static-shape hist."""
    def policy(params, input_n, hist):
        hs = hist if stateful else None
        d, new_hs = spec.step_fn(params, input_n, hs)
        if norm_logits:
            d = d._replace(action_logits=_normalise_logits(d.action_logits))
        new_hist = new_hs[-HIST_WINDOW:] if stateful else hist
        return d, new_hist
    return policy


def build_fns(name: str, adapter: CraftaxEnvAdapter, d_model: int, T: int, lr: float,
              ent_coef: float, mode: str):
    spec = CANDIDATE_REGISTRY[name]
    stateful = spec.stateful
    policy = _make_policy(spec, stateful, norm_logits=mode.endswith("normlogits"))

    def rollout_one(params, key, greedy, random_policy):
        k_reset, k_run = jax.random.split(key)
        input_n, env_state, actions_data = adapter.reset(k_reset)
        hist = _init_hist(stateful, d_model)
        ach0 = jnp.zeros((NUM_ACHIEVEMENTS,))

        def body(carry, t):
            input_n, env_state, hist, alive, ach = carry
            k = jax.random.fold_in(k_run, t)
            k_act, k_env = jax.random.split(k)
            d, new_hist = policy(params, input_n, hist)
            logits = d.action_logits
            sampled = jax.random.categorical(k_act, logits)
            a_greedy = jnp.argmax(logits)
            a_rand = jax.random.randint(k_act, (), 0, adapter.num_actions)
            act = jnp.where(random_policy, a_rand, jnp.where(greedy, a_greedy, sampled)).astype(jnp.int32)
            n_input, n_state, reward, done, _ = adapter.step(
                k_env, env_state, act, actions_data, step_count=t, prev_history=input_n.history)
            ach = jnp.maximum(ach, n_state.achievements.astype(jnp.float32))
            out = dict(input_n=input_n, hist=hist, act=act, reward=reward * alive, mask=alive)
            n_alive = alive * (1.0 - done.astype(jnp.float32))
            return (n_input, n_state, new_hist, n_alive, ach), out

        (_, _, _, _, ach), traj = jax.lax.scan(
            body, (input_n, env_state, hist, jnp.array(1.0), ach0), jnp.arange(T))
        return traj, ach

    rollout_batch = jax.jit(jax.vmap(rollout_one, in_axes=(None, 0, None, None)))

    def rtg(r):
        def f(g, x):
            g = x + GAMMA * g
            return g, g
        _, out = jax.lax.scan(f, 0.0, r, reverse=True)
        return out

    def loss_fn(params, traj):
        # traj leaves: (B, T, ...). Returns/advantages are constants w.r.t. params.
        mask, rew = traj["mask"], traj["reward"]
        G = jax.vmap(rtg)(rew)
        n = jnp.maximum(mask.sum(), 1.0)
        # baseline: mean return-to-go per timestep across the batch (variance reduction, no oracle)
        base = (G * mask).sum(0) / jnp.maximum(mask.sum(0), 1.0)
        adv = (G - base[None, :]) * mask
        adv = adv / (jnp.sqrt((adv ** 2).sum() / n) + 1e-6)
        weight = jax.lax.stop_gradient(G if mode == "argmax_legacy" else adv)

        # rematerialised per-episode loss, sequenced with lax.map to bound activation memory (8 GB GPU)
        @jax.checkpoint
        def ep_loss(x):
            ep, w, m = x

            def per_step(input_n, hist, act):
                d, _ = policy(params, input_n, hist)
                logp = jax.nn.log_softmax(d.action_logits)
                ent = -jnp.sum(jnp.exp(logp) * logp)
                aux = jnp.mean((d.estimated_costs - 1.0) ** 2)
                return logp[act], ent, aux

            logp, ent, aux = jax.vmap(per_step)(ep["input_n"], ep["hist"], ep["act"])
            pg = -(logp * w) * m
            return jnp.stack([pg.sum(), (ent * m).sum(), (aux * m).sum()])

        parts = jax.lax.map(ep_loss, (traj, weight, mask)).sum(0)
        loss = parts[0] / n - ent_coef * parts[1] / n + 0.1 * parts[2] / n
        return loss, parts[1] / n

    # apply_if_finite: skip updates whose grads are NaN/inf (seen in variant_5_1's MDP-search path); documented in README
    opt = optax.apply_if_finite(
        optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate=lr)), max_consecutive_errors=1000)

    @jax.jit
    def update(params, opt_state, traj):
        (loss, ent), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, traj)
        upd, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, loss, ent

    return spec, rollout_batch, update, opt


def evaluate(rollout_batch, params, key, n_eps: int, batch: int, greedy: bool, random_policy: bool = False):
    ach_all, rets, lens = [], [], []
    for i in range(0, n_eps, batch):
        keys = jax.random.split(jax.random.fold_in(key, i), batch)
        traj, ach = rollout_batch(params, keys, jnp.array(greedy), jnp.array(random_policy))
        ach_all.append(np.asarray(ach))
        rets.append(np.asarray(traj["reward"].sum(1)))
        lens.append(np.asarray(traj["mask"].sum(1)))
    ach = np.concatenate(ach_all)[:n_eps]
    rates = [float(np.mean(ach[:, i]) * 100.0) for i in range(NUM_ACHIEVEMENTS)]
    return dict(
        crafter_score=calculate_crafter_score(rates),
        mean_return=float(np.concatenate(rets)[:n_eps].mean()),
        mean_unlocked=float(ach.sum(1).mean()),
        mean_len=float(np.concatenate(lens)[:n_eps].mean()),
        achievement_rates=rates,
    )


def run_experiment(name: str, out_dir: str, updates: int = 200, batch: int = 8, T: int = 200,
                   d_model: int = 32, lr: float = 1e-3, ent_coef: float = 0.01, mode: str = "reinforce",
                   eval_every: int = 25, eval_eps: int = 32, seed: int = 2026, tag: Optional[str] = None,
                   ckpt_every_updates: int = 25, log_fn=print) -> Dict[str, Any]:
    """Trains one candidate for updates*batch episodes (must stay < 50,000)."""
    assert updates * batch < 50000, "episode cap per proposal is 50,000"
    tag = tag or f"{name}__{mode}"
    os.makedirs(out_dir, exist_ok=True)
    adapter = CraftaxEnvAdapter(max_episode_steps=T)
    spec, rollout_batch, update, opt = build_fns(name, adapter, d_model, T, lr, ent_coef, mode)
    key = jax.random.PRNGKey(seed)
    k_init, k_train, k_eval = jax.random.split(key, 3)
    params = spec.init_fn(k_init, d_model=d_model, num_actions=adapter.num_actions,
                          action_feat_dim=adapter.action_feat_dim, num_costs=adapter.num_costs,
                          num_resources=adapter.num_resources, target_dim=8)
    opt_state = opt.init(params)
    n_params = int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
    ckpt = AsyncCheckpointManager(os.path.join(out_dir, "checkpoints", tag), save_every=ckpt_every_updates)

    log_path = os.path.join(out_dir, f"{tag}.train_log.jsonl")
    curve: List[Dict[str, Any]] = []
    evals: List[Dict[str, Any]] = []
    t_start = time.time()
    log_fn(f"[{tag}] params={n_params} backend={jax.default_backend()} updates={updates} batch={batch} T={T}")
    with open(log_path, "w", encoding="utf-8") as lf:
        for u in range(1, updates + 1):
            keys = jax.random.split(jax.random.fold_in(k_train, u), batch)
            traj, ach = rollout_batch(params, keys, jnp.array(False), jnp.array(False))
            params, opt_state, loss, ent = update(params, opt_state, traj)
            rec = dict(update=u, episodes=u * batch,
                       train_return=float(traj["reward"].sum(1).mean()),
                       train_unlocked=float(np.asarray(ach).sum(1).mean()),
                       train_len=float(traj["mask"].sum(1).mean()),
                       loss=float(loss), entropy=float(ent), wall_s=time.time() - t_start)
            rec["skipped_updates"] = int(opt_state.notfinite_count) if hasattr(opt_state, "notfinite_count") else 0
            if not np.isfinite(rec["loss"]):
                rec["nonfinite"] = True
            curve.append(rec)
            lf.write(json.dumps(rec) + "\n"); lf.flush()
            ckpt.maybe_save(params, u, config={"candidate": name, "mode": mode})
            if u % eval_every == 0 or u == updates:
                ev = evaluate(rollout_batch, params, k_eval, eval_eps, batch, greedy=False)
                ev.update(update=u, episodes=u * batch)
                evals.append(ev)
                log_fn(f"[{tag}] upd {u}/{updates} ep={u*batch} train_ret={rec['train_return']:.3f} "
                       f"eval_ret={ev['mean_return']:.3f} unlocked={ev['mean_unlocked']:.2f} "
                       f"crafter={ev['crafter_score']:.3f} H={rec['entropy']:.2f} t={rec['wall_s']:.0f}s")
    ckpt.maybe_save(params, updates, config={"candidate": name, "mode": mode}, force=True)
    ckpt.wait_for_pending(); ckpt.close()

    final_sampled = evaluate(rollout_batch, params, k_eval, 64, batch, greedy=False)
    final_greedy = evaluate(rollout_batch, params, k_eval, 64, batch, greedy=True)
    init_params = spec.init_fn(k_init, d_model=d_model, num_actions=adapter.num_actions,
                               action_feat_dim=adapter.action_feat_dim, num_costs=adapter.num_costs,
                               num_resources=adapter.num_resources, target_dim=8)
    untrained = evaluate(rollout_batch, init_params, k_eval, 64, batch, greedy=False)
    random_ref = evaluate(rollout_batch, params, k_eval, 64, batch, greedy=False, random_policy=True)
    result = dict(
        tag=tag, candidate=name, mode=mode, n_params=n_params, backend=jax.default_backend(),
        devices=str(jax.devices()),
        config=dict(updates=updates, batch=batch, episodes=updates * batch, T=T, d_model=d_model, lr=lr,
                    ent_coef=ent_coef, seed=seed),
        wall_s=time.time() - t_start,
        final_sampled=final_sampled, final_greedy=final_greedy,
        untrained_sampled=untrained, random_policy=random_ref,
        eval_history=evals, achievement_names=ACHIEVEMENT_NAMES,
    )
    with open(os.path.join(out_dir, f"{tag}.result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log_fn(f"[{tag}] DONE wall={result['wall_s']:.0f}s final sampled ret={final_sampled['mean_return']:.3f} "
           f"vs random {random_ref['mean_return']:.3f} vs untrained {untrained['mean_return']:.3f}")
    return result
