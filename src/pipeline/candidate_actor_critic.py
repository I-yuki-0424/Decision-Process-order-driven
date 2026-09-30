"""Actor-critic (PPO + GAE + learned value baseline) learner for the DPOD candidate policies on Craftax
(TASK-20260927-011 / arc.tex sec. 2.8; arc-proposals TASK-20260929-016).

New module; src/pipeline/candidate_experiment.py is untouched. The actor is any registered candidate (default
transformer_branch, exactly as in the REINFORCE Stage-B runs); only the learner changes:

  * REINFORCE baseline (per-timestep batch mean of return-to-go, no state dependence)  ->  learned V(s) critic + GAE(lambda)
  * one on-policy gradient step per batch                                             ->  PPO clipped ratio, several epochs
  * critic = small MLP on the 39-d observation summary + t/T ONLY, identical in every arm (the extra 'anticipation' /
    oracle_act features never reach the critic), so learner power is the same across arms.

Positive-control note (arc.tex sec. 2.5 `oracle_act`): the feature block is the TRUE 1-step reward of each of the 17
actions (answer leakage by design). `build_greedy1_rollout` scores the trivial policy "argmax of that leaked reward"
with the same evaluator so the ceiling of what a 1-step-lookahead signal can give is measured, not assumed.
All returns come from real CraftaxFutureAdapter env steps and real forward passes.
"""
import json
import os
import time
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.environment.craftax_env_adapter import NUM_ACHIEVEMENTS, masked_achievements
from src.environment.craftax_obs_adapter import BASE_DIM
from src.model.checkpoint import AsyncCheckpointManager
from src.pipeline.candidate_benchmark import CANDIDATE_REGISTRY
from src.pipeline.candidate_experiment import _init_hist, _make_policy, evaluate

GAMMA = 0.99


def init_critic(key, in_dim: int, width: int = 128):
    k1, k2, k3 = jax.random.split(key, 3)
    s = lambda k, i, o, g: jax.random.normal(k, (i, o)) * (g / np.sqrt(i))
    return dict(w1=s(k1, in_dim, width, np.sqrt(2.0)), b1=jnp.zeros(width),
                w2=s(k2, width, width, np.sqrt(2.0)), b2=jnp.zeros(width),
                w3=s(k3, width, 1, 1.0), b3=jnp.zeros(1))


def critic_apply(cp, x):
    h = jnp.tanh(x @ cp["w1"] + cp["b1"])
    h = jnp.tanh(h @ cp["w2"] + cp["b2"])
    return (h @ cp["w3"] + cp["b3"])[0]


def critic_input(input_n, t, T):
    return jnp.concatenate([input_n.state.resource_levels[:BASE_DIM], jnp.array([t / T])])


def build_ac_fns(name: str, adapter, d_model: int, T: int, lr: float, ent_coef: float, vf_coef: float,
                 clip_eps: float, gae_lambda: float, epochs: int, minibatches: int, batch: int):
    spec = CANDIDATE_REGISTRY[name]
    stateful = spec.stateful
    policy = _make_policy(spec, stateful)

    def rollout_one(params, key, greedy, random_policy):
        k_reset, k_run = jax.random.split(key)
        input_n, env_state, actions_data = adapter.reset(k_reset)
        hist = _init_hist(stateful, d_model)
        ach0 = jnp.zeros((NUM_ACHIEVEMENTS,))

        def body(carry, t):
            input_n, env_state, hist, alive, ach = carry
            k = jax.random.fold_in(k_run, t)
            k_act, k_env = jax.random.split(k)
            d, new_hist = policy(params["actor"], input_n, hist)
            logits = d.action_logits
            logp_all = jax.nn.log_softmax(logits)
            sampled = jax.random.categorical(k_act, logits)
            a_greedy = jnp.argmax(logits)
            a_rand = jax.random.randint(k_act, (), 0, adapter.num_actions)
            act = jnp.where(random_policy, a_rand, jnp.where(greedy, a_greedy, sampled)).astype(jnp.int32)
            v = critic_apply(params["critic"], critic_input(input_n, t, T))
            n_input, n_state, reward, done, _ = adapter.step(
                k_env, env_state, act, actions_data, step_count=t, prev_history=input_n.history)
            ach = jnp.maximum(ach, masked_achievements(n_state.achievements, done, alive))
            out = dict(input_n=input_n, hist=hist, act=act, reward=reward * alive, mask=alive,
                       logp=logp_all[act], value=v)
            n_alive = alive * (1.0 - done.astype(jnp.float32))
            return (n_input, n_state, new_hist, n_alive, ach), out

        (_, _, _, _, ach), traj = jax.lax.scan(
            body, (input_n, env_state, hist, jnp.array(1.0), ach0), jnp.arange(T))
        return traj, ach

    rollout_batch = jax.jit(jax.vmap(rollout_one, in_axes=(None, 0, None, None)))

    def gae(rew, val, mask):
        # mask_t = 1 while the episode is alive at step t; the step where done fires still has mask 1 and is terminal.
        next_mask = jnp.concatenate([mask[1:], jnp.zeros(1)])
        next_val = jnp.concatenate([val[1:], jnp.zeros(1)]) * next_mask

        def f(carry, x):
            r, v, nv, nm = x
            delta = r + GAMMA * nv - v
            a = delta + GAMMA * gae_lambda * nm * carry
            return a, a
        _, adv = jax.lax.scan(f, 0.0, (rew, val, next_val, next_mask), reverse=True)
        return adv * mask, (adv + val) * mask

    def loss_fn(params, ep_batch):
        """ep_batch: dict of (M, T, ...) leaves for M episodes incl. adv, ret (already computed)."""
        n = jnp.maximum(ep_batch["mask"].sum(), 1.0)

        @jax.checkpoint
        def ep_loss(ep):
            def per_step(input_n, hist, act, t):
                d, _ = policy(params["actor"], input_n, hist)
                logp_all = jax.nn.log_softmax(d.action_logits)
                ent = -jnp.sum(jnp.exp(logp_all) * logp_all)
                v = critic_apply(params["critic"], critic_input(input_n, t, T))
                return logp_all[act], ent, v

            logp, ent, v = jax.vmap(per_step)(ep["input_n"], ep["hist"], ep["act"], jnp.arange(T))
            m = ep["mask"]
            ratio = jnp.exp(logp - ep["logp"])
            adv = ep["adv"]
            pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv) * m
            vl = 0.5 * (v - ep["ret"]) ** 2 * m
            kl = (ep["logp"] - logp) * m
            cf = (jnp.abs(ratio - 1.0) > clip_eps) * m
            return jnp.stack([pg.sum(), vl.sum(), (ent * m).sum(), kl.sum(), cf.sum()])

        parts = jax.lax.map(ep_loss, ep_batch).sum(0) / n
        loss = parts[0] + vf_coef * parts[1] - ent_coef * parts[2]
        return loss, parts

    def make_opt(total_steps):
        sched = optax.linear_schedule(lr, 0.0, total_steps)
        return optax.apply_if_finite(
            optax.chain(optax.clip_by_global_norm(0.5), optax.adamw(learning_rate=sched, eps=1e-5)),
            max_consecutive_errors=1000)

    mb_size = batch // minibatches

    def make_update(opt):
        @jax.jit
        def update(params, opt_state, traj, key):
            adv, ret = jax.vmap(gae)(traj["reward"], traj["value"], traj["mask"])
            mask = traj["mask"]
            nm = jnp.maximum(mask.sum(), 1.0)
            mean = (adv * mask).sum() / nm
            std = jnp.sqrt((((adv - mean) * mask) ** 2).sum() / nm)
            adv_n = (adv - mean) / (std + 1e-8) * mask
            data = {k: traj[k] for k in ("input_n", "hist", "act", "mask", "logp")}
            data["adv"], data["ret"] = adv_n, ret

            def epoch(carry, ekey):
                params, opt_state = carry
                perm = jax.random.permutation(ekey, batch)

                def mb(c, idx):
                    p, os_ = c
                    sub = jax.tree_util.tree_map(lambda x: x[idx], data)
                    (loss, parts), g = jax.value_and_grad(loss_fn, has_aux=True)(p, sub)
                    upd, os_ = opt.update(g, os_, p)
                    return (optax.apply_updates(p, upd), os_), parts

                idxs = perm[: minibatches * mb_size].reshape(minibatches, mb_size)
                (params, opt_state), parts = jax.lax.scan(mb, (params, opt_state), idxs)
                return (params, opt_state), parts.mean(0)

            (params, opt_state), parts = jax.lax.scan(epoch, (params, opt_state), jax.random.split(key, epochs))
            return params, opt_state, parts[-1]
        return update

    return spec, rollout_batch, make_opt, make_update


def build_greedy1_rollout(adapter, T: int):
    """Reference policy: argmax of the leaked 1-step reward block (oracle_act extras[:17]); ties broken uniformly.
    Requires an adapter with feature_source='oracle_act'. Same signature/outputs as the learner's rollout_batch."""
    n = adapter.num_actions

    def rollout_one(params, key, greedy, random_policy):
        k_reset, k_run = jax.random.split(key)
        input_n, env_state, actions_data = adapter.reset(k_reset)
        ach0 = jnp.zeros((NUM_ACHIEVEMENTS,))

        def body(carry, t):
            input_n, env_state, alive, ach = carry
            k = jax.random.fold_in(k_run, t)
            k_act, k_env = jax.random.split(k)
            rew_block = input_n.state.resource_levels[BASE_DIM:BASE_DIM + n]
            noise = jax.random.uniform(k_act, (n,)) * 1e-6
            act = jnp.argmax(rew_block + noise).astype(jnp.int32)
            n_input, n_state, reward, done, _ = adapter.step(
                k_env, env_state, act, actions_data, step_count=t, prev_history=input_n.history)
            ach = jnp.maximum(ach, masked_achievements(n_state.achievements, done, alive))
            out = dict(reward=reward * alive, mask=alive)
            return (n_input, n_state, alive * (1.0 - done.astype(jnp.float32)), ach), out

        (_, _, _, ach), traj = jax.lax.scan(body, (input_n, env_state, jnp.array(1.0), ach0), jnp.arange(T))
        return traj, ach

    return jax.jit(jax.vmap(rollout_one, in_axes=(None, 0, None, None)))


def run_ac_experiment(name: str, out_dir: str, adapter, tag: str, updates: int = 200, batch: int = 64, T: int = 250,
                      d_model: int = 256, lr: float = 3e-4, ent_coef: float = 0.01, vf_coef: float = 0.5,
                      clip_eps: float = 0.2, gae_lambda: float = 0.95, epochs: int = 4, minibatches: int = 4,
                      eval_every: int = 25, eval_eps: int = 64, seed: int = 0, ckpt_every_updates: int = 100,
                      log_fn=print) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    spec, rollout_batch, make_opt, make_update = build_ac_fns(
        name, adapter, d_model, T, lr, ent_coef, vf_coef, clip_eps, gae_lambda, epochs, minibatches, batch)
    opt = make_opt(updates * epochs * minibatches)
    update = make_update(opt)
    key = jax.random.PRNGKey(seed)
    k_init, k_crit, k_train, k_eval = jax.random.split(key, 4)
    actor = spec.init_fn(k_init, d_model=d_model, num_actions=adapter.num_actions,
                         action_feat_dim=adapter.action_feat_dim, num_costs=adapter.num_costs,
                         num_resources=adapter.num_resources, target_dim=8)
    params = dict(actor=actor, critic=init_critic(k_crit, BASE_DIM + 1))
    opt_state = opt.init(params)
    n_actor = int(sum(x.size for x in jax.tree_util.tree_leaves(actor)))
    ckpt = AsyncCheckpointManager(os.path.join(out_dir, "checkpoints", tag), save_every=ckpt_every_updates)
    log_path = os.path.join(out_dir, f"{tag}.train_log.jsonl")
    evals: List[Dict[str, Any]] = []
    t0 = time.time()
    log_fn(f"[{tag}] actor_params={n_actor} backend={jax.default_backend()} updates={updates} batch={batch} T={T}")
    with open(log_path, "w", encoding="utf-8") as lf:
        for u in range(1, updates + 1):
            keys = jax.random.split(jax.random.fold_in(k_train, u), batch)
            traj, ach = rollout_batch(params, keys, jnp.array(False), jnp.array(False))
            params, opt_state, parts = update(params, opt_state, traj, jax.random.fold_in(k_train, 10_000_000 + u))
            parts = np.asarray(parts)
            rec = dict(update=u, episodes=u * batch, train_return=float(traj["reward"].sum(1).mean()),
                       train_unlocked=float(np.asarray(ach).sum(1).mean()), train_len=float(traj["mask"].sum(1).mean()),
                       pg=float(parts[0]), vl=float(parts[1]), entropy=float(parts[2]), approx_kl=float(parts[3]),
                       clipfrac=float(parts[4]), wall_s=time.time() - t0)
            lf.write(json.dumps(rec) + "\n"); lf.flush()
            ckpt.maybe_save(params, u, config={"candidate": name, "learner": "actor_critic"})
            if u % eval_every == 0 or u == updates:
                ev = evaluate(rollout_batch, params, k_eval, eval_eps, batch, greedy=False)
                ev.update(update=u, episodes=u * batch)
                evals.append(ev)
                log_fn(f"[{tag}] upd {u}/{updates} ep={u*batch} train_ret={rec['train_return']:.3f} "
                       f"eval_ret={ev['mean_return']:.3f} unlocked={ev['mean_unlocked']:.2f} "
                       f"crafter={ev['crafter_score']:.3f} H={rec['entropy']:.2f} vl={rec['vl']:.3f} t={rec['wall_s']:.0f}s")
    ckpt.maybe_save(params, updates, config={"candidate": name, "learner": "actor_critic"}, force=True)
    ckpt.wait_for_pending(); ckpt.close()
    final_sampled = evaluate(rollout_batch, params, k_eval, 256, batch, greedy=False)
    final_greedy = evaluate(rollout_batch, params, k_eval, 256, batch, greedy=True)
    random_ref = evaluate(rollout_batch, params, k_eval, 256, batch, greedy=False, random_policy=True)
    result = dict(tag=tag, candidate=name, learner="actor_critic", n_actor_params=n_actor, backend=jax.default_backend(),
                  config=dict(updates=updates, batch=batch, episodes=updates * batch, T=T, d_model=d_model, lr=lr,
                              ent_coef=ent_coef, vf_coef=vf_coef, clip_eps=clip_eps, gae_lambda=gae_lambda,
                              epochs=epochs, minibatches=minibatches, seed=seed),
                  wall_s=time.time() - t0, final_sampled=final_sampled, final_greedy=final_greedy,
                  random_policy=random_ref, eval_history=evals)
    with open(os.path.join(out_dir, f"{tag}.result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log_fn(f"[{tag}] DONE wall={result['wall_s']:.0f}s final sampled ret={final_sampled['mean_return']:.3f} "
           f"crafter={final_sampled['crafter_score']:.3f} (random {random_ref['mean_return']:.3f})")
    return result
