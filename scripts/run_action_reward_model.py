"""Train the learned 1-step action-reward model on simulator-branched labels (offline) and report how much of the oracle_act signal it recovers.

Data: random-action trajectories; at every visited state the simulator is branched once per action (same per-timestep key convention as
CraftaxFutureAdapter 'oracle_act') -> label = true 1-step reward of each of the 17 actions. Split by episode. Diagnostics on held-out episodes:
R2 over all 17 outputs; and, over states where the best action has positive reward, the hit rate of argmax(pred) on a best action versus the
uniform-random and 'always the globally best action' baselines.
Outputs: <out>/ar_full.pkl (params), <out>/ar_result.json.
"""
import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, ".")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402

from src.environment.craftax_env_adapter import CraftaxEnvAdapter  # noqa: E402
from src.model.candidates import action_reward_model as arm  # noqa: E402


def gen(n_env, steps, seed):
    a = CraftaxEnvAdapter()
    env, P = a.raw_env, a.raw_env.default_params

    def one(key):
        kr, kw = jax.random.split(key)
        obs, st = env.reset(kr, P)

        def body(c, t):
            obs, st, alive = c
            k = jax.random.fold_in(kw, t)
            ka, ks = jax.random.split(k)
            kk = jax.random.fold_in(jax.random.PRNGKey(11), st.timestep)
            rew = jax.vmap(lambda ac: env.step(kk, st, ac, P)[2])(jnp.arange(arm.N_ACT))
            act = jax.random.randint(ka, (), 0, arm.N_ACT)
            o2, s2, _, d, _ = env.step(ks, st, act, P)
            return (o2, s2, alive * (1 - d.astype(jnp.float32))), (obs, rew, alive)
        _, out = jax.lax.scan(body, (obs, st, jnp.array(1.0)), jnp.arange(steps))
        return out
    return [np.asarray(x) for x in jax.jit(jax.vmap(one))(jax.random.split(jax.random.PRNGKey(seed), n_env))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--train-steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--chunk", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("backend", jax.default_backend(), flush=True)
    t0 = time.time()
    parts = [gen(min(a.chunk, a.envs - c), a.steps, a.seed * 1000 + c) for c in range(0, a.envs, a.chunk)]
    obs, rew, alive = [np.concatenate([p[i] for p in parts]) for i in range(3)]
    print(f"data {obs.shape} gen {time.time()-t0:.0f}s", flush=True)
    ntr = int(a.envs * 0.8)
    fl = lambda x: x.reshape(-1, *x.shape[2:])
    tr = [fl(x[:ntr]) for x in (obs, rew, alive)]
    te = [fl(x[ntr:]) for x in (obs, rew, alive)]
    mtr, mte = tr[2] > 0, te[2] > 0
    tr = [x[mtr] for x in tr]
    te = [x[mte] for x in te]
    print("train states", len(tr[0]), "test states", len(te[0]), "frac states with any nonzero reward",
          float((np.abs(tr[1]).max(1) > 0).mean()), "reward per-action mean", np.round(tr[1].mean(0), 3).tolist(), flush=True)
    params = arm.init_params(jax.random.PRNGKey(a.seed + 1), obs.shape[-1], a.hidden)
    params["mu"], params["sd"] = jnp.asarray(tr[0].mean(0)), jnp.asarray(tr[0].std(0) + 1e-3)
    X, Y = jnp.asarray(tr[0]), jnp.asarray(tr[1])
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))
    st = opt.init(params)

    @jax.jit
    def step(p, st, idx):
        def loss(p):
            return ((jax.vmap(lambda x: arm.predict(p, x))(X[idx]) - Y[idx]) ** 2).mean()
        l, g = jax.value_and_grad(loss)(p)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    rng = np.random.default_rng(a.seed)
    for s in range(1, a.train_steps + 1):
        params, st, l = step(params, st, jnp.asarray(rng.integers(0, len(X), 512)))
        if s % 1000 == 0:
            print(f"step {s} loss {float(l):.5f}", flush=True)
    pred = np.concatenate([np.asarray(jax.vmap(lambda x: arm.predict(params, x))(jnp.asarray(te[0][i:i + 4096])))
                           for i in range(0, len(te[0]), 4096)])
    true = te[1]
    var = float(((true - true.mean(0)) ** 2).mean())
    r2 = 1 - float(((pred - true) ** 2).mean()) / max(var, 1e-12)
    # action-DEPENDENT part only: many rewards are the same for every action (passive vitals changes), which no policy can exploit
    dep = np.where(true.max(1) - true.min(1) > 1e-6)[0]
    tc, pc = true - true.mean(1, keepdims=True), pred - pred.mean(1, keepdims=True)
    r2c = 1 - float(((pc - tc) ** 2).mean()) / max(float((tc ** 2).mean()), 1e-12)
    pos = dep
    best_glob = int(np.argmax(tr[1].mean(0)))  # hit rates below are over ACTION-DEPENDENT test states
    hit = lambda idx_fn: float(np.mean([true[i, idx_fn(i)] >= true[i].max() - 1e-6 for i in pos])) if len(pos) else float("nan")
    res = dict(config=vars(a), n_train=int(len(X)), n_test=int(len(te[0])), test_r2=r2, test_var=var,
               test_r2_action_dependent_part=r2c, frac_test_states_action_dependent=float(len(dep) / len(true)),
               frac_test_states_with_positive_best=float((true.max(1) > 0).mean()),
               hit_rate_pred=hit(lambda i: int(np.argmax(pred[i]))),
               hit_rate_uniform_random=float(np.mean([(true[i] >= true[i].max() - 1e-6).mean() for i in pos])) if len(pos) else float("nan"),
               hit_rate_globally_best_action=hit(lambda i: best_glob),
               per_action_corr=[float(np.corrcoef(pred[:, k], true[:, k])[0, 1]) if true[:, k].std() > 0 else None for k in range(arm.N_ACT)],
               wall_s=time.time() - t0)
    print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=1), flush=True)
    pickle.dump(jax.tree_util.tree_map(np.asarray, params), open(f"{a.out}/ar_full.pkl", "wb"))
    json.dump(res, open(f"{a.out}/ar_result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
