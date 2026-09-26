"""Stage A (v2): learn the passive (noop) future of the WHOLE observation summary, decoupled from any policy.

Data: trajectories from a chosen behaviour policy; at every visited state the simulator is branched with 8 noop steps and the
39-dim summary is recorded at horizons 1,2,4,8 (offline labels only). Split by episode.
Arms (all evaluated on held-out episodes, per feature group, as skill vs persistence = 1 - MSE / MSE(delta=0)):
  mean      : constant mean delta (no input)
  mlp_sum   : MLP on the 39-d summary
  mlp_full  : MLP on the full 1345-d symbolic observation
Behaviour: random actions, with probability --p-noop replaced by noop.
Outputs: wm_<arm>.pkl (params incl. delta std), delta_stats.pkl, stageA_result.json
"""
import argparse, json, os, pickle, sys, time
sys.path.insert(0, ".")
import jax, jax.numpy as jnp, numpy as np, optax
from src.environment.craftax_env_adapter import CraftaxEnvAdapter
from src.environment.craftax_obs_adapter import obs_to_features
from src.model.candidates import passive_wm_full as pw

HZ = pw.HORIZONS
GROUPS = dict(vitals=(0, 4), inventory=(4, 16), light_sleep=(16, 18), blocks=(18, 35), mobs=(35, 39))


def gen_data(n_env, steps, seed, p_noop):
    a = CraftaxEnvAdapter(); env, P = a.raw_env, a.raw_env.default_params

    def one(key):
        kr, kw = jax.random.split(key)
        obs, st = env.reset(kr, P)

        def body(c, t):
            obs, st, alive = c
            k = jax.random.fold_in(kw, t); ka, kn, ks, kb = jax.random.split(k, 4)
            f0 = obs_to_features(obs)

            def nb(cc, i):
                o, s, al = cc
                o, s2, r, d, _ = env.step(jax.random.fold_in(kn, i), s, 0, P)
                al = al * (1 - d.astype(jnp.float32))
                return (o, s2, al), (obs_to_features(o), al)
            _, (fs, oks) = jax.lax.scan(nb, (obs, st, alive), jnp.arange(HZ[-1]))
            sel = jnp.array([h - 1 for h in HZ])
            act = jnp.where(jax.random.uniform(kb) < p_noop, 0, jax.random.randint(ka, (), 0, 17))
            o2, s2, _, d, _ = env.step(ks, st, act, P)
            return (o2, s2, alive * (1 - d.astype(jnp.float32))), (obs, f0, fs[sel], oks[sel] * alive)
        _, out = jax.lax.scan(body, (obs, st, jnp.array(1.0)), jnp.arange(steps))
        return out
    return [np.asarray(x) for x in jax.jit(jax.vmap(one))(jax.random.split(jax.random.PRNGKey(seed), n_env))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=250); ap.add_argument("--train-steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--p-noop", type=float, default=0.0)
    ap.add_argument("--hidden", type=int, default=512); ap.add_argument("--chunk", type=int, default=64)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    print("backend", jax.default_backend(), flush=True)
    t0 = time.time(); parts = []
    for c in range(0, a.envs, a.chunk):  # chunked to stay inside 8 GB
        parts.append(gen_data(min(a.chunk, a.envs - c), a.steps, a.seed * 1000 + c, a.p_noop))
        print(f"  gen chunk {c}", flush=True)
    obs, f0, fut, ok = [np.concatenate([p[i] for p in parts]) for i in range(4)]  # (E,T,1345)(E,T,39)(E,T,4,39)(E,T,4)
    print(f"data {obs.shape} gen {time.time()-t0:.0f}s", flush=True)
    E = a.envs; ntr = int(E * 0.8); fl = lambda x: x.reshape(-1, *x.shape[2:])
    tr = [fl(x[:ntr]) for x in (obs, f0, fut, ok)]; te = [fl(x[ntr:]) for x in (obs, f0, fut, ok)]
    delta_tr = tr[2] - tr[1][:, None, :]                                   # (N,4,39)
    w = tr[3][..., None]
    dsd = np.sqrt(((delta_tr ** 2) * w).sum(0) / np.maximum(w.sum(0), 1)) + 1e-3   # rms delta = persistence error scale
    dmean = ((delta_tr * w).sum(0) / np.maximum(w.sum(0), 1))
    print("alive-fraction per horizon", tr[3].mean(0), flush=True)
    to_j = lambda s: [jnp.asarray(x) for x in s]
    tr_j, te_j = to_j(tr), to_j(te); dsd_j = jnp.asarray(dsd); dmean_j = jnp.asarray(dmean)

    def target(f0_, fut_): return (fut_ - f0_[:, None, :]) / dsd_j       # standardised delta
    def gsl(x, g): lo, hi = GROUPS[g]; return x[..., lo:hi]

    res = dict(config=vars(a), n_train=int(len(tr[0])), n_test=int(len(te[0])), horizons=list(HZ))

    def evaluate(fn, split):
        obs_, f0_, fut_, ok_ = split
        tot = {g: np.zeros(4) for g in GROUPS}; n = np.zeros(4)
        for i in range(0, len(obs_), 4096):
            sl = slice(i, i + 4096); tg = target(f0_[sl], fut_[sl]); m = ok_[sl]
            pred = fn(obs_[sl], f0_[sl])
            for g in GROUPS:
                e = ((gsl(pred, g) - gsl(tg, g)) ** 2).mean(-1) * m
                tot[g] += np.asarray(e.sum(0))
            n += np.asarray(m.sum(0))
        return {g: (tot[g] / np.maximum(n, 1)).tolist() for g in GROUPS}

    res["persistence"] = evaluate(lambda o, f: jnp.zeros((o.shape[0], len(HZ), pw.F)), te_j)
    mean_pred = dmean_j / dsd_j
    res["mean"] = evaluate(lambda o, f: jnp.tile(mean_pred[None], (o.shape[0], 1, 1)), te_j)

    for arm, in_dim in (("mlp_sum", pw.F), ("mlp_full", obs.shape[-1])):
        params = pw.init_params(jax.random.PRNGKey(a.seed + 1), in_dim, a.hidden)
        X = tr[1] if arm == "mlp_sum" else tr[0]
        params["mu"], params["sd"] = jnp.asarray(X.mean(0)), jnp.asarray(X.std(0) + 1e-3); params["dsd"] = dsd_j
        n_par = int(sum(v.size for k, v in params.items() if k not in ("mu", "sd", "dsd")))
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3)); st = opt.init(params)
        sel = (lambda o, f: f) if arm == "mlp_sum" else (lambda o, f: o)

        def loss_fn(p, x, tg, m):
            pred = jax.vmap(lambda xi: pw.predict_std_delta(p, xi))(x)
            return (((pred - tg) ** 2).mean(-1) * m).sum() / jnp.maximum(m.sum(), 1)

        @jax.jit
        def step(p, st, idx, sel=sel):
            x = sel(tr_j[0][idx], tr_j[1][idx]); tg = target(tr_j[1][idx], tr_j[2][idx])
            l, g = jax.value_and_grad(loss_fn)(p, x, tg, tr_j[3][idx])
            u, st = opt.update(g, st, p); return optax.apply_updates(p, u), st, l
        rng = np.random.default_rng(a.seed); t1 = time.time()
        for s in range(1, a.train_steps + 1):
            params, st, l = step(params, st, jnp.asarray(rng.integers(0, len(tr[0]), 512)))
            if s % 1000 == 0: print(f"[{arm}] step {s} loss {float(l):.4f} t={time.time()-t1:.0f}s", flush=True)
        fn = lambda o, f, p=params, sel=sel: jax.vmap(lambda xi: pw.predict_std_delta(p, xi))(sel(o, f))
        res[arm] = evaluate(fn, te_j); res[arm + "_params"] = n_par
        pickle.dump(jax.tree_util.tree_map(np.asarray, params), open(f"{a.out}/wm_{arm}.pkl", "wb"))
    pickle.dump(dict(dsd=dsd), open(f"{a.out}/delta_stats.pkl", "wb"))
    for arm in ("mean", "mlp_sum", "mlp_full"):
        res["skill_" + arm] = {g: [1 - res[arm][g][h] / max(res["persistence"][g][h], 1e-12) for h in range(4)] for g in GROUPS}
    json.dump(res, open(f"{a.out}/stageA_result.json", "w"), indent=1)
    print("SKILL vs persistence (columns: horizons", HZ, ")")
    for arm in ("mean", "mlp_sum", "mlp_full"):
        for g in GROUPS: print(f"{arm:9s} {g:12s}", np.round(res['skill_' + arm][g], 3))


if __name__ == "__main__":
    main()
