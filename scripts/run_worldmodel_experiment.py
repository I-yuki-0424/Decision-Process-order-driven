"""Stage A of the Idea-6 study: learn Craftax noop ("do nothing") vitals dynamics.

Data: random-policy trajectories in the real env; at each visited state the simulator is branched with HORIZON
noop steps to obtain the true passive future (offline data only). Held-out split is by environment/episode.
Models: persistence / mlp / hn (see src/model/candidates/passive_world_model.py). Metric: multi-step MSE in
standardised vitals units and skill vs persistence (1 - MSE/MSE_persistence), on held-out episodes.
"""
import argparse, json, os, pickle, sys, time
sys.path.insert(0, ".")
import jax, jax.numpy as jnp, numpy as np, optax
from src.environment.craftax_env_adapter import CraftaxEnvAdapter
from src.environment.craftax_obs_adapter import obs_to_features, BASE_DIM
from src.model.candidates import passive_world_model as pwm

H = pwm.HORIZON


def gen_data(n_env, steps, seed):
    a = CraftaxEnvAdapter(); env, P = a.raw_env, a.raw_env.default_params

    def vit(s):
        return jnp.array([s.player_health, s.player_food, s.player_drink, s.player_energy], jnp.float32)

    def one(key):
        kr, kw = jax.random.split(key)
        obs, st = env.reset(kr, P)

        def body(c, t):
            obs, st = c
            k = jax.random.fold_in(kw, t); ka, kn, ks = jax.random.split(k, 3)
            feats = obs_to_features(obs)

            def nb(cc, i):
                o, s, alive = cc
                o, s2, r, d, _ = env.step(jax.random.fold_in(kn, i), s, 0, P)
                return (o, s2, alive * (1 - d.astype(jnp.float32))), (vit(s2), alive * (1 - d.astype(jnp.float32)))
            _, (fut, ok) = jax.lax.scan(nb, (obs, st, jnp.array(1.0)), jnp.arange(H))
            act = jax.random.randint(ka, (), 0, 17)
            o2, s2, _, d, _ = env.step(ks, st, act, P)
            return (o2, s2), (feats, vit(st), fut, ok)
        _, out = jax.lax.scan(body, (obs, st), jnp.arange(steps))
        return out
    f = jax.jit(jax.vmap(one))
    return [np.asarray(x) for x in f(jax.random.split(jax.random.PRNGKey(seed), n_env))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=200); ap.add_argument("--train-steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    print("backend", jax.default_backend(), flush=True)
    t0 = time.time()
    feats, vit, fut, ok = gen_data(a.envs, a.steps, a.seed)  # (E,T,F) (E,T,4) (E,T,H,4) (E,T,H)
    print(f"data {feats.shape} gen {time.time()-t0:.0f}s", flush=True)
    E = a.envs; ntr = int(E * 0.8)
    fl = lambda x: x.reshape(-1, *x.shape[2:])
    tr = [fl(x[:ntr]) for x in (feats, vit, fut, ok)]; te = [fl(x[ntr:]) for x in (feats, vit, fut, ok)]
    mu, sd = tr[0].mean(0), tr[0].std(0) + 1e-3
    # obs intrinsics are the raw vitals times a constant scale; recover it from data (least squares) so targets
    # live in the same units as the encoder's q. Fail loudly if the relation is not a clean per-dim scale.
    ratio = (tr[0][:, :4] * tr[1]).sum(0) / (tr[1] ** 2).sum(0)
    resid = np.abs(tr[0][:, :4] - tr[1] * ratio).max()
    print("intrinsics/vitals scale", ratio, "max resid", resid, flush=True)
    assert resid < 1e-3, "intrinsics are not a pure rescaling of the vitals"
    vm, vs = mu[:4], sd[:4]
    # passive-change statistics (how much vitals actually change under noop) -- for the write-up
    chg = np.abs((tr[2][:, -1] - tr[1][:, None, :][:, 0]) ) 
    stats = dict(frac_vitals_changing_h8=float((chg.max(-1) > 1e-6).mean()), n_train=int(len(tr[0])), n_test=int(len(te[0])))
    print("passive stats", stats, flush=True)
    tr_j = [jnp.asarray(x) for x in tr]; te_j = [jnp.asarray(x) for x in te]
    norm = lambda v: (v * jnp.asarray(ratio) - vm) / vs  # raw vitals -> feature units -> standardised

    def loss_fn(params, kind, f, v, fu, m):
        pred = jax.vmap(lambda x: pwm.predict(params, kind, x))(f)          # (B,H,4)
        err = ((pred - norm(fu)) ** 2).mean(-1) * m
        return err.sum() / jnp.maximum(m.sum(), 1.0)

    def evaluate(params, kind, split):
        f, v, fu, m = split
        outs = []
        for i in range(0, len(f), 2048):
            fi, fui, mi = f[i:i+2048], fu[i:i+2048], m[i:i+2048]
            if kind == "persist":
                pred = jnp.tile(norm(v[i:i+2048])[:, None, :], (1, H, 1))
            else:
                pred = jax.vmap(lambda x: pwm.predict(params, kind, x))(fi)
            outs.append((((pred - norm(fui)) ** 2).mean(-1) * mi, mi))
        se = jnp.concatenate([o[0] for o in outs]); mm = jnp.concatenate([o[1] for o in outs])
        return np.asarray(se.sum(0) / mm.sum(0))                              # (H,)

    res = dict(stats=stats, horizon=H, config=vars(a))
    res["persistence"] = evaluate(None, "persist", te_j).tolist()
    for kind in ("mlp", "hn"):
        params = pwm.init_params(jax.random.PRNGKey(a.seed + 1), kind, BASE_DIM)
        params["enc"]["mu"], params["enc"]["sd"] = jnp.asarray(mu), jnp.asarray(sd)
        n_par = int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3)); st = opt.init(params)

        @jax.jit
        def step(params, st, idx):
            l, g = jax.value_and_grad(loss_fn)(params, kind, tr_j[0][idx], tr_j[1][idx], tr_j[2][idx], tr_j[3][idx])
            u, st = opt.update(g, st, params); return optax.apply_updates(params, u), st, l
        rng = np.random.default_rng(a.seed); curve = []; t1 = time.time()
        for s in range(1, a.train_steps + 1):
            idx = jnp.asarray(rng.integers(0, len(tr[0]), 512))
            params, st, l = step(params, st, idx)
            if s % 250 == 0:
                te_mse = evaluate(params, kind, te_j)
                curve.append(dict(step=s, train_loss=float(l), test_mse_mean=float(te_mse.mean())))
                print(f"[{kind}] step {s} train {float(l):.4f} test(mean over h) {te_mse.mean():.4f} t={time.time()-t1:.0f}s", flush=True)
        res[kind] = evaluate(params, kind, te_j).tolist(); res[kind + "_params"] = n_par; res[kind + "_curve"] = curve
        pickle.dump(jax.tree_util.tree_map(np.asarray, params), open(f"{a.out}/wm_{kind}.pkl", "wb"))
    per = np.array(res["persistence"])
    for k in ("mlp", "hn"):
        res[k + "_skill_vs_persistence"] = (1 - np.array(res[k]) / per).tolist()
    json.dump(res, open(f"{a.out}/worldmodel_result.json", "w"), indent=2)
    print(json.dumps({k: res[k] for k in ("persistence", "mlp", "hn", "mlp_skill_vs_persistence", "hn_skill_vs_persistence")}, indent=1))


if __name__ == "__main__":
    main()
