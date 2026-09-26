"""Stage A2: is the passive motion of MOBS (the 'opponent's next move' analogue) predictable from the current observation?

Target: the 7x9x4 mob occupancy channels of the local view at t+h under noop, h in (1,2,4,8), from the full 1345-d observation.
Metrics on held-out episodes, restricted to samples with >=1 mob in view at t (otherwise trivially empty):
  persistence   : mob map at t+h := mob map at t
  mlp           : trained predictor
  Reported: MSE, and the 'movement recall' = fraction of the cells that became occupied at t+h AND were empty at t that the model
  scores above 0.5 (persistence scores 0 by construction), plus precision of those calls.
The player does not move under noop, so the local view frame is fixed; changes are pure mob dynamics (+ spawn/despawn).
"""
import argparse, json, os, sys, time
sys.path.insert(0, ".")
import jax, jax.numpy as jnp, numpy as np, optax
from src.environment.craftax_env_adapter import CraftaxEnvAdapter

HZ = (1, 2, 4, 8)
VIEW, NB, NM = 63, 17, 4
MOB0 = slice(0, 0)  # placeholder (layout below)


def mob_map(obs):  # (1345,) -> (63*4,)  obs[:1323] reshaped (63, 21): blocks 0:17, mobs 17:21
    return obs[:VIEW * (NB + NM)].reshape(VIEW, NB + NM)[:, NB:].reshape(-1)


def gen(n_env, steps, seed):
    a = CraftaxEnvAdapter(); env, P = a.raw_env, a.raw_env.default_params

    def one(key):
        kr, kw = jax.random.split(key); obs, st = env.reset(kr, P)

        def body(c, t):
            obs, st, alive = c
            k = jax.random.fold_in(kw, t); ka, kn, ks = jax.random.split(k, 3)

            def nb(cc, i):
                o, s, al = cc
                o, s2, r, d, _ = env.step(jax.random.fold_in(kn, i), s, 0, P)
                al = al * (1 - d.astype(jnp.float32)); return (o, s2, al), (mob_map(o), al)
            _, (mm, oks) = jax.lax.scan(nb, (obs, st, alive), jnp.arange(HZ[-1]))
            sel = jnp.array([h - 1 for h in HZ])
            act = jax.random.randint(ka, (), 0, 17)
            o2, s2, _, d, _ = env.step(ks, st, act, P)
            return (o2, s2, alive * (1 - d.astype(jnp.float32))), (obs, mm[sel], oks[sel] * alive)
        return jax.lax.scan(body, (obs, st, jnp.array(1.0)), jnp.arange(steps))[1]
    return [np.asarray(x) for x in jax.jit(jax.vmap(one))(jax.random.split(jax.random.PRNGKey(seed), n_env))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--envs", type=int, default=384)
    ap.add_argument("--steps", type=int, default=250); ap.add_argument("--train-steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--chunk", type=int, default=64)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    print("backend", jax.default_backend(), flush=True)
    parts = [gen(min(a.chunk, a.envs - c), a.steps, a.seed * 1000 + c) for c in range(0, a.envs, a.chunk)]
    obs, fut, ok = [np.concatenate([p[i] for p in parts]) for i in range(3)]  # (E,T,1345)(E,T,4,252)(E,T,4)
    ntr = int(a.envs * 0.8); fl = lambda x: x.reshape(-1, *x.shape[2:])
    cur = obs[..., :VIEW * (NB + NM)].reshape(*obs.shape[:2], VIEW, NB + NM)[..., NB:].reshape(*obs.shape[:2], -1)
    has = (cur.sum(-1) > 0) & (ok[..., 0] > 0)                               # >=1 mob in view and alive
    tr = [fl(x[:ntr])[fl(has[:ntr])] for x in (obs, cur, fut, ok)]; te = [fl(x[ntr:])[fl(has[ntr:])] for x in (obs, cur, fut, ok)]
    print("samples with mob in view: train", len(tr[0]), "test", len(te[0]), flush=True)
    tr_j = [jnp.asarray(x) for x in tr]; te_j = [jnp.asarray(x) for x in te]
    mu, sd = tr[0].mean(0), tr[0].std(0) + 1e-3
    k = jax.random.split(jax.random.PRNGKey(a.seed), 3)
    hd = 512; out = len(HZ) * VIEW * NM
    p = dict(w1=jax.random.normal(k[0], (obs.shape[-1], hd)) / np.sqrt(obs.shape[-1]), b1=jnp.zeros(hd),
             w2=jax.random.normal(k[1], (hd, hd)) / np.sqrt(hd), b2=jnp.zeros(hd),
             w3=jax.random.normal(k[2], (hd, out)) * 0.01, b3=jnp.zeros(out))

    def fwd(p, o):  # predicts DELTA logits-free: cur + residual, clipped to [0,1] range by construction of loss/eval
        h = (o - mu) / sd
        h = jax.nn.gelu(h @ p["w1"] + p["b1"]); h = jax.nn.gelu(h @ p["w2"] + p["b2"])
        cur_ = o[..., :VIEW * (NB + NM)].reshape(*o.shape[:-1], VIEW, NB + NM)[..., NB:].reshape(*o.shape[:-1], 1, VIEW * NM)
        return cur_ + (h @ p["w3"] + p["b3"]).reshape(*o.shape[:-1], len(HZ), VIEW * NM)

    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3)); st = opt.init(p)

    @jax.jit
    def step(p, st, idx):
        def lf(p):
            pred = fwd(p, tr_j[0][idx]); m = tr_j[3][idx][..., None]
            return (((pred - tr_j[2][idx]) ** 2) * m).sum() / jnp.maximum(m.sum() * pred.shape[-1], 1)
        l, g = jax.value_and_grad(lf)(p); u, st = opt.update(g, st, p); return optax.apply_updates(p, u), st, l
    rng = np.random.default_rng(a.seed)
    for s in range(1, a.train_steps + 1):
        p, st, l = step(p, st, jnp.asarray(rng.integers(0, len(tr[0]), 512)))
        if s % 1000 == 0: print(f"step {s} loss {float(l):.5f}", flush=True)

    o_, c_, f_, k_ = te_j
    pred = np.asarray(jnp.concatenate([fwd(p, o_[i:i + 2048]) for i in range(0, len(o_), 2048)]))
    f_ = np.asarray(f_); c_ = np.asarray(c_)[:, None, :]; m = np.asarray(k_)
    res = dict(config=vars(a), horizons=list(HZ), n_test=int(len(o_)))
    per = ((c_ - f_) ** 2).mean(-1); mod = ((pred - f_) ** 2).mean(-1)
    res["mse_persistence"] = [float((per[:, h] * m[:, h]).sum() / m[:, h].sum()) for h in range(4)]
    res["mse_model"] = [float((mod[:, h] * m[:, h]).sum() / m[:, h].sum()) for h in range(4)]
    res["skill"] = [1 - a_ / max(b_, 1e-12) for a_, b_ in zip(res["mse_model"], res["mse_persistence"])]
    rec, prec, nnew = [], [], []
    for h in range(4):
        new = (f_[:, h] > 0.5) & (c_[:, 0] < 0.5) & (m[:, h:h + 1] > 0)      # cell newly occupied by a mob at t+h
        call = (pred[:, h] > 0.5) & (c_[:, 0] < 0.5) & (m[:, h:h + 1] > 0)
        rec.append(float((new & call).sum() / max(new.sum(), 1))); prec.append(float((new & call).sum() / max(call.sum(), 1)))
        nnew.append(int(new.sum()))
    res.update(newcell_recall=rec, newcell_precision=prec, n_newcells=nnew)
    json.dump(res, open(f"{a.out}/mobmap_result.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
