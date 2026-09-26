"""Does PRE-TRAINING the passive ("do nothing", u=0) dynamics, decoupled from the action-effect, improve planning with few action-labelled data?

Operator hypothesis (2026-09-27): a very accurate "if nothing is done" model makes action selection easy; train it beforehand and
independently. Testable form on a system where it is well posed (torque-limited pendulum swing-up, same task/planner as
scripts/run_wm_planning.py: CEM MPC, horizon 60, model used for planning only, env = true dynamics):

  passive data  : trajectories with u=0 (plentiful, cheap: no action needed)   -> N_P trajectories
  active data   : trajectories with random torque (scarce)                     -> N_A trajectories
  ic 'narrow'   : passive ICs from the same region as active data (|theta0|<1.5)  (adds mass, no new coverage)
  ic 'wide'     : passive ICs from the whole circle (theta0 in [-pi,pi])         (needs the ability to release the system anywhere;
                  NOT available for most real robots -- reported separately)

Arms (all integrate with RK4, multi-step window loss):
  mlp_joint     : dx = MLP(x,u), active data only                          [monolithic baseline]
  mlp_all       : dx = MLP(x,u), passive+active pooled                      [does pooling alone give the gain?]
  fact_joint    : dx = f(x) + g(x) u, passive+active pooled, jointly trained [structure only]
  fact_pre      : same structure; f pretrained on passive ONLY then FROZEN; g trained on active only [operator's decoupled scheme]
  hn_joint      : port-Hamiltonian (known kinetic, learned V, R, const G), active only  [existing 'hn' arm]
  hn_pre        : port-Hamiltonian; V,R pretrained on passive only, frozen; only G (2 numbers) trained on active data
Structure caveat (by design, stated in the write-up): the true system IS control-affine with constant G, so factored arms have a
correct inductive bias here; this is a test of the decoupling idea on a case where it is expected to work, not a general proof.
References: true-model MPC, zero-torque.
"""
import argparse, json, os, sys, time
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import jax, jax.numpy as jnp, numpy as np, optax
import run_wm_planning as W

DT, UMAX, TRAIN_H, HID = W.DT, W.UMAX, W.TRAIN_H, W.HID


def collect(rng, n, L, ic, passive):
    if ic == "narrow": x = np.stack([rng.uniform(-1.5, 1.5, n), rng.uniform(-1, 1, n)], -1)
    else: x = np.stack([rng.uniform(-np.pi, np.pi, n), rng.uniform(-2, 2, n)], -1)
    xs, us = [x], []; xj = jnp.asarray(x, jnp.float32)
    step = jax.jit(jax.vmap(lambda x, u: W.rk4(W.true_field, x, u))); u = np.zeros(n)
    for t in range(L):
        if not passive and t % 5 == 0: u = rng.uniform(-UMAX, UMAX, n)
        xj = step(xj, jnp.asarray(u, jnp.float32)); xs.append(np.asarray(xj)); us.append(u.copy())
    return np.stack(xs, 1), np.stack(us, 1)


def windows(xs, us, noise, sd, seed):
    rng = np.random.default_rng(seed); obs = xs + noise * sd * rng.standard_normal(xs.shape)
    starts = np.arange(0, us.shape[1] - TRAIN_H + 1)
    Wn = np.stack([obs[:, s:s + TRAIN_H + 1] for s in starts], 1).reshape(-1, TRAIN_H + 1, 2)
    Un = np.stack([us[:, s:s + TRAIN_H] for s in starts], 1).reshape(-1, TRAIN_H)
    return jnp.asarray(Wn, jnp.float32), jnp.asarray(Un, jnp.float32)


def make(arm, key):
    k = jax.random.split(key, 3)
    if arm in ("mlp_joint", "mlp_all"): return {"f": W.mlp(k[0], 3, 2, 0.1)}
    if arm in ("fact_joint", "fact_pre"): return {"f": W.mlp(k[0], 2, 2, 0.1), "g": W.mlp(k[1], 2, 2, 0.1)}
    if arm in ("hn_joint", "hn_pre"): return W.init("hn", key)
    raise ValueError(arm)


def field(arm, P, x, u):
    if arm in ("mlp_joint", "mlp_all"): return W.mlpf(P["f"], jnp.concatenate([x, u[None]]))
    if arm in ("fact_joint", "fact_pre"): return W.mlpf(P["f"], x) + W.mlpf(P["g"], x) * u
    return W.model_field("hn", P, x, u)


def fit(arm, P, Wd, Ud, sdj, steps, seed, trainable):
    """Adam on window loss; only top-level keys in `trainable` are updated (others frozen)."""
    sched = optax.cosine_decay_schedule(2e-3, steps); opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched)); st = opt.init(P)

    def loss(P, w, u):
        ff = lambda x, uu: field(arm, P, x, uu)
        _, pred = jax.lax.scan(lambda x, uu: (W.rk4(ff, x, uu),) * 2, w[0], u)
        return jnp.mean(((pred - w[1:]) / sdj) ** 2)

    @jax.jit
    def run(P, st, key):
        def body(c, k):
            P, st = c; idx = jax.random.randint(k, (64,), 0, Wd.shape[0])
            l, g = jax.value_and_grad(lambda P: jnp.mean(jax.vmap(lambda w, u: loss(P, w, u))(Wd[idx], Ud[idx])))(P)
            g = {kk: (v if kk in trainable else jax.tree_util.tree_map(jnp.zeros_like, v)) for kk, v in g.items()}
            up, st = opt.update(g, st, P)
            up = {kk: (v if kk in trainable else jax.tree_util.tree_map(jnp.zeros_like, v)) for kk, v in up.items()}
            return (optax.apply_updates(P, up), st), l
        (P, st), ls = jax.lax.scan(body, (P, st), jax.random.split(key, steps)); return P, ls
    P, ls = run(P, st, jax.random.PRNGKey(seed + 1))
    return P, float(ls[-20:].mean())


def rollout_err(arm, P, seed, n=64, steps=40):
    """Open-loop nRMSE over 40 steps with random torques from FULL-circle ICs (tests extrapolation), and passive-only (u=0)."""
    rng = np.random.default_rng(seed + 777); out = {}
    for name, passive in (("act", False), ("pas", True)):
        xs, us = collect(rng, n, steps, "wide", passive)
        sd = xs.reshape(-1, 2).std(0)
        ff = lambda x, uu: field(arm, P, x, uu)
        pred = jax.vmap(lambda x0, u: jax.lax.scan(lambda x, uu: (W.rk4(ff, x, uu),) * 2, x0, u)[1])(jnp.asarray(xs[:, 0], jnp.float32), jnp.asarray(us, jnp.float32))
        out[name] = float(np.sqrt((((np.asarray(pred) - xs[:, 1:]) / sd) ** 2).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", nargs="+", default=["mlp_joint", "mlp_all", "fact_joint", "fact_pre", "hn_joint", "hn_pre"])
    ap.add_argument("--n-active", nargs="+", type=int, default=[8, 32]); ap.add_argument("--n-passive", type=int, default=256)
    ap.add_argument("--ics", nargs="+", default=["narrow", "wide"]); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--noise", type=float, default=0.02); ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--episodes", type=int, default=24)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); res = []; t0 = time.time()
    print("backend", jax.default_backend(), flush=True)
    ret, best = W.mpc_eval(lambda x, u: W.true_field(x, u), a.episodes, 0)
    res.append(dict(arm="true_model_mpc", ret=float(ret.mean()), success=float((best < 0.3).mean())))
    print("reference true-model MPC", res[-1], flush=True)
    for ic in a.ics:
        for na in a.n_active:
            for seed in a.seeds:
                xa, ua = collect(np.random.default_rng(seed + 1000 * na), na, 60, "narrow", False)
                xp, up = collect(np.random.default_rng(seed + 5000), a.n_passive, 60, ic, True)
                sd = np.concatenate([xa, xp]).reshape(-1, 2).std(0) + 1e-6; sdj = jnp.asarray(sd, jnp.float32)
                Wa, Ua = windows(xa, ua, a.noise, sd, seed); Wp, Up = windows(xp, up, a.noise, sd, seed + 1)
                Wall, Uall = jnp.concatenate([Wa, Wp]), jnp.concatenate([Ua, Up])
                for arm in a.arms:
                    P = make(arm, jax.random.PRNGKey(seed))
                    if arm in ("mlp_joint", "hn_joint"): P, tl = fit(arm, P, Wa, Ua, sdj, a.steps, seed, set(P))
                    elif arm in ("mlp_all", "fact_joint"): P, tl = fit(arm, P, Wall, Uall, sdj, a.steps, seed, set(P))
                    elif arm == "fact_pre":
                        P, _ = fit(arm, P, Wp, Up, sdj, a.steps, seed, {"f"})           # passive only: g gets zero gradient (u=0)
                        P, tl = fit(arm, P, Wa, Ua, sdj, a.steps, seed, {"g"})           # f frozen
                    else:  # hn_pre: V (v), R (L) from passive data; G alone from active data
                        P, _ = fit(arm, P, Wp, Up, sdj, a.steps, seed, {"v", "L", "a"})
                        P, tl = fit(arm, P, Wa, Ua, sdj, max(a.steps // 2, 500), seed, {"G"})
                    ret, best = W.mpc_eval(lambda x, u, P=P, arm=arm: field(arm, P, x, u), a.episodes, 100 + seed)
                    re = rollout_err(arm, P, seed)
                    r = dict(arm=arm, ic=ic, n_active=na, n_passive=a.n_passive, seed=seed, train_loss=tl, ret=float(ret.mean()),
                             success=float((best < 0.3).mean()), min_dist=float(best.mean()), nrmse_active_wideIC=re["act"],
                             nrmse_passive_wideIC=re["pas"], params=int(sum(x.size for x in jax.tree_util.tree_leaves(P))))
                    res.append(r)
                    print(f"{ic:6s} Na={na:3d} s={seed} {arm:11s} ret={r['ret']:8.1f} succ={r['success']:.2f} mind={r['min_dist']:.2f} "
                          f"nrmse act/pas={re['act']:.3f}/{re['pas']:.3f} t={time.time()-t0:.0f}s", flush=True)
                    json.dump(res, open(f"{a.out}/pfirst_results.json", "w"))


if __name__ == "__main__":
    main()
