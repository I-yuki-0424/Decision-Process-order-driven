"""Does world-model quality translate into decision quality? Torque-limited pendulum swing-up with model-predictive control (CEM planner).

True system: theta' = p, p' = -sin(theta) - 0.05 p + u, |u| <= 0.7 (too weak to lift directly: needs pumping),
start hanging (theta ~ 0), goal theta = pi. reward = -(1 + cos theta) - 0.01 u^2 per step, dt = 0.05, 400 steps.
Offline data: N short trajectories with random piecewise-constant torques from ICs near the bottom (|theta0| < 1.5), so
the upright region is rarely/never visited: the model must EXTRAPOLATE. Planner: CEM MPC (3 iters x 256 sequences, 6 blocks,
horizon 60, replanning every step) using ONLY the learned model; the environment is stepped with the true dynamics.
Arms (each learns G u): mlp | hn (known kinetic, learned V, R) | phys_hn (small-angle preset V=q^2/2 [WRONG at large angle] + learned V, R)
| hyb (phys_hn + MLP residual on (x,u)) | oracle_phys (TRUE potential -cos(theta) as preset; upper bound, uses task knowledge on purpose).
References: true-dynamics MPC (planner ceiling) and a zero-torque policy.
"""
import argparse, json, os, sys, time
sys.path.insert(0, ".")
import jax, jax.numpy as jnp, numpy as np, optax
from src.model.candidates.world_model import HamiltonianOps

DT, UMAX, HOR, NSAMP, EPLEN, TRAIN_H, HID = 0.05, 0.7, 60, 256, 400, 10, 64
BLK, NITER, NELITE = 10, 3, 32


def true_field(x, u):
    return jnp.stack([x[1], -jnp.sin(x[0]) - 0.05 * x[1] + u])


def rk4(f, x, u):
    k1 = f(x, u); k2 = f(x + DT / 2 * k1, u); k3 = f(x + DT / 2 * k2, u); k4 = f(x + DT * k3, u)
    return x + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def reward(x, u):
    return -(1.0 + jnp.cos(x[0])) - 0.01 * u ** 2


def dense(key, i, o, s=1.0):
    return dict(w=jax.random.normal(key, (i, o)) * s / jnp.sqrt(i), b=jnp.zeros(o))


def mlp(key, i, o, last):
    k = jax.random.split(key, 3); return [dense(k[0], i, HID), dense(k[1], HID, HID), dense(k[2], HID, o, last)]


def mlpf(p, x):
    h = jnp.tanh(x @ p[0]["w"] + p[0]["b"]); h = jnp.tanh(h @ p[1]["w"] + p[1]["b"]); return h @ p[2]["w"] + p[2]["b"]


def init(arm, key):
    k = jax.random.split(key, 5); P = {}
    if arm == "mlp":
        P["f"] = mlp(k[0], 3, 2, 0.1); return P
    P["v"] = HamiltonianOps.init_parameters(k[0], 2 if arm in ("phys_feat", "oracle_feat", "skip_feat", "skip_feat0", "oracle_skip") else 1, HID);
    P["a"] = jnp.zeros(()) if arm == "skip_feat0" else jnp.ones(())  # skip-connection scale on the preset potential
    P["L"] = jax.random.normal(k[1], (2, 2)) * 0.01
    P["G"] = jnp.zeros(2)
    if arm == "hyb": P["res"] = mlp(k[2], 3, 2, 0.01)
    return P


J = HamiltonianOps.symplectic_matrix(1)


def model_field(arm, P, x, u):
    if arm == "mlp": return mlpf(P["f"], jnp.concatenate([x, u[None]]))
    def H(z):
        q, p = z[:1], z[1:]
        if arm in ("skip_feat", "skip_feat0", "oracle_skip"):  # preset as feature of the learned potential PLUS a learned-scale direct skip
            feat = 0.5 * q[0] ** 2 if arm != "oracle_skip" else -jnp.cos(q[0])
            return 0.5 * jnp.sum(p ** 2) + HamiltonianOps.mlp_scalar(P["v"], jnp.concatenate([q, feat[None]])) + P["a"] * feat
        if arm in ("phys_feat", "oracle_feat"):  # preset potential enters as an INPUT FEATURE of the learned potential (no structural lock-in)
            feat = 0.5 * q[0] ** 2 if arm == "phys_feat" else -jnp.cos(q[0])
            return 0.5 * jnp.sum(p ** 2) + HamiltonianOps.mlp_scalar(P["v"], jnp.concatenate([q, feat[None]]))
        h = 0.5 * jnp.sum(p ** 2) + HamiltonianOps.mlp_scalar(P["v"], q)
        if arm in ("phys_hn", "hyb"): h = h + 0.5 * q[0] ** 2
        if arm == "oracle_phys": h = h - jnp.cos(q[0])
        return h
    R = HamiltonianOps.dissipation_matrix(P["L"])
    f = (J - R) @ jax.grad(H)(x) + P["G"] * u
    if arm == "hyb": f = f + mlpf(P["res"], jnp.concatenate([x, u[None]]))
    return f


def collect(rng, n, L=60):
    x = np.stack([rng.uniform(-1.5, 1.5, n), rng.uniform(-1, 1, n)], -1)
    xs, us = [x], []
    xj = jnp.asarray(x, jnp.float32)
    step = jax.jit(jax.vmap(lambda x, u: rk4(true_field, x, u)))
    u = np.zeros(n)
    for t in range(L):
        if t % 5 == 0: u = rng.uniform(-UMAX, UMAX, n)
        xj = step(xj, jnp.asarray(u, jnp.float32)); xs.append(np.asarray(xj)); us.append(u.copy())
    return np.stack(xs, 1), np.stack(us, 1)  # (n,L+1,2), (n,L)


def train(arm, xs, us, noise, seed, steps):
    sd = xs.reshape(-1, 2).std(0) + 1e-6
    rng = np.random.default_rng(seed); obs = xs + noise * sd * rng.standard_normal(xs.shape)
    L = us.shape[1]; starts = np.arange(0, L - TRAIN_H + 1)
    W = jnp.asarray(np.stack([obs[:, s:s + TRAIN_H + 1] for s in starts], 1).reshape(-1, TRAIN_H + 1, 2), jnp.float32)
    U = jnp.asarray(np.stack([us[:, s:s + TRAIN_H] for s in starts], 1).reshape(-1, TRAIN_H), jnp.float32)
    sdj = jnp.asarray(sd, jnp.float32)
    P = init(arm, jax.random.PRNGKey(seed))
    sched = optax.cosine_decay_schedule(2e-3, steps); opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched)); st = opt.init(P)

    def loss(P, w, u):
        f = lambda x, uu: model_field(arm, P, x, uu)
        _, pred = jax.lax.scan(lambda x, uu: (rk4(f, x, uu),) * 2, w[0], u)
        return jnp.mean(((pred - w[1:]) / sdj) ** 2)

    @jax.jit
    def run(P, st, key):
        def body(c, k):
            P, st = c; idx = jax.random.randint(k, (64,), 0, W.shape[0])
            l, g = jax.value_and_grad(lambda P: jnp.mean(jax.vmap(lambda w, u: loss(P, w, u))(W[idx], U[idx])))(P)
            up, st = opt.update(g, st, P); return (optax.apply_updates(P, up), st), l
        (P, st), ls = jax.lax.scan(body, (P, st), jax.random.split(key, steps)); return P, ls
    P, ls = run(P, st, jax.random.PRNGKey(seed + 1))
    return P, float(ls[-20:].mean())


def mpc_eval(field_fn, n_ep, seed, noise_obs=0.0):
    """field_fn(x,u)->dx used for planning; env uses the true dynamics. Returns per-episode return, min |theta-pi| reached."""
    def plan(x, key):
        nb = HOR // BLK

        def roll(us):  # us (nb,)
            def body(c, u):
                x2 = rk4(field_fn, c, u); return x2, reward(x2, u)
            return jnp.sum(jax.lax.scan(body, x, jnp.repeat(us, BLK))[1])

        def cem(c, k):
            mu, sg = c
            cand = jnp.clip(mu + sg * jax.random.normal(k, (NSAMP, nb)), -UMAX, UMAX)
            R = jax.vmap(roll)(cand); el = cand[jnp.argsort(-R)[:NELITE]]
            return (el.mean(0), el.std(0) + 0.05), None
        (mu, _), _ = jax.lax.scan(cem, (jnp.zeros(nb), jnp.full(nb, UMAX)), jax.random.split(key, NITER))
        return mu[0]

    def episode(key):
        k0, k1 = jax.random.split(key); x0 = jnp.stack([jax.random.uniform(k0, (), minval=-0.2, maxval=0.2), 0.0])

        def body(c, t):
            x, best = c; kt = jax.random.fold_in(k1, t)
            xo = x + noise_obs * jax.random.normal(kt, (2,)) * jnp.array([1.0, 1.0])
            u = plan(xo, kt); x2 = rk4(true_field, x, u)
            d = jnp.abs(((x2[0] - jnp.pi + jnp.pi) % (2 * jnp.pi)) - jnp.pi)  # angular distance to upright
            return (x2, jnp.minimum(best, d)), reward(x2, u)
        (xf, best), rs = jax.lax.scan(body, (x0, jnp.array(10.0)), jnp.arange(EPLEN))
        return rs.sum(), best
    return jax.jit(jax.vmap(episode))(jax.random.split(jax.random.PRNGKey(seed), n_ep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--arms", nargs="+", default=["mlp", "hn", "phys_hn", "hyb", "oracle_phys"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[16, 64, 256]); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--noise", type=float, default=0.02); ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--episodes", type=int, default=24)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); res = []; t0 = time.time()
    print("backend", jax.default_backend(), flush=True)
    ret, best = mpc_eval(true_field2 := (lambda x, u: true_field(x, u)), a.episodes, 0)
    res.append(dict(arm="true_model_mpc", ret=float(ret.mean()), success=float((best < 0.3).mean()), seed=0, n_traj=0))
    ret, best = mpc_eval(lambda x, u: jnp.zeros(2) + jnp.stack([x[1], 0.0]) * 0, a.episodes, 0)  # planner with a null model ~ arbitrary
    res.append(dict(arm="null_model_mpc", ret=float(ret.mean()), success=float((best < 0.3).mean()), seed=0, n_traj=0))
    print("references", res, flush=True)
    for arm in a.arms:
        for n in a.sizes:
            for seed in a.seeds:
                xs, us = collect(np.random.default_rng(seed + 1000 * n), n)
                P, tl = train(arm, xs, us, a.noise, seed, a.steps)
                cover = float(np.mean(np.abs(xs[..., 0]) > 2.5))
                ret, best = mpc_eval(lambda x, u, P=P, arm=arm: model_field(arm, P, x, u), a.episodes, 100 + seed)
                r = dict(arm=arm, n_traj=n, seed=seed, train_loss=tl, ret=float(ret.mean()), success=float((best < 0.3).mean()),
                         min_dist=float(best.mean()), frac_data_theta_gt_2p5=cover, params=int(sum(x.size for x in jax.tree_util.tree_leaves(P))))
                res.append(r)
                print(f"{arm:12s} N={n:4d} s={seed} ret={r['ret']:8.1f} success={r['success']:.2f} mindist={r['min_dist']:.2f} "
                      f"cover={cover:.3f} par={r['params']} t={time.time()-t0:.0f}s", flush=True)
                json.dump(res, open(f"{a.out}/planning_results.json", "w"))
    json.dump(res, open(f"{a.out}/planning_results.json", "w"))


if __name__ == "__main__":
    main()
