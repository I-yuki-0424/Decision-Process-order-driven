"""Closed-form-physics benchmark for the Idea-6 world-model family (Hamiltonian prior vs generic dynamics).

Tasks (all with known exact equations; data = numerically exact integration + observation noise):
  kepler    : planar two-body (reduced), H = |p|^2/2 - 1/|q|, n=2, conservative, nonlinear
  fall_drag : falling body, H = p^2/2 + g q with quadratic drag -c|v|v, n=1, DISSIPATIVE (breaks pure Hamiltonian)
  sine      : harmonic oscillator H = (p^2+q^2)/2, n=1, conservative, linear (noisy observations)
Arms (all learn a continuous vector field, same RK4 integrator unless noted; ~equal hidden width):
  mlp        : dx = MLP(x)                        (generic neural ODE)
  hn_gen     : dx = J grad H_theta(q,p), H = MLP  (generic Hamiltonian NN)
  hn_sep     : existing HamiltonianOps: T=|p|^2/2m known, V=MLP(q) learned, J only (from world_model.py, unchanged)
  port_sep   : hn_sep with learned PSD dissipation R, (J-R) grad H   (Approach E, existing code)
  idea6_full : existing WorldModel.step, K=1, presets OFF, its own semi-implicit Euler (8 substeps)
  oracle_port: port_sep + the TRUE potential given as a preset (upper bound; uses task knowledge on purpose)
  persistence: x_{t+k}=x_t (reference)
Protocol: train on short windows (horizon 10) from N trajectories, test open-loop over a long horizon (20-25x longer)
from held-out initial conditions, in-distribution (ID) and out-of-distribution (OOD). Metrics: normalised RMSE vs
horizon; drift of the TRUE invariants (energy, angular momentum) along the predicted rollout.
"""
import argparse, json, os, sys, time
sys.path.insert(0, ".")
import jax, jax.numpy as jnp, numpy as np, optax
from src.model.candidates.world_model import FormulaPresets, HamiltonianOps, WorldModel

TRAIN_H = 10
HID = 64
ETA = jnp.array([1.0, 0, 0, 0, 0, 0, 0])
NOPRESET = jnp.zeros((FormulaPresets.N_POT,), dtype=bool)


# ------------------------------------------------------------------ tasks (ground truth) -----------------------
class Task:
    def __init__(s, name):
        s.name = name
        s.n = 2 if name == "kepler" else 1
        s.dt, s.test_h, s.traj_len = {"kepler": (0.05, 250, 60), "fall_drag": (0.05, 200, 60), "sine": (0.1, 250, 60)}[name]

    def field(s, x):  # true dx/dt, numpy, x (...,2n)
        n = s.n; q, p = x[..., :n], x[..., n:]
        if s.name == "kepler":
            r = np.linalg.norm(q, axis=-1, keepdims=True)
            return np.concatenate([p, -q / r ** 3], -1)
        if s.name == "fall_drag":
            return np.concatenate([p, -1.0 - 0.3 * np.abs(p) * p], -1)  # g=1, c=0.3, m=1, q=height
        return np.concatenate([p, -q], -1)

    def V(s, q):  # true potential (jnp) for the oracle arm
        if s.name == "kepler": return -1.0 / jnp.sqrt(jnp.sum(q ** 2) + 1e-6)
        if s.name == "fall_drag": return 1.0 * q[0]
        return 0.5 * q[0] ** 2

    def invariants(s, x):  # jnp (...,2n) -> (...,k) true conserved quantities (energy [, ang mom])
        n = s.n; q, p = x[..., :n], x[..., n:]
        if s.name == "kepler":
            E = 0.5 * jnp.sum(p ** 2, -1) - 1.0 / jnp.linalg.norm(q, axis=-1)
            L = q[..., 0] * p[..., 1] - q[..., 1] * p[..., 0]
            return jnp.stack([E, L], -1)
        if s.name == "fall_drag":
            return (0.5 * p[..., 0] ** 2 + q[..., 0])[..., None]  # NOT conserved (drag) -- reported as energy dissipation
        return (0.5 * (p[..., 0] ** 2 + q[..., 0] ** 2))[..., None]

    def ic(s, rng, m, ood=False):
        if s.name == "kepler":
            r = rng.uniform(1.6, 2.2, m) if ood else rng.uniform(0.7, 1.3, m)
            th = rng.uniform(0, 2 * np.pi, m); eps = rng.uniform(-0.25, 0.25, m); v = np.sqrt(1 / r) * (1 + eps)
            return np.stack([r * np.cos(th), r * np.sin(th), -v * np.sin(th), v * np.cos(th)], -1)
        if s.name == "fall_drag":
            return np.stack([rng.uniform(15, 25, m) if ood else rng.uniform(5, 10, m), rng.uniform(-2, 2, m)], -1)
        A = rng.uniform(2, 3, m) if ood else rng.uniform(0.5, 1.5, m); ph = rng.uniform(0, 2 * np.pi, m)
        return np.stack([A * np.cos(ph), -A * np.sin(ph)], -1)

    def simulate(s, x0, steps):  # exact-ish: RK4 with 50 substeps/step in float64
        xs = [x0]; x = x0.copy(); h = s.dt / 50
        for _ in range(steps):
            for _ in range(50):
                k1 = s.field(x); k2 = s.field(x + h / 2 * k1); k3 = s.field(x + h / 2 * k2); k4 = s.field(x + h * k3)
                x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            xs.append(x)
        return np.stack(xs, 1)  # (m, steps+1, 2n)


# ------------------------------------------------------------------ models -----------------------------------
def mlp_init(key, i, o, last=0.1):
    k1, k2, k3 = jax.random.split(key, 3)
    return dict(w1=jax.random.normal(k1, (i, HID)) / jnp.sqrt(i), b1=jnp.zeros(HID),
                w2=jax.random.normal(k2, (HID, HID)) / jnp.sqrt(HID), b2=jnp.zeros(HID),
                w3=jax.random.normal(k3, (HID, o)) * last / jnp.sqrt(HID), b3=jnp.zeros(o))


def mlp(p, x):
    h = jnp.tanh(x @ p["w1"] + p["b1"]); h = jnp.tanh(h @ p["w2"] + p["b2"]); return h @ p["w3"] + p["b3"]


def build(arm, task, key):
    n = task.n; d = 2 * n; k1, k2 = jax.random.split(key)
    if arm == "mlp":
        P = mlp_init(k1, d, d)
        return P, lambda P, x: mlp(P, x)
    if arm == "hn_gen":
        P = mlp_init(k1, d, 1)
        return P, lambda P, x: (jnp.block([[jnp.zeros((n, n)), jnp.eye(n)], [-jnp.eye(n), jnp.zeros((n, n))]])
                                @ jax.grad(lambda z: mlp(P, z)[0])(x))
    if arm in ("hn_sep", "port_sep", "oracle_port"):
        P = dict(h=HamiltonianOps.init_parameters(k1, n, HID), L=jax.random.normal(k2, (d, d)) * 0.01)
        use_R = arm != "hn_sep"

        def H(P, x):
            h = HamiltonianOps.hamiltonian(P["h"], x, ETA, NOPRESET)
            return h + task.V(x[:n]) if arm == "oracle_port" else h

        def f(P, x):
            g = jax.grad(H, argnums=1)(P, x)
            J = HamiltonianOps.symplectic_matrix(n)
            R = HamiltonianOps.dissipation_matrix(P["L"]) if use_R else 0.0
            return (J - R) @ g
        return P, f
    if arm == "idea6_full" or arm.startswith("i6_"):
        P = WorldModel.init_parameters(k1, n, 1, 1, 1, HID)
        if "hninit" in arm:  # HNN MLP init scaled like the other arms (1/sqrt(fan_in)) instead of 0.1 constant
            kk = jax.random.split(k2, 2)
            P["hnn"]["w1"] = jax.random.normal(kk[0], P["hnn"]["w1"].shape) / jnp.sqrt(n)
            P["hnn"]["w2"] = jax.random.normal(kk[1], P["hnn"]["w2"].shape) / jnp.sqrt(HID)
        if "rbig" in arm:  # dissipation matrix init as in port_sep (0.01 already) -> larger, 0.1
            P["R_L"] = P["R_L"] * 10.0
        return P, None
    raise ValueError(arm)


def make_rollout(arm, task, f):
    dt = task.dt
    if arm == "idea6_full":
        mask = jnp.zeros((1, FormulaPresets.N_POT + FormulaPresets.N_FORCE), dtype=bool)

        def step(P, x):
            for _ in range(8):
                x, _ = WorldModel.step(P, x, jnp.zeros(1), jnp.zeros(1), ETA, mask, dt / 8)
            return x
    elif arm.startswith("i6_"):
        # Ablations of WorldModel (component isolation). Flags in the arm name:
        #   nores : residual MLP r_theta removed (res_w2/res_b2 forced to 0, no gradient)
        #   rk4   : RK4 on the identical vector field instead of semi-implicit Euler x8 substeps
        #   nogate: mode vector fixed to [1.] (K=1, so gate is provably a no-op; checks gate plumbing/gradients)
        #   noR   : learned PSD dissipation matrix removed (J-only)
        #   hninit/rbig: init variants (see build)
        flags = arm[3:].split("_")
        mask = jnp.zeros((1, FormulaPresets.N_POT + FormulaPresets.N_FORCE), dtype=bool)
        z1 = jnp.zeros(1)

        def field(P, x):
            if "nores" in flags:
                P = {**P, "res_w2": jax.lax.stop_gradient(P["res_w2"]) * 0.0, "res_b2": jax.lax.stop_gradient(P["res_b2"]) * 0.0}
            if "noR" in flags:  # remove learned dissipation R (conservative J-only, as hn_sep)
                P = {**P, "R_L": jax.lax.stop_gradient(P["R_L"]) * 0.0}
            m = jnp.ones(1) if "nogate" in flags else WorldModel.mode_vector(P, x, z1)[0]
            return WorldModel._vector_field(P, x, z1, z1, ETA, mask, m)

        def step(P, x):
            if "rk4" in flags:
                k1 = field(P, x); k2 = field(P, x + dt / 2 * k1); k3 = field(P, x + dt / 2 * k2); k4 = field(P, x + dt * k3)
                return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            nn = x.shape[0] // 2
            for _ in range(8):
                h = dt / 8
                dx = field(P, x); p_new = x[nn:] + h * dx[nn:]
                dq = field(P, jnp.concatenate([x[:nn], p_new]))[:nn]
                x = jnp.concatenate([x[:nn] + h * dq, p_new])
            return x
    else:
        def step(P, x):
            k1 = f(P, x); k2 = f(P, x + dt / 2 * k1); k3 = f(P, x + dt / 2 * k2); k4 = f(P, x + dt * k3)
            return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def rollout(P, x0, T):
        return jax.lax.scan(lambda x, _: (step(P, x), step(P, x)), x0, None, length=T)[1]
    return rollout


# ------------------------------------------------------------------ experiment -------------------------------
def run_one(task, arm, n_traj, noise, seed, steps, test):
    rng = np.random.default_rng(seed)
    x0 = task.ic(rng, n_traj); traj = task.simulate(x0, task.traj_len)               # (N, L+1, 2n)
    sd = traj.reshape(-1, 2 * task.n).std(0) + 1e-6
    obs = traj + noise * sd * rng.standard_normal(traj.shape)
    # windows: all start positions
    L = task.traj_len; starts = np.arange(0, L - TRAIN_H + 1)
    W_in = obs[:, starts][:, :, :][..., :]                                              # (N, S, 2n) inputs
    W_out = np.stack([obs[:, s + 1:s + 1 + TRAIN_H] for s in starts], 1)                # (N, S, H, 2n)
    W_in = W_in.reshape(-1, 2 * task.n); W_out = W_out.reshape(-1, TRAIN_H, 2 * task.n)
    W_in, W_out, sd_j = jnp.asarray(W_in, jnp.float32), jnp.asarray(W_out, jnp.float32), jnp.asarray(sd, jnp.float32)

    P, f = build(arm, task, jax.random.PRNGKey(seed))
    roll = make_rollout(arm, task, f)
    n_par = int(sum(x.size for x in jax.tree_util.tree_leaves(P)))
    sched = optax.cosine_decay_schedule(2e-3, steps)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched)); ost = opt.init(P)

    def loss(P, xi, yo):
        pred = jax.vmap(lambda x: roll(P, x, TRAIN_H))(xi)
        return jnp.mean(((pred - yo) / sd_j) ** 2)

    @jax.jit
    def train(P, ost, key):
        def body(c, k):
            P, ost = c
            idx = jax.random.randint(k, (128,), 0, W_in.shape[0])
            l, g = jax.value_and_grad(loss)(P, W_in[idx], W_out[idx])
            u, ost = opt.update(g, ost, P); return (optax.apply_updates(P, u), ost), l
        (P, ost), ls = jax.lax.scan(body, (P, ost), jax.random.split(key, steps)); return P, ls
    P, ls = train(P, ost, jax.random.PRNGKey(seed + 1))
    out = dict(params=n_par, final_train_loss=float(ls[-10:].mean()))
    for split, (x_te, truth) in test.items():
        xin = x_te + noise * sd * np.random.default_rng(seed + 7).standard_normal(x_te.shape)  # noisy observation of x0
        pred = np.asarray(jax.jit(jax.vmap(lambda x: roll(P, x, task.test_h)))(jnp.asarray(xin, jnp.float32)))
        out[split] = metrics(task, pred, truth, sd)
    return out


def metrics(task, pred, truth, sd):
    err = np.sqrt((((pred - truth) / sd) ** 2).mean(-1))                               # (M, T) normalised RMSE
    inv_p = np.asarray(task.invariants(jnp.asarray(pred))); inv_t = np.asarray(task.invariants(jnp.asarray(truth)))
    drift = np.abs(inv_p - inv_t[:, :1]) / (np.abs(inv_t[:, :1]) + 1e-3)               # drift of TRUE invariants along prediction
    T = err.shape[1]
    return dict(rmse_h10=float(np.median(err[:, 9])), rmse_mid=float(np.median(err[:, T // 2 - 1])),
                rmse_end=float(np.median(err[:, -1])), rmse_curve=np.median(err, 0)[::max(1, T // 25)].tolist(),
                inv_drift_end=np.median(drift[:, -1], 0).tolist(),
                finite_frac=float(np.isfinite(pred).all((1, 2)).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--tasks", nargs="+", default=["kepler", "fall_drag", "sine"])
    ap.add_argument("--arms", nargs="+", default=["mlp", "hn_gen", "hn_sep", "port_sep", "idea6_full", "oracle_port"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[16, 128, 1024]); ap.add_argument("--noises", nargs="+", type=float, default=[0.0, 0.02])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2]); ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    print("backend", jax.default_backend(), flush=True)
    res = []; t0 = time.time()
    for tn in a.tasks:
        task = Task(tn); rng = np.random.default_rng(999)
        test = {}
        for split, ood in (("id", False), ("ood", True)):
            x0 = task.ic(rng, 64, ood); test[split] = (x0, task.simulate(x0, task.test_h)[:, 1:])
        sd_ref = task.simulate(task.ic(np.random.default_rng(5), 256), task.traj_len).reshape(-1, 2 * task.n).std(0) + 1e-6
        for split in test:  # persistence reference
            pers = np.repeat(test[split][0][:, None], task.test_h, 1)
            res.append(dict(task=tn, arm="persistence", n_traj=0, noise=0.0, seed=0, **{"params": 0, split: metrics(task, pers, test[split][1], sd_ref)}))
        for arm in a.arms:
            for n_traj in a.sizes:
                for noise in a.noises:
                    for seed in a.seeds:
                        try:
                            r = run_one(task, arm, n_traj, noise, seed, a.steps, test)
                        except Exception as e:
                            r = dict(error=repr(e))
                        r.update(task=tn, arm=arm, n_traj=n_traj, noise=noise, seed=seed); res.append(r)
                        if "error" not in r:
                            print(f"{tn:9s} {arm:11s} N={n_traj:5d} noise={noise:.2f} s={seed} par={r['params']:6d} "
                                  f"ID h10={r['id']['rmse_h10']:.3f} end={r['id']['rmse_end']:.3f} | OOD end={r['ood']['rmse_end']:.3f} "
                                  f"t={time.time()-t0:.0f}s", flush=True)
                        else:
                            print(tn, arm, n_traj, "ERROR", r["error"], flush=True)
                json.dump(res, open(f"{a.out}/physics_results.json", "w"))
    json.dump(res, open(f"{a.out}/physics_results.json", "w"))


if __name__ == "__main__":
    main()
