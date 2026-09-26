"""Hybrid world model benchmark: known (possibly INCOMPLETE/WRONG) physics + Hamiltonian/port core + Dreamer-lite latent residual.

Operator proposal (2026-09-26): unify (1) Hamiltonian-network layers, (2) layers embedding known physical equations and
(3) a DreamerV3-style learned part, fused (e.g. by a Transformer) into one world model.
Critical-review reading: the known equations + HN carry structure/extrapolation; the learned latent part must supply what the
equations miss. Hence tasks where the preset is deliberately incomplete or wrong:
  kepler_pert  : true = Newton + REPULSIVE +0.25/r^5 perturbation + linear drag 0.02 (an attractive one made 30% of orbits plunge/eject: first run invalid); preset = Newton mu=1.0
  kepler_wrongmu: same truth, preset mu=0.7 (wrong constant)
  fall_drag    : true = gravity + quadratic drag; preset = gravity only
  pendulum     : true = -sin(q) with damping; preset = harmonic (small-angle) oscillator; OOD = large amplitude
'Dreamer-lite' = GRU + categorical stochastic latent (RSSM, straight-through, 1% unimix, DreamerV3 KL balancing 0.5/0.1 with
free bits 1 nat, teacher-forced posterior pass + open-loop imagination pass). NOT a full DreamerV3 (no symlog heads, no
actor-critic, no image encoder). All models are continuous-time vector fields integrated with the same RK4.
Arms: mlp | hn | phys | phys_hn | rssm | hyb_sum_mlp | hyb_sum_rssm | hyb_tf_rssm  (+ 'persistence' reference).
"""
import argparse, json, os, sys, time
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import jax, jax.numpy as jnp, numpy as np, optax
import run_physics_benchmark as pb
from src.model.candidates.world_model import HamiltonianOps

HID = 64
TRAIN_H = pb.TRAIN_H
NG, NC = 4, 8            # categorical latent: 4 groups x 8 classes
DH = 32                  # GRU hidden (overridable with --dh)
ZD = NG * NC


# ---------------------------------------------------------------- tasks --------------------------------------------
class HTask(pb.Task):
    def __init__(s, name, mu_preset=None):
        s.name = name
        s.n = 2 if name.startswith("kepler") else 1
        s.dt, s.test_h, s.traj_len = {"kepler": (0.05, 250, 60), "fall": (0.05, 200, 60), "pendulum": (0.1, 200, 60)}[
            "kepler" if name.startswith("kepler") else ("fall" if name == "fall_drag" else "pendulum")]
        s.mu_p = 0.7 if name == "kepler_wrongmu" else 1.0

    def field(s, x):
        n = s.n; q, p = x[..., :n], x[..., n:]
        if s.name.startswith("kepler"):
            r = np.linalg.norm(q, axis=-1, keepdims=True)
            return np.concatenate([p, -q / r ** 3 + 0.25 * q / r ** 5 - 0.02 * p], -1)
        if s.name == "fall_drag":
            return np.concatenate([p, -1.0 - 0.3 * np.abs(p) * p], -1)
        return np.concatenate([p, -np.sin(q) - 0.05 * p], -1)

    def Vp(s, q):  # preset potential (the "known" equation; incomplete/wrong by design)
        if s.name.startswith("kepler"): return -s.mu_p / jnp.sqrt(jnp.sum(q ** 2) + 1e-6)
        if s.name == "fall_drag": return 1.0 * q[0]
        return 0.5 * q[0] ** 2

    def invariants(s, x):  # only used by pb.metrics; Newtonian/harmonic energy (NOT conserved by the true system)
        n = s.n; q, p = x[..., :n], x[..., n:]
        if s.name.startswith("kepler"):
            return (0.5 * jnp.sum(p ** 2, -1) - 1.0 / jnp.linalg.norm(q, axis=-1))[..., None]
        return (0.5 * p[..., 0] ** 2 + q[..., 0] ** 2 / 2)[..., None]

    def ic(s, rng, m, ood=False):
        if s.name.startswith("kepler"):
            r = rng.uniform(1.6, 2.2, m) if ood else rng.uniform(0.7, 1.3, m)
            th = rng.uniform(0, 2 * np.pi, m); eps = rng.uniform(-0.25, 0.25, m); v = np.sqrt(1 / r) * (1 + eps)
            return np.stack([r * np.cos(th), r * np.sin(th), -v * np.sin(th), v * np.cos(th)], -1)
        if s.name == "fall_drag":
            return np.stack([rng.uniform(15, 25, m) if ood else rng.uniform(5, 10, m), rng.uniform(-2, 2, m)], -1)
        q0 = rng.uniform(1.8, 2.6, m) * rng.choice([-1, 1], m) if ood else rng.uniform(-1, 1, m)
        return np.stack([q0, rng.uniform(-0.5, 0.5, m)], -1)


# ---------------------------------------------------------------- components -------------------------------------
def dense(key, i, o, s=1.0):
    return dict(w=jax.random.normal(key, (i, o)) * s / jnp.sqrt(i), b=jnp.zeros(o))


def lin(p, x): return x @ p["w"] + p["b"]


def mlp2(key, i, o, last=1.0, h=None):
    h = h or HID
    k = jax.random.split(key, 3)
    return [dense(k[0], i, h), dense(k[1], h, h), dense(k[2], h, o, last)]


V3 = False  # DreamerV3-style blocks (--v3): parameter-free LayerNorm + SiLU in MLPs/GRU, symlog on RSSM decoder inputs


def _ln(x): return (x - x.mean()) / jnp.sqrt(x.var() + 1e-5)


def _act(x): return jax.nn.silu(_ln(x)) if V3 else jnp.tanh(x)


def symlog(x): return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def mlp2f(p, x):
    return lin(p[2], _act(lin(p[1], _act(lin(p[0], x)))))


def init_params(arm, task, key):
    n = task.n; d = 2 * n; ks = jax.random.split(key, 8); P = {}
    if arm in ("phys_feat", "skip_feat", "skip_feat0"):
        if arm == "skip_feat": P["a"] = jnp.ones(())
        if arm == "skip_feat0": P["a"] = jnp.zeros(())  # skip starts CLOSED; learned gate decides whether to trust the preset
        P["hn"] = dict(v=HamiltonianOps.init_parameters(ks[0], task.n + 1, HID), L=jax.random.normal(ks[1], (d, d)) * 0.01)
    if arm in ("hn", "phys_hn", "phys_hn_mu", "hyb_sum_mlp", "hyb_gate_mlp", "hyb_force_mlp", "hyb_sum_rssm", "hyb_tf_rssm"):
        P["hn"] = dict(v=HamiltonianOps.init_parameters(ks[0], n, HID), L=jax.random.normal(ks[1], (d, d)) * 0.01)
    if arm == "phys_hn_mu":
        P["mu"] = jnp.zeros(())  # log-scale of the known potential: known FORM, unknown constant
    if arm in ("mlp", "hyb_sum_mlp", "hyb_gate_mlp", "hyb_force_mlp"):
        P["res"] = mlp2(ks[2], d, d, 0.1 if arm == "mlp" else 0.01)
    if arm == "hyb_gate_mlp":
        P["gate"] = dict(w=jnp.zeros((d, d)), b=jnp.full((d,), -3.0))  # gate starts closed: sigmoid(-3)=0.05
    if arm in ("rssm", "hyb_sum_rssm", "hyb_tf_rssm"):
        last = 1.0 if arm == "rssm" else 0.01
        P["rssm"] = dict(
            init=mlp2(ks[3], d, DH), gru=dict(wz=dense(ks[4], DH + ZD, DH), wr=dense(ks[5], DH + ZD, DH), wh=dense(ks[6], DH + ZD, DH)),
            prior=mlp2(ks[7], DH, ZD, h=max(8, HID // 2)), post=mlp2(jax.random.fold_in(key, 11), DH + d, ZD, h=max(8, HID // 2)),
            dec=mlp2(jax.random.fold_in(key, 12), DH + ZD + d, d, last))
    if arm == "hyb_tf_rssm":
        kk = jax.random.split(jax.random.fold_in(key, 13), 8)
        P["tf"] = dict(emb=[dense(kk[i], 2 * d, 32) for i in range(3)], pos=jax.random.normal(kk[3], (3, 32)) * 0.1,
                       q=dense(kk[4], 32, 32), k=dense(kk[5], 32, 32), v=dense(kk[6], 32, 32), out=dense(kk[7], 32 * 3, d))
    return P


def gru(p, h, z):
    xz = jnp.concatenate([h, z])
    f = _ln if V3 else (lambda a: a)
    u = jax.nn.sigmoid(f(lin(p["wz"], xz))); r = jax.nn.sigmoid(f(lin(p["wr"], xz)))
    c = jnp.tanh(f(lin(p["wh"], jnp.concatenate([r * h, z]))))
    return (1 - u) * h + u * c


def unimix(logits):
    pr = jax.nn.softmax(logits.reshape(NG, NC), -1)
    return 0.99 * pr + 0.01 / NC


def sample_z(probs, key, sample):
    if sample:
        idx = jax.random.categorical(key, jnp.log(probs))
    else:
        idx = jnp.argmax(probs, -1)
    hard = jax.nn.one_hot(idx, NC)
    return (hard + probs - jax.lax.stop_gradient(probs)).reshape(-1)


def kl_cat(p, q):  # sum over groups of KL(p||q)
    return jnp.sum(p * (jnp.log(p) - jnp.log(q)))


def make_model(arm, task):
    n = task.n; dt = task.dt
    J = HamiltonianOps.symplectic_matrix(n)
    use_hn = "hn" in arm or arm in ("phys_hn", "phys_feat", "skip_feat", "skip_feat0")
    use_phys = arm in ("phys", "phys_hn", "phys_hn_mu", "hyb_sum_mlp", "hyb_gate_mlp", "hyb_force_mlp", "hyb_sum_rssm", "hyb_tf_rssm")
    has_rssm = arm in ("rssm", "hyb_sum_rssm", "hyb_tf_rssm")

    def Htot(P, x):
        q, p = x[:n], x[n:]
        h = 0.0
        if arm in ("phys_feat", "skip_feat", "skip_feat0"):  # preset potential as an input feature of the learned potential (+ learned-scale skip for skip_feat)
            hh = 0.5 * jnp.sum(p ** 2) + HamiltonianOps.mlp_scalar(P["hn"]["v"], jnp.concatenate([q, task.Vp(q)[None]]))
            return hh + P["a"] * task.Vp(q) if arm != "phys_feat" else hh
        if use_phys: h = h + 0.5 * jnp.sum(p ** 2) + task.Vp(q) * (jnp.exp(P["mu"]) if arm == "phys_hn_mu" else 1.0)
        if use_hn:
            h = h + HamiltonianOps.mlp_scalar(P["hn"]["v"], q)
            if not use_phys: h = h + 0.5 * jnp.sum(p ** 2)  # known kinetic term (as in hn_sep)
        return h

    def f_port(P, x):
        R = HamiltonianOps.dissipation_matrix(P["hn"]["L"]) if use_hn else 0.0
        return (J - R) @ jax.grad(lambda z: Htot(P, z))(x)

    def f_phys(x):
        return J @ jax.grad(lambda z: 0.5 * jnp.sum(z[n:] ** 2) + task.Vp(z[:n]))(x)

    def res_field(P, x, ctx):
        if arm == "hyb_force_mlp": return jnp.concatenate([jnp.zeros(n), mlp2f(P["res"], x)[n:]])  # unmodelled FORCE only; dq/dt = p kinematics kept
        if arm in ("mlp", "hyb_sum_mlp", "hyb_gate_mlp"): return mlp2f(P["res"], x)
        h, z = ctx; return mlp2f(P["rssm"]["dec"], jnp.concatenate([h, z, symlog(x) if V3 else x]))

    def tf_fuse(P, x, fp, fh, r):
        T = P["tf"]; toks = jnp.stack([lin(T["emb"][i], jnp.concatenate([f, x])) for i, f in enumerate((fp, fh, r))]) + T["pos"]
        att = jax.nn.softmax(lin(T["q"], toks) @ lin(T["k"], toks).T / jnp.sqrt(32.0), -1)
        toks = toks + att @ lin(T["v"], toks)
        return lin(T["out"], jnp.tanh(toks).reshape(-1))

    def field(P, x, ctx):
        if arm == "mlp": return res_field(P, x, None)
        if arm == "rssm": return res_field(P, x, ctx)
        if arm == "phys": return f_phys(x)
        fpo = f_port(P, x)
        if arm in ("phys_hn", "phys_hn_mu", "hn", "phys_feat", "skip_feat", "skip_feat0"): return fpo
        r = res_field(P, x, ctx)
        if arm == "hyb_tf_rssm":
            return tf_fuse(P, x, f_phys(x), fpo - f_phys(x), r)
        if arm == "hyb_gate_mlp":
            return fpo + jax.nn.sigmoid(x @ P["gate"]["w"] + P["gate"]["b"]) * r
        return fpo + r

    def rk4(P, x, ctx):
        k1 = field(P, x, ctx); k2 = field(P, x + dt / 2 * k1, ctx); k3 = field(P, x + dt / 2 * k2, ctx); k4 = field(P, x + dt * k3, ctx)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def init_state(P, x0):
        if not has_rssm: return (jnp.zeros(1), jnp.zeros(1))
        h0 = jnp.tanh(mlp2f(P["rssm"]["init"], x0)); return (h0, jnp.zeros(ZD))

    def step_prior(P, st, x, key, sample):
        if not has_rssm: return (st, rk4(P, x, None), None)
        h, z = st; h2 = gru(P["rssm"]["gru"], h, z)
        pri = unimix(mlp2f(P["rssm"]["prior"], h2)); z2 = sample_z(pri, key, sample)
        return ((h2, z2), rk4(P, x, (h2, z2)), pri)

    def rollout(P, x0, T, key=None, sample=False):
        keys = jax.random.split(key if key is not None else jax.random.PRNGKey(0), T)

        def body(c, k):
            st, x = c; st2, x2, _ = step_prior(P, st, x, k, sample); return (st2, x2), x2
        return jax.lax.scan(body, (init_state(P, x0), x0), keys)[1]

    def teacher_forced(P, xs, key):
        """xs (H+1, d) observed; returns (pred (H,d), kl_dyn, kl_rep) with posterior z conditioned on the observation."""
        keys = jax.random.split(key, xs.shape[0] - 1)
        st0 = init_state(P, xs[0])

        def body(st, inp):
            x_t, x_next, k = inp; h, z = st; h2 = gru(P["rssm"]["gru"], h, z)
            pri = unimix(mlp2f(P["rssm"]["prior"], h2))
            post = unimix(mlp2f(P["rssm"]["post"], jnp.concatenate([h2, x_next])))
            z2 = sample_z(post, k, True)
            pred = rk4(P, x_t, (h2, z2))
            sg = jax.lax.stop_gradient
            kl_dyn = jnp.maximum(1.0, kl_cat(sg(post), pri)); kl_rep = jnp.maximum(1.0, kl_cat(post, sg(pri)))
            return (h2, z2), (pred, kl_dyn, kl_rep)
        _, (pred, kd, kr) = jax.lax.scan(body, st0, (xs[:-1], xs[1:], keys))
        return pred, kd.mean(), kr.mean()

    return dict(rollout=rollout, teacher_forced=teacher_forced, has_rssm=has_rssm, field=field, init_state=init_state,
                f_port=f_port, f_phys=f_phys, res_field=res_field)


# ---------------------------------------------------------------- experiment ---------------------------------------
def run_one(task, arm, n_traj, noise, seed, steps, test, lam_res=0.0, hid_scale=1.0, xin_override=None):
    rng = np.random.default_rng(seed)
    x0 = task.ic(rng, n_traj); traj = task.simulate(x0, task.traj_len)
    sd = traj.reshape(-1, 2 * task.n).std(0) + 1e-6
    obs = traj + noise * sd * rng.standard_normal(traj.shape)
    starts = np.arange(0, task.traj_len - TRAIN_H + 1)
    W = np.stack([obs[:, s:s + TRAIN_H + 1] for s in starts], 1).reshape(-1, TRAIN_H + 1, 2 * task.n)
    W = jnp.asarray(W, jnp.float32); sd_j = jnp.asarray(sd, jnp.float32)
    P = init_params(arm, task, jax.random.PRNGKey(seed)); M = make_model(arm, task)
    n_par = int(sum(x.size for x in jax.tree_util.tree_leaves(P)))
    persist = arm == "phys"
    if persist:
        out = dict(params=0, final_train_loss=float("nan"))
    else:
        sched = optax.cosine_decay_schedule(2e-3, steps)
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched)); ost = opt.init(P)

        def loss(P, xs, key):
            k1, k2 = jax.random.split(key)
            imag = M["rollout"](P, xs[0], TRAIN_H, k1, True)
            l = jnp.mean(((imag - xs[1:]) / sd_j) ** 2)
            if M["has_rssm"]:
                pred, kd, kr = M["teacher_forced"](P, xs, k2)
                l = l + jnp.mean(((pred - xs[1:]) / sd_j) ** 2) + 0.5 * 0.01 * kd + 0.1 * 0.01 * kr
            if lam_res > 0 and arm.startswith("hyb_") and arm not in ("hyb_tf_rssm", "hyb_gate_mlp"):
                ctx = (M["init_state"](P, xs[0])[0], jnp.zeros(ZD)) if M["has_rssm"] else None
                r = M["res_field"](P, xs[0], ctx); fp = M["f_port"](P, xs[0])
                l = l + lam_res * jnp.sum(r ** 2) / (jnp.sum(fp ** 2) + 1e-3)
            return l

        @jax.jit
        def train(P, ost, key):
            def body(c, k):
                P, ost = c; kb, kl = jax.random.split(k)
                idx = jax.random.randint(kb, (64,), 0, W.shape[0])
                l, g = jax.value_and_grad(lambda P: jnp.mean(jax.vmap(lambda xs, kk: loss(P, xs, kk))(W[idx], jax.random.split(kl, 64))))(P)
                u, ost = opt.update(g, ost, P); return (optax.apply_updates(P, u), ost), l
            (P, ost), ls = jax.lax.scan(body, (P, ost), jax.random.split(key, steps)); return P, ls
        P, ls = train(P, ost, jax.random.PRNGKey(seed + 1))
        out = dict(params=n_par, final_train_loss=float(ls[-20:].mean()))
    for split, (x_te, truth) in test.items():
        xin = x_te + noise * sd * np.random.default_rng(seed + 7).standard_normal(x_te.shape)
        if xin_override is not None and split in xin_override: xin = xin_override[split]
        pred = np.asarray(jax.jit(jax.vmap(lambda x: M["rollout"](P, x, task.test_h)))(jnp.asarray(xin, jnp.float32)))
        out[split] = pb.metrics(task, pred, truth, sd)
        out[split]["rmse_mean"] = float(np.median(np.mean(np.sqrt((((pred - truth) / sd) ** 2).mean(-1)), 1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--tasks", nargs="+", default=["kepler_pert", "kepler_wrongmu", "fall_drag", "pendulum"])
    ap.add_argument("--arms", nargs="+", default=["mlp", "hn", "phys", "phys_hn", "rssm", "hyb_sum_mlp", "hyb_sum_rssm", "hyb_tf_rssm"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[64, 512]); ap.add_argument("--noises", nargs="+", type=float, default=[0.0, 0.02])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2]); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lam-res", type=float, default=0.0)
    ap.add_argument("--hid", type=int, default=64); ap.add_argument("--dh", type=int, default=32)
    ap.add_argument("--tag", default="")
    ap.add_argument("--v3", action="store_true"); ap.add_argument("--ng", type=int, default=4); ap.add_argument("--nc", type=int, default=8)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    global HID, DH, V3, NG, NC, ZD
    HID, DH, V3, NG, NC = a.hid, a.dh, a.v3, a.ng, a.nc; ZD = NG * NC
    print("backend", jax.default_backend(), flush=True)
    res = []; t0 = time.time()
    for tn in a.tasks:
        task = HTask(tn); rng = np.random.default_rng(999); test = {}
        for split, ood in (("id", False), ("ood", True)):
            x0 = task.ic(rng, 64, ood); test[split] = (x0, task.simulate(x0, task.test_h)[:, 1:])
        sd_ref = task.simulate(task.ic(np.random.default_rng(5), 256), task.traj_len).reshape(-1, 2 * task.n).std(0) + 1e-6
        for split in test:
            pers = np.repeat(test[split][0][:, None], task.test_h, 1)
            m = pb.metrics(task, pers, test[split][1], sd_ref)
            m["rmse_mean"] = float(np.median(np.mean(np.sqrt((((pers - test[split][1]) / sd_ref) ** 2).mean(-1)), 1)))
            res.append(dict(task=tn, arm="persistence", n_traj=0, noise=0.0, seed=0, params=0, **{split: m}))
        for arm in a.arms:
            for n_traj in a.sizes:
                for noise in a.noises:
                    for seed in a.seeds:
                        try:
                            r = run_one(task, arm, n_traj, noise, seed, a.steps, test, a.lam_res)
                        except Exception as e:
                            import traceback; traceback.print_exc(); r = dict(error=repr(e))
                        r.update(task=tn, arm=arm, n_traj=n_traj, noise=noise, seed=seed, lam_res=a.lam_res, hid=a.hid, dh=a.dh, v3=a.v3, ng=a.ng, nc=a.nc, steps=a.steps); res.append(r)
                        if "error" not in r:
                            print(f"{tn:14s} {arm:13s} N={n_traj:4d} nz={noise:.2f} s={seed} par={r['params']:6d} "
                                  f"ID mean={r['id']['rmse_mean']:.3f} end={r['id']['rmse_end']:.3f} | OOD mean={r['ood']['rmse_mean']:.3f} "
                                  f"t={time.time()-t0:.0f}s", flush=True)
                json.dump(res, open(f"{a.out}/hybrid_results.json", "w"))
    json.dump(res, open(f"{a.out}/hybrid_results.json", "w"))


if __name__ == "__main__":
    main()
