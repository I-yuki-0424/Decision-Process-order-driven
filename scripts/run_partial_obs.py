"""Partial observation test: only position q is observed (noisy); momentum p must be estimated before rolling out dynamics.

Tests the role a recurrent/DreamerV3-style component is expected to play (state estimation under partial observability),
decoupled from the dynamics model:
  estimators: fd     = finite difference of the last two noisy q observations (no learning)
              learned= MLP filter over the last K noisy q observations, trained (with true p from simulation as the
                       training label) -- an explicit stand-in for a learned recurrent filter
              oracle = true state plus the same relative noise (the full-observation protocol used so far)
  dynamics  : phys_hn | hn | mlp | hyb_sum_mlp   (trained as before on full noisy states)
Metric: mean nRMSE of the predicted q,p rollout (state-space error) vs the true future, ID initial conditions.
"""
import argparse, json, os, sys, time
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import jax, jax.numpy as jnp, numpy as np, optax
import run_hybrid_worldmodel as hw
import run_physics_benchmark as pb

K = 6  # observation window (K+1 noisy positions)


def make_windows(task, rng, m, noise, sd):
    x0 = task.ic(rng, m)
    tr = task.simulate(x0, K + task.test_h)                          # (m, K+1+T, 2n)
    q = tr[..., :task.n]
    qo = q[:, :K + 1] + noise * sd[:task.n] * rng.standard_normal(q[:, :K + 1].shape)
    return qo, tr[:, K], tr[:, K + 1:]                               # obs window, true x at end of window, future truth


def train_filter(task, noise, sd, seed, steps=3000):
    rng = np.random.default_rng(seed + 100)
    qo, x_end, _ = make_windows(task, rng, 4000, noise, sd)
    sdj = jnp.asarray(sd, jnp.float32)
    X = jnp.asarray((qo - qo[:, -1:]).reshape(len(qo), -1) / sd[:task.n].mean(), jnp.float32)   # differences w.r.t. last obs
    Y = jnp.asarray((x_end[:, task.n:]) / sd[task.n:], jnp.float32)                              # target: p (standardised)
    P = hw.mlp2(jax.random.PRNGKey(seed), X.shape[1], task.n)
    opt = optax.adam(2e-3); st = opt.init(P)

    @jax.jit
    def train(P, st, key):
        def body(c, k):
            P, st = c; idx = jax.random.randint(k, (256,), 0, X.shape[0])
            l, g = jax.value_and_grad(lambda P: jnp.mean((jax.vmap(lambda x: hw.mlp2f(P, x))(X[idx]) - Y[idx]) ** 2))(P)
            u, st = opt.update(g, st, P); return (optax.apply_updates(P, u), st), l
        return jax.lax.scan(body, (P, st), jax.random.split(key, steps))[0]
    P, _ = train(P, st, jax.random.PRNGKey(seed + 1))
    return lambda qo_: np.asarray(jax.vmap(lambda x: hw.mlp2f(P, x))(jnp.asarray((qo_ - qo_[:, -1:]).reshape(len(qo_), -1) / sd[:task.n].mean(), jnp.float32))) * sd[task.n:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--tasks", nargs="+", default=["pendulum", "fall_drag", "kepler_pert"])
    ap.add_argument("--arms", nargs="+", default=["phys_hn", "hn", "mlp", "hyb_sum_mlp"])
    ap.add_argument("--noises", nargs="+", type=float, default=[0.02, 0.05]); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--n-traj", type=int, default=64)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); res = []; t0 = time.time()
    for tn in a.tasks:
        task = hw.HTask(tn)
        sd = task.simulate(task.ic(np.random.default_rng(5), 256), task.traj_len).reshape(-1, 2 * task.n).std(0) + 1e-6
        for noise in a.noises:
            for seed in a.seeds:
                filt = train_filter(task, noise, sd, seed)
                rng = np.random.default_rng(999)
                qo, x_end, fut = make_windows(task, rng, 64, noise, sd)
                rr = np.random.default_rng(seed + 7)
                est = {
                    "oracle": x_end + noise * sd * rr.standard_normal(x_end.shape),
                    "fd": np.concatenate([qo[:, -1], (qo[:, -1] - qo[:, -2]) / task.dt], -1),
                    "learned": np.concatenate([qo[:, -1], filt(qo)], -1),
                }
                test = {"id": (x_end, fut)}
                for arm in a.arms:
                    for e, xin in est.items():
                        r = hw.run_one(task, arm, a.n_traj, 0.02, seed, a.steps, test, xin_override={"id": xin.astype(np.float32)})
                        r.update(task=tn, arm=arm, estimator=e, obs_noise=noise, seed=seed); res.append(r)
                        print(f"{tn:11s} nz={noise:.2f} s={seed} {arm:12s} est={e:8s} ID mean={r['id']['rmse_mean']:.3f} t={time.time()-t0:.0f}s", flush=True)
                json.dump(res, open(f"{a.out}/partial_results.json", "w"))
    json.dump(res, open(f"{a.out}/partial_results.json", "w"))


if __name__ == "__main__":
    main()
