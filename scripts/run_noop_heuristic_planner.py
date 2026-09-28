"""Operationalises the operator's 2026-09-27 proposal without an RL training loop (see docs/core/STATE.yaml
TASK-20260927-009 for the critical review this design answers).

A noop-only prediction is the same value added to every action's predicted outcome, so it cannot by itself rank actions
(argmax is invariant to a term that does not depend on the action). It can only matter combined with something that DOES
depend on the action: here, the existing hand-specified CRAFTAX_RESOURCE_EFFECTS table (approximate game mechanics, not
learned, not the simulator). The test: does adding the world model's anticipation of the noop future to that known-effects
table -- i.e. "which vital is about to become the bottleneck" -- change which action looks best, and does that change how
long the player survives?

Arms (deterministic decision rules, no gradient-based policy; evaluated on real Craftax-Classic episodes, T=250 cap):
  random          : uniform random action (floor reference)
  noop_always     : constant no-op (how long does the player survive doing nothing)
  effects_only    : argmax_a min_vital(vitals_t + effects[a])                              -- current status only, no WM
  effects_wm_h1   : argmax_a min_vital(vitals_t + effects[a] + noop_pred_delta_h1)          -- +1-step anticipation
  effects_wm_h8   : argmax_a min_vital(vitals_t + effects[a] + noop_pred_delta_h8)          -- +8-step anticipation (tests
                    the operator's "assume the predicted t+1 is the next t" idea at a longer horizon; known to be less
                    accurate per the h8 skill number, so this also tests the compounding-error risk)
Training-procedure ablation (operator: "update W after one full episode, not after each inference"):
  pooled    : shuffled minibatches across all episodes (what earlier Stage A used)
  episodic  : one gradient step per whole episode (its own steps only), episodes visited in random order
Bootstrap ("retrain the WM on episodes where the player lives longer"): collect episodes under the best heuristic found,
retrain a fresh WM on THAT data only, recompute the heuristic, re-evaluate survival and held-out prediction error.

Caveat stated up front: CRAFTAX_RESOURCE_EFFECTS is a static approximation (ignores whether an action can currently
succeed, e.g. "drink" with no water in view) and gives zero effect on health for every action (health only reacts to
starvation/damage), so the effects_only baseline is a fairly weak floor by construction -- if the WM arms cannot even beat
that weak floor, that is a strong negative result, not a strong positive one for the WM if they do.
"""
import argparse, json, os, sys, time
sys.path.insert(0, ".")
import jax, jax.numpy as jnp, numpy as np, optax
from src.environment.craftax_env_adapter import (
    CraftaxEnvAdapter, CRAFTAX_RESOURCE_EFFECTS, NUM_ACHIEVEMENTS, calculate_crafter_score, masked_achievements,
)
from src.environment.craftax_obs_adapter import obs_to_features, BASE_DIM

HORIZONS = (1, 2, 4, 8)
EFFECTS = jnp.array(CRAFTAX_RESOURCE_EFFECTS, jnp.float32)[:, :4]  # (17,4) known action effect on health,food,drink,energy
N_ACT = EFFECTS.shape[0]


def vit(s):
    return jnp.array([s.player_health, s.player_food, s.player_drink, s.player_energy], jnp.float32)


# ---------------------------------------------------------------- noop model (raw vitals units, no output standardisation)
def init_model(key, hidden=256, in_dim=BASE_DIM, n_h=len(HORIZONS)):
    k1, k2, k3 = jax.random.split(key, 3)
    return dict(w1=jax.random.normal(k1, (in_dim, hidden)) / jnp.sqrt(in_dim), b1=jnp.zeros(hidden),
                w2=jax.random.normal(k2, (hidden, hidden)) / jnp.sqrt(hidden), b2=jnp.zeros(hidden),
                w3=jax.random.normal(k3, (hidden, 4 * n_h)) * 0.01, b3=jnp.zeros(4 * n_h),
                mu=jnp.zeros(in_dim), sd=jnp.ones(in_dim))


def predict_delta(p, feats):  # (BASE_DIM,) -> (n_h,4) RAW predicted vitals delta at each horizon, under noop
    h = (feats - p["mu"]) / p["sd"]
    h = jax.nn.gelu(h @ p["w1"] + p["b1"]); h = jax.nn.gelu(h @ p["w2"] + p["b2"])
    return (h @ p["w3"] + p["b3"]).reshape(len(HORIZONS), 4)


# ---------------------------------------------------------------- behaviour policies (all jit-traceable)
def act_random(feats, v0, key, params):
    return jax.random.randint(key, (), 0, N_ACT)


def act_noop(feats, v0, key, params):
    return jnp.int32(0)


def act_effects_only(feats, v0, key, params):
    good = jnp.min(v0[None, :] + EFFECTS, axis=-1)
    return jnp.argmax(good)


def make_act_effects_wm(h_idx):
    def f(feats, v0, key, params):
        d = predict_delta(params[0], feats)[h_idx]
        good = jnp.min(v0[None, :] + EFFECTS + d[None, :], axis=-1)
        return jnp.argmax(good)
    return f


ARMS = dict(random=act_random, noop_always=act_noop, effects_only=act_effects_only,
            effects_wm_h1=make_act_effects_wm(0), effects_wm_h8=make_act_effects_wm(3))


# ---------------------------------------------------------------- data collection (also serves as heuristic evaluation)
def gen_data(n_env, steps, seed, act_fn, act_params):
    a = CraftaxEnvAdapter(); env, P = a.raw_env, a.raw_env.default_params

    def one(key):
        kr, kw = jax.random.split(key)
        obs, st = env.reset(kr, P)

        def body(c, t):
            obs, st, alive, ach = c
            k = jax.random.fold_in(kw, t); ka, kn, ks = jax.random.split(k, 3)
            feats = obs_to_features(obs); v0 = vit(st)

            def nb(cc, i):
                o, s, al = cc
                o, s2, r, d, _ = env.step(jax.random.fold_in(kn, i), s, 0, P)
                al = al * (1 - d.astype(jnp.float32)); return (o, s2, al), (vit(s2), al)
            _, (fut, ok) = jax.lax.scan(nb, (obs, st, alive), jnp.arange(HORIZONS[-1]))
            sel = jnp.array([h - 1 for h in HORIZONS])
            act = act_fn(feats, v0, ka, act_params)
            o2, s2, r, d, _ = env.step(ks, st, act, P)
            ach2 = jnp.maximum(ach, masked_achievements(s2.achievements, d, alive))  # not the auto-reset life
            n_alive = alive * (1.0 - d.astype(jnp.float32))
            return (o2, s2, n_alive, ach2), (feats, v0, fut[sel], ok[sel] * alive, alive)
        (_, _, _, ach), out = jax.lax.scan(
            body, (obs, st, jnp.array(1.0), jnp.zeros(NUM_ACHIEVEMENTS)), jnp.arange(steps))
        return out, ach
    outs, ach = jax.jit(jax.vmap(one))(jax.random.split(jax.random.PRNGKey(seed), n_env))
    return [np.asarray(x) for x in outs], np.asarray(ach)  # (feats,v0,fut,ok,alive) each (E,T,...); ach (E,NUM_ACH)


def build_targets(feats, v0, fut, ok):  # -> flat (N,BASE_DIM), (N,n_h,4) delta, (N,n_h) mask, split by episode
    E, T = feats.shape[:2]
    delta = fut - v0[:, :, None, :]
    return feats, delta, ok  # keep (E,T,...) shape; caller flattens/splits as needed


def fit_pooled(params, feats, delta, mask, steps, seed, batch=512, lr=1e-3):
    fl = lambda x: x.reshape(-1, *x.shape[2:])
    f, d, m = jnp.asarray(fl(feats)), jnp.asarray(fl(delta)), jnp.asarray(fl(mask))
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr)); st = opt.init(params)

    def loss_fn(p, idx):
        pred = jax.vmap(lambda x: predict_delta(p, x))(f[idx])
        err = ((pred - d[idx]) ** 2).mean(-1) * m[idx]
        return err.sum() / jnp.maximum(m[idx].sum(), 1.0)

    @jax.jit
    def step(p, st, idx):
        l, g = jax.value_and_grad(loss_fn)(p, idx); u, st = opt.update(g, st, p); return optax.apply_updates(p, u), st, l
    rng = np.random.default_rng(seed); n = len(f)
    for s in range(steps):
        params, st, l = step(params, st, jnp.asarray(rng.integers(0, n, batch)))
    return params, float(l)


def fit_episodic(params, feats, delta, mask, epochs, seed, lr=1e-3):
    """One gradient step per whole episode (operator: 'update W after one full episode, not after each inference')."""
    f, d, m = jnp.asarray(feats), jnp.asarray(delta), jnp.asarray(mask)
    E = f.shape[0]
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr)); st = opt.init(params)

    def loss_fn(p, e):
        pred = jax.vmap(lambda x: predict_delta(p, x))(f[e])
        err = ((pred - d[e]) ** 2).mean(-1) * m[e]
        return err.sum() / jnp.maximum(m[e].sum(), 1.0)

    @jax.jit
    def step(p, st, e):
        l, g = jax.value_and_grad(loss_fn)(p, e); u, st = opt.update(g, st, p); return optax.apply_updates(p, u), st, l
    rng = np.random.default_rng(seed); l = 0.0
    for ep in range(epochs):
        for e in rng.permutation(E):
            params, st, l = step(params, st, jnp.asarray(int(e)))
    return params, float(l)


def eval_mse(params, feats, delta, mask):  # held-out per-horizon MSE, raw vitals^2 units
    fl = lambda x: x.reshape(-1, *x.shape[2:])
    f, d, m = fl(feats), fl(delta), fl(mask)
    outs = []
    for i in range(0, len(f), 4096):
        pred = jax.vmap(lambda x: predict_delta(params, x))(jnp.asarray(f[i:i + 4096]))
        outs.append((((pred - d[i:i + 4096]) ** 2).mean(-1) * m[i:i + 4096], m[i:i + 4096]))
    se = np.concatenate([np.asarray(o[0]) for o in outs]); mm = np.concatenate([np.asarray(o[1]) for o in outs])
    return (se.sum(0) / np.maximum(mm.sum(0), 1)).tolist()


def survival_stats(alive, ach):
    lens = alive.sum(1)
    rates = [float(np.mean(ach[:, i]) * 100.0) for i in range(NUM_ACHIEVEMENTS)]
    return dict(mean_len=float(lens.mean()), std_len=float(lens.std()), median_len=float(np.median(lens)),
                crafter_score=calculate_crafter_score(rates), n=int(len(lens)))


def split(feats, v0, fut, ok, ntr_frac=0.8):
    E = feats.shape[0]; ntr = int(E * ntr_frac)
    delta = fut - v0[:, :, None, :]
    return (feats[:ntr], delta[:ntr], ok[:ntr]), (feats[ntr:], delta[ntr:], ok[ntr:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=250); ap.add_argument("--eval-envs", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--train-steps", type=int, default=4000)
    ap.add_argument("--episodic-epochs", type=int, default=3)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); res = {}; t0 = time.time()
    print("backend", jax.default_backend(), flush=True)

    print("[phase1] collecting random-policy data for WM training...", flush=True)
    (feats, v0, fut, ok, alive), ach = gen_data(a.envs, a.steps, a.seed, act_random, ())
    tr, te = split(feats, v0, fut, ok)
    print(f"  data {feats.shape} t={time.time()-t0:.0f}s", flush=True)

    p_pooled, l_pooled = fit_pooled(init_model(jax.random.PRNGKey(a.seed + 1)), *tr, a.train_steps, a.seed)
    p_epis, l_epis = fit_episodic(init_model(jax.random.PRNGKey(a.seed + 2)), *tr, a.episodic_epochs, a.seed)
    res["wm_training_ablation"] = dict(
        pooled_train_loss=l_pooled, pooled_heldout_mse=eval_mse(p_pooled, *te),
        episodic_train_loss=l_epis, episodic_heldout_mse=eval_mse(p_epis, *te), horizons=list(HORIZONS))
    print("  pooled held-out MSE  ", res["wm_training_ablation"]["pooled_heldout_mse"], flush=True)
    print("  episodic held-out MSE", res["wm_training_ablation"]["episodic_heldout_mse"], flush=True)

    print("[phase2] evaluating heuristics on real episodes...", flush=True)
    res["round1"] = {}
    round1_data = {}
    for name, fn in ARMS.items():
        params = (p_pooled,) if name.startswith("effects_wm") else ()
        (f2, v2, fu2, ok2, al2), ach2 = gen_data(a.eval_envs, a.steps, a.seed + 100, fn, params)
        res["round1"][name] = survival_stats(al2, ach2)
        round1_data[name] = (f2, v2, fu2, ok2, al2)
        print(f"  {name:14s} len={res['round1'][name]['mean_len']:.1f}+-{res['round1'][name]['std_len']:.1f} "
              f"median={res['round1'][name]['median_len']:.0f} crafter={res['round1'][name]['crafter_score']:.2f} "
              f"t={time.time()-t0:.0f}s", flush=True)
    # secondary: effects_wm_h1 heuristic driven by the EPISODIC-trained model (does the training procedure change the outcome?)
    (f3, v3, fu3, ok3, al3), ach3 = gen_data(a.eval_envs, a.steps, a.seed + 100, ARMS["effects_wm_h1"], (p_epis,))
    res["round1"]["effects_wm_h1_episodic_model"] = survival_stats(al3, ach3)
    print(f"  effects_wm_h1(episodic model) len={res['round1']['effects_wm_h1_episodic_model']['mean_len']:.1f} "
          f"t={time.time()-t0:.0f}s", flush=True)

    best = max((n for n in res["round1"] if n != "random"), key=lambda n: res["round1"][n]["mean_len"])
    print(f"[bootstrap] best arm = {best} (mean_len={res['round1'][best]['mean_len']:.1f}); retraining WM on its episodes", flush=True)
    if best in round1_data:
        f2, v2, fu2, ok2, al2 = round1_data[best]
    else:
        f2, v2, fu2, ok2, al2 = f3, v3, fu3, ok3, al3
    tr2, te2 = split(f2, v2, fu2, ok2)
    p_boot, l_boot = fit_pooled(init_model(jax.random.PRNGKey(a.seed + 3)), *tr2, a.train_steps, a.seed)
    res["bootstrap"] = dict(source_arm=best, train_loss=l_boot, heldout_mse=eval_mse(p_boot, *te2))
    print("  bootstrap held-out MSE (own distribution)", res["bootstrap"]["heldout_mse"], flush=True)
    # cross-eval: does the bootstrapped model also fit better on the ORIGINAL random-policy held-out set, or only its own?
    res["bootstrap"]["heldout_mse_on_random_data"] = eval_mse(p_boot, *te)
    (f4, v4, fu4, ok4, al4), ach4 = gen_data(a.eval_envs, a.steps, a.seed + 200, ARMS["effects_wm_h1"], (p_boot,))
    res["round2_effects_wm_h1"] = survival_stats(al4, ach4)
    print(f"  round2 effects_wm_h1(bootstrapped model) len={res['round2_effects_wm_h1']['mean_len']:.1f} "
          f"t={time.time()-t0:.0f}s", flush=True)

    json.dump(res, open(f"{a.out}/result.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
