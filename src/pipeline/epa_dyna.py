"""Dyna-style PPO (TASK-20261125-041): extra on-policy PPO updates on rollouts imagined by a one-step model.

Motivation (TASK-026/028): at the EP-A budget the GRU-TP learner is limited by samples (2M real steps would clear G1.2),
and it cannot reuse real data more than ~4 passes (more epochs / replay were harmful). Dyna supplies data that is fresh
for the current policy without new environment steps.

Per update:
  1. the real PPO update of `Trainer` (unchanged maths; the real batch is also kept for the model);
  2. the model is trained on executed transitions only (ring buffer of the last `buf_k` real rollouts; P4, P10);
  3. from `im_envs` start points drawn from the real rollout (observation + the GRU state the policy had there) the
     CURRENT policy acts for `im_h` steps inside the model; a sampled `done` restarts that row at another real start point;
  4. PPO on the imagined batch (GAE with the policy's own value, `im_epochs` x `im_minibatches`). The imagined loss is
     per step with the stored GRU state (no backprop through the recurrence); real data still trains the recurrence.

model = "learned": a small conv/MLP model of the symbolic observation (7x9 map of 17 block classes + 4 mob flags, inventory
        and intrinsics as 10-class values, direction, light, sleeping) plus reward and termination. Counted in params_total.
model = "oracle":  the real simulator stepped from copies of the visited env states. Simulator access at decision time for
        training data (P3/P10): an UPPER-BOUND CONTROL that is never EP-A and never gate evidence; it tells how much the
        learned model leaves on the table at the same imagination settings.
Nothing here is game knowledge (P5): the model learns every effect from data; the observation layout is the env's format.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.model.epa_policies import N_ACT, make_optimizer
from src.pipeline.epa_harness import PPOConfig, Trainer, _vn_stats, env_step

GH, GW, NB, NM = 7, 9, 17, 4          # map view 7 x 9, 17 block classes, 4 mob flags (Craftax-Classic symbolic layout)
NCELL, TILE = GH * GW, NB + NM        # 63 cells x 21
MAPD = NCELL * TILE                   # 1323
NINV, NINT, NV = 12, 4, 10            # inventory 12 and intrinsics 4 values, each an integer 0..9 stored as value / 10
REW_BINS = jnp.linspace(-1.0, 2.0, 31)  # model v2: two-hot reward support (step 0.1; DreamerV3-style encoding, no game knowledge)


# ----------------------------------------------------------------------------------------------------------------------
# Learned one-step model
# ----------------------------------------------------------------------------------------------------------------------
def _dense(key, i, o, scale=1.0):
    return {"w": jax.random.normal(key, (i, o)) * scale / jnp.sqrt(i), "b": jnp.zeros((o,))}


def _conv(key, i, o, scale=1.0):
    return {"w": jax.random.normal(key, (3, 3, i, o)) * scale / jnp.sqrt(9 * i), "b": jnp.zeros((o,))}


def _ap(p, x):
    return x @ p["w"] + p["b"]


def _cv(p, x):   # x (B, 7, 9, C), SAME padding
    y = jax.lax.conv_general_dilated(x, p["w"], (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return y + p["b"]


def init_model(key, c=64, hidden=256, version=1):
    """version 1: TASK-041 S1 model. version 2 (after the S1 diagnostic): the map and mob losses are SUMMED over the 63 cells
    (v1 averaged them, so a changed cell weighed 1/63 of an inventory slot and the model learned to copy the view), and reward
    is a two-hot categorical over REW_BINS (v1 regressed the mean and was worse than predicting 0). v2 is marked by the 'rew2' head."""
    k = iter(jax.random.split(key, 13))
    vec_in = 22 + N_ACT
    extra = {"rew2": _dense(jax.random.fold_in(key, 0xBE), hidden, REW_BINS.shape[0], 0.1)} if version == 2 else {}
    return extra | {"v": _dense(next(k), vec_in, c), "c0": _conv(next(k), TILE + c, c), "c1": _conv(next(k), c, c),
            "c2": _conv(next(k), c, c), "g": _dense(next(k), NCELL * c + c, hidden), "gm": _dense(next(k), hidden, c),
            "map": _dense(next(k), 2 * c, TILE, 0.1),
            "vec": _dense(next(k), hidden, NINV * NV + NINT * NV + 4 + 1 + 1 + 1 + 1, 0.1)}


def model_forward(m, obs, act):
    """obs (B, 1345), act (B,) -> dict of logits / predictions for the next observation, reward and termination."""
    B = obs.shape[0]
    grid = obs[:, :MAPD].reshape(B, GH, GW, TILE)
    v = jax.nn.gelu(_ap(m["v"], jnp.concatenate([obs[:, MAPD:], jax.nn.one_hot(act, N_ACT)], -1)))
    x = jax.nn.gelu(_cv(m["c0"], jnp.concatenate([grid, jnp.broadcast_to(v[:, None, None], (B, GH, GW, v.shape[-1]))], -1)))
    x = x + jax.nn.gelu(_cv(m["c1"], x))
    x = x + jax.nn.gelu(_cv(m["c2"], x))
    g = jax.nn.gelu(_ap(m["g"], jnp.concatenate([x.reshape(B, -1), v], -1)))
    gm = jnp.broadcast_to(_ap(m["gm"], g)[:, None, None], x.shape)
    mp = _ap(m["map"], jnp.concatenate([x, gm], -1)).reshape(B, NCELL, TILE)
    vo = _ap(m["vec"], g)
    i = 0

    def take(n):
        nonlocal i
        out = vo[:, i:i + n]
        i += n
        return out

    inv, intr, dirl = take(NINV * NV).reshape(B, NINV, NV), take(NINT * NV).reshape(B, NINT, NV), take(4)
    light, sleep, rew, done = take(1)[:, 0], take(1)[:, 0], take(1)[:, 0], take(1)[:, 0]
    out = dict(block=mp[..., :NB], mob=mp[..., NB:], inv=inv, intr=intr, dir=dirl, light=light, sleep=sleep, rew=rew, done=done)
    if "rew2" in m:   # v2: categorical reward; `rew` becomes its expectation
        out["rew_logits"] = _ap(m["rew2"], g)
        out["rew"] = (jax.nn.softmax(out["rew_logits"]) * REW_BINS).sum(-1)
    return out


def _twohot(r):
    r = jnp.clip(r, REW_BINS[0], REW_BINS[-1])
    i = jnp.clip(jnp.searchsorted(REW_BINS, r, side="right") - 1, 0, REW_BINS.shape[0] - 2)
    lo, hi = REW_BINS[i], REW_BINS[i + 1]
    wh = (r - lo) / (hi - lo)
    return jax.nn.one_hot(i, REW_BINS.shape[0]) * (1 - wh)[..., None] + jax.nn.one_hot(i + 1, REW_BINS.shape[0]) * wh[..., None]


def _split_obs(o):
    B = o.shape[0]
    grid = o[:, :MAPD].reshape(B, NCELL, TILE)
    e = o[:, MAPD:]
    return dict(block=jnp.argmax(grid[..., :NB], -1), mob=grid[..., NB:], inv=jnp.clip(jnp.round(e[:, :NINV] * 10), 0, NV - 1).astype(jnp.int32),
                intr=jnp.clip(jnp.round(e[:, NINV:NINV + NINT] * 10), 0, NV - 1).astype(jnp.int32),
                dir=jnp.argmax(e[:, 16:20], -1), light=e[:, 20], sleep=e[:, 21])


def model_loss(m, obs, act, nobs, rew, done, rew_w=10.0):
    """Cross-entropy / BCE / MSE of the next observation (transitions that ended an episode are dropped: the env returns
    the first observation of a new episode there), plus reward and termination on every transition."""
    p = model_forward(m, obs, act)
    t = _split_obs(nobs)
    w = 1.0 - done.astype(jnp.float32)
    ce = lambda lg, y: -jnp.take_along_axis(jax.nn.log_softmax(lg), y[..., None], -1)[..., 0]
    bce = lambda lg, y: optax.sigmoid_binary_cross_entropy(lg, y)
    v2 = "rew2" in m
    cells = (lambda x: x.sum(-1)) if v2 else (lambda x: x.mean(-1))
    l_obs = (cells(ce(p["block"], t["block"])) + cells(bce(p["mob"], t["mob"]).sum(-1)) + ce(p["inv"], t["inv"]).sum(-1)
             + ce(p["intr"], t["intr"]).sum(-1) + ce(p["dir"], t["dir"]) + (jax.nn.sigmoid(p["light"]) - t["light"]) ** 2
             + bce(p["sleep"], t["sleep"]))
    l_obs = (l_obs * w).sum() / jnp.maximum(w.sum(), 1.0)
    l_rew = -(_twohot(rew) * jax.nn.log_softmax(p["rew_logits"])).sum(-1) if v2 else rew_w * (p["rew"] - rew) ** 2
    l_rd = (l_rew + bce(p["done"], done.astype(jnp.float32))).mean()
    return l_obs + l_rd


def model_eval(m, obs, act, nobs, rew, done):
    """Held-out quality of the model on a real rollout it has not been trained on yet (diagnostics only, never used to act or select):
    [block acc. all cells, block acc. on cells that changed, mob-flag error per obs, inventory+intrinsics exact, reward MSE,
     reward MSE of predicting 0, mean predicted reward on events r >= 0.5, predicted done prob mean, real done rate]."""
    p = model_forward(m, obs, act)
    t, t0 = _split_obs(nobs), _split_obs(obs)
    w = 1.0 - done.astype(jnp.float32)
    wd = jnp.maximum(w.sum(), 1.0)
    pb = jnp.argmax(p["block"], -1)
    hit = (pb == t["block"]).astype(jnp.float32)
    ch = (t["block"] != t0["block"]).astype(jnp.float32) * w[:, None]
    mob = jnp.abs((jax.nn.sigmoid(p["mob"]) > 0.5).astype(jnp.float32) - t["mob"]).sum((-1, -2))
    vec = ((jnp.argmax(p["inv"], -1) == t["inv"]).all(-1) & (jnp.argmax(p["intr"], -1) == t["intr"]).all(-1)).astype(jnp.float32)
    ev = (rew >= 0.5).astype(jnp.float32)
    return jnp.stack([(hit.mean(-1) * w).sum() / wd, (hit * ch).sum() / jnp.maximum(ch.sum(), 1.0), (mob * w).sum() / wd,
                      (vec * w).sum() / wd, ((p["rew"] - rew) ** 2).mean(), (rew ** 2).mean(),
                      (p["rew"] * ev).sum() / jnp.maximum(ev.sum(), 1.0), jax.nn.sigmoid(p["done"]).mean(), done.astype(jnp.float32).mean()])


def model_sample(m, obs, act, key, mode=False):
    """Next observation in the env's own 1345-d format, reward (mean) and termination (sampled). mode=True takes the most likely
    value of every observation part (argmax / p > 0.5) instead of sampling it: less imagined noise, no model-uncertainty spread."""
    p = model_forward(m, obs, act)
    B = obs.shape[0]
    k = jax.random.split(key, 7)
    cat = (lambda kk, lg: jnp.argmax(lg, -1)) if mode else (lambda kk, lg: jax.random.categorical(kk, lg))
    ber = (lambda kk, lg: (lg > 0).astype(jnp.float32)) if mode else (lambda kk, lg: jax.random.bernoulli(kk, jax.nn.sigmoid(lg)).astype(jnp.float32))
    block = jax.nn.one_hot(cat(k[0], p["block"]), NB)
    mob = ber(k[1], p["mob"])
    inv = cat(k[2], p["inv"]).astype(jnp.float32) / 10.0
    intr = cat(k[3], p["intr"]).astype(jnp.float32) / 10.0
    dirv = jax.nn.one_hot(cat(k[4], p["dir"]), 4)
    sleep = ber(k[5], p["sleep"])
    nobs = jnp.concatenate([jnp.concatenate([block, mob], -1).reshape(B, MAPD), inv, intr, dirv,
                            jax.nn.sigmoid(p["light"])[:, None], sleep[:, None]], -1)
    done = jax.random.bernoulli(k[6], jax.nn.sigmoid(p["done"]))
    rew = p["rew"]
    if "rew_logits" in p:   # v2: a sampled reward value (an imagined achievement is 0 or ~1, not a smeared mean); mode -> most likely bin
        rew = REW_BINS[cat(jax.random.fold_in(key, 7), p["rew_logits"])]
    return nobs, rew, done


# ----------------------------------------------------------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------------------------------------------------------
def _take(tree, t, n):
    return jax.tree_util.tree_map(lambda x: x[t, n], tree)


class DynaTrainer(Trainer):
    def __init__(self, arm, cfg: PPOConfig, model="learned", im_envs=256, im_h=16, im_epochs=2, im_minibatches=8,
                 im_start=0.1, buf_k=8, wm_steps=8, wm_batch=1024, wm_lr=1e-3, wm_c=64, wm_hidden=256, im_mode=False, wm_version=1):
        if model not in ("learned", "oracle"):
            raise ValueError("model must be 'learned' or 'oracle'")
        if not arm.recurrent or cfg.lr_sched != "linear":
            raise ValueError("DynaTrainer supports recurrent arms with the linear lr schedule")
        if (im_envs * im_h) % im_minibatches:
            raise ValueError("im_envs * im_h must be divisible by im_minibatches")
        self.model, self.im_envs, self.im_h, self.im_epochs, self.im_mb = model, im_envs, im_h, im_epochs, im_minibatches
        self.buf_k, self.wm_steps, self.wm_batch, self.wm_c, self.wm_hidden = buf_k, wm_steps, wm_batch, wm_c, wm_hidden
        self.im_mode, self.wm_version = im_mode, wm_version
        super().__init__(arm, cfg)
        self.im_start_u = int(im_start * self.n_updates)
        # one lr schedule over every optimiser step, real and imagined (the base class only counts the real ones)
        n_opt = max(self.n_updates * cfg.epochs * cfg.minibatches + (self.n_updates - self.im_start_u) * im_epochs * im_minibatches, 1)
        lr = optax.linear_schedule(cfg.lr, 0.0, n_opt) if cfg.anneal_lr else cfg.lr
        if cfg.warmup:
            nw = max(int(cfg.warmup * n_opt), 1)
            decay = lr if callable(lr) else (lambda c, v=lr: v)
            lr = lambda c, decay=decay, nw=nw: decay(c) * jnp.minimum((c + 1) / nw, 1.0)
        self.opt = make_optimizer(arm, lr, cfg.max_grad_norm)
        self.wm_opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(wm_lr))
        self.update = jax.jit(self._update)
        shapes = jax.eval_shape(lambda k: init_model(k, wm_c, wm_hidden, wm_version), jax.random.PRNGKey(0))
        self.model_params = 0 if model == "oracle" else int(sum(np.prod(x.shape) for x in jax.tree_util.tree_leaves(shapes)))
        self.im_steps_total = (self.n_updates - self.im_start_u) * im_envs * im_h

    def init(self, key):
        st = super().init(key)
        if self.model == "learned":
            K, T, N = self.buf_k, self.cfg.num_steps, self.cfg.num_envs
            wm = init_model(jax.random.fold_in(key, 0xD1A), self.wm_c, self.wm_hidden, self.wm_version)
            st["wm"], st["wm_opt"] = wm, self.wm_opt.init(wm)
            od = st["obs"].shape[-1]
            st["wb"] = dict(obs=jnp.zeros((K, T, N, od), jnp.bfloat16), nobs=jnp.zeros((K, T, N, od), jnp.bfloat16),
                            act=jnp.zeros((K, T, N), jnp.int32), rew=jnp.zeros((K, T, N)), done=jnp.zeros((K, T, N), jnp.bool_))
        return st

    # real rollout of the base class, plus the env state before every step (oracle only) and a copy of the batch for the model
    def _rollout(self, params, st, key):
        arm, N = self.arm, self.cfg.num_envs
        oracle = self.model == "oracle"

        def body(c, k):
            env_st, obs, carry, ep_ret = c
            k1, k2 = jax.random.split(k)
            logits, value, caux = arm.step(params, carry, obs)
            a = jax.random.categorical(k1, logits)
            logp = jnp.take_along_axis(jax.nn.log_softmax(logits), a[:, None], 1)[:, 0]
            nobs, env_st2, r, d = env_step(jax.random.split(k2, N), env_st, a)
            ep2 = ep_ret + r
            tr = dict(obs=obs, carry=carry, act=a, logp=logp, val=value, rew=r, done=d, nobs=nobs, fin_ret=jnp.where(d, ep2, 0.0))
            if oracle:
                tr["env"] = env_st
            return (env_st2, nobs, arm.advance(caux, obs, a, d), jnp.where(d, 0.0, ep2)), tr

        (env_st, obs, carry, ep_ret), traj = jax.lax.scan(
            body, (st["env"], st["obs"], st["carry"], st["ep_ret"]), jax.random.split(key, self.cfg.num_steps))
        _, last_v, _ = arm.step(params, carry, obs)
        self._traj = traj   # read by _update within the same trace
        envs = traj.pop("env", None)
        self._traj_env = envs
        return env_st, obs, carry, ep_ret, traj, last_v

    # -- model training on executed transitions ------------------------------------------------------------------------
    def _train_model(self, st, traj, key):
        wb = st["wb"]
        slot = st["u"] % self.buf_k
        put = lambda b, x: jax.lax.dynamic_update_index_in_dim(b, x.astype(b.dtype), slot, 0)
        wb = {k: put(wb[k], traj[k]) for k in wb}
        filled = jnp.minimum(st["u"] + 1, self.buf_k)
        T, N = self.cfg.num_steps, self.cfg.num_envs

        def step(c, k):
            wm, o = c
            ka, kb, kc = jax.random.split(k, 3)
            i = jax.random.randint(ka, (self.wm_batch,), 0, filled)
            t = jax.random.randint(kb, (self.wm_batch,), 0, T)
            n = jax.random.randint(kc, (self.wm_batch,), 0, N)
            mb = {kk: wb[kk][i, t, n] for kk in wb}
            loss, g = jax.value_and_grad(model_loss)(wm, mb["obs"].astype(jnp.float32), mb["act"], mb["nobs"].astype(jnp.float32),
                                                     mb["rew"], mb["done"])
            upd, o = self.wm_opt.update(g, o, wm)
            return (optax.apply_updates(wm, upd), o), loss

        (wm, wo), losses = jax.lax.scan(step, (st["wm"], st["wm_opt"]), jax.random.split(key, self.wm_steps))
        return dict(st, wm=wm, wm_opt=wo, wb=wb), losses.mean()

    # -- imagination ---------------------------------------------------------------------------------------------------
    def _imagine(self, params, st, traj, envs, key):
        arm, B, H = self.arm, self.im_envs, self.im_h
        T, N = self.cfg.num_steps, self.cfg.num_envs
        oracle = self.model == "oracle"

        def draw(k):
            kt, kn = jax.random.split(k)
            t, n = jax.random.randint(kt, (B,), 0, T), jax.random.randint(kn, (B,), 0, N)
            return traj["obs"][t, n], _take(traj["carry"], t, n), (_take(envs, t, n) if oracle else None)

        k0, key = jax.random.split(key)
        obs, carry, env = draw(k0)

        def body(c, k):
            obs, carry, env = c
            k1, k2, k3 = jax.random.split(k, 3)
            logits, value, caux = arm.step(params, carry, obs)
            a = jax.random.categorical(k1, logits)
            logp = jnp.take_along_axis(jax.nn.log_softmax(logits), a[:, None], 1)[:, 0]
            if oracle:
                nobs, env2, r, d = env_step(jax.random.split(k2, B), env, a)
            else:
                nobs, r, d = model_sample(st["wm"], obs, a, k2, self.im_mode)
                env2 = None
            h2 = arm.advance(caux, obs, a, d)
            ro, rc, re = draw(k3)   # a terminated row restarts at another real start point
            sel = lambda x, y: jnp.where(d.reshape((B,) + (1,) * (x.ndim - 1)), x, y)
            nobs2 = sel(ro, nobs)
            h3 = jax.tree_util.tree_map(sel, rc, h2)
            env3 = jax.tree_util.tree_map(sel, re, env2) if oracle else None
            out = dict(obs=obs, carry=carry, act=a, logp=logp, val=value, rew=r, done=d)
            return (nobs2, h3, env3), out

        (obs, carry, _), im = jax.lax.scan(body, (obs, carry, env), jax.random.split(key, H))
        _, last_v, _ = arm.step(params, carry, obs)
        return im, last_v

    def _loss_im(self, params, mb):
        logits, value, _ = self.arm.step(params, mb["carry"], mb["obs"])
        loss, aux, ex = self._ppo_terms(logits, value, mb)
        return loss, jnp.concatenate([aux, ex])

    def _im_update(self, params, opt, st, traj, envs, key):
        cfg = self.cfg
        ki, ku = jax.random.split(key)
        im, last_v = self._imagine(params, st, traj, envs, ki)
        if cfg.value_norm:   # same scale handling as the real update, with the current stats (imagined returns do not update them)
            mu, sd = _vn_stats(st["vn"])
            im, last_v = dict(im, val=im["val"] * sd + mu), last_v * sd + mu
        adv, ret = self._gae(im, last_v)
        if cfg.value_norm:
            im, ret = dict(im, val=(im["val"] - mu) / sd), (ret - mu) / sd
        if cfg.adv_norm == "batch":
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        batch = dict(im, adv=adv, ret=ret)
        batch = jax.tree_util.tree_map(lambda x: x.reshape((x.shape[0] * x.shape[1],) + x.shape[2:]), batch)
        size = self.im_envs * self.im_h

        def epoch(c, kk):
            perm = jax.random.permutation(kk, size)
            mbs = jax.tree_util.tree_map(lambda x: x[perm].reshape((self.im_mb, size // self.im_mb) + x.shape[1:]), batch)

            def mb_step(c2, mb):
                p, o = c2
                (_, aux), g = jax.value_and_grad(self._loss_im, has_aux=True)(p, mb)
                upd, o = self.opt.update(g, o, p)
                return (optax.apply_updates(p, upd), o), aux

            return jax.lax.scan(mb_step, c, mbs)

        (params, opt), aux = jax.lax.scan(epoch, (params, opt), jax.random.split(ku, self.im_epochs))
        m = aux.mean((0, 1))   # [pg, vl, ent, kl, clipfrac]
        return params, opt, jnp.stack([im["rew"].sum(0).mean(), im["done"].mean().astype(jnp.float32), m[3]])

    def _update(self, st):
        st_in = st
        st, stats = super()._update(st)        # real PPO update; self._traj / self._traj_env were set while tracing it
        traj, envs = self._traj, self._traj_env
        key, km, ki = jax.random.split(st["key"], 3)
        st = dict(st, key=key, **{k: st_in[k] for k in ("wm", "wm_opt", "wb") if k in st_in})   # the base update drops extra keys
        wl, mev = jnp.zeros(()), jnp.zeros((9,))
        if self.model == "learned":
            flat = lambda x: x.reshape((-1,) + x.shape[2:])
            mev = model_eval(st_in["wm"], *(flat(traj[k]) for k in ("obs", "act", "nobs", "rew", "done")))   # before training on it
            st, wl = self._train_model(dict(st, u=st_in["u"]), traj, km)
            st = dict(st, u=st_in["u"] + 1)
        on = st_in["u"] >= self.im_start_u
        params, opt, ist = jax.lax.cond(on, lambda: self._im_update(st["params"], st["opt"], st, traj, envs, ki),
                                        lambda: (st["params"], st["opt"], jnp.zeros((3,))))
        return dict(st, params=params, opt=opt), jnp.concatenate([stats, wl[None], ist, mev])
