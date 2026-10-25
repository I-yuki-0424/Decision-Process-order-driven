"""TASK-20261025-028: PPO with replay of the last K rollouts (data reuse at a fixed env-step budget).

Each update trains on the new rollout plus the K-1 previous ones, so every env transition takes part in K updates instead
of one. No extra env steps: the buffer only holds transitions the agent executed (P4, P10). Off-policy correction:
  * proximal policy (Hilton et al. 2021, "Batch size-invariance for policy optimization"): the PPO ratio is taken
    against the policy at the start of this update (logp_prox, value_prox from one forward pass over the buffer), not
    against the stale behaviour policy that collected an old rollout;
  * advantages and value targets: GAE with truncated importance weights (V-trace form, Espeholt et al. 2018),
    vs_t = v_t + rho_t delta_t + gamma (1 - d_t) c_t (vs_{t+1} - v_{t+1}),  rho_t = min(rho_bar, pi_prox / mu),
    c_t = lam min(1, pi_prox / mu). With K = 1 the ratio is 1 and this is the GAE of Trainer (tests/test_epa_replay.py).
The minibatch size (envs per minibatch) equals the base learner's; the number of minibatches per epoch is K times larger.
Recurrent arms only: an old rollout restarts from the GRU state stored when it was collected (stale under the current
parameters; no burn-in), recorded as a known approximation.
"""
import jax
import jax.numpy as jnp
import optax

from src.pipeline.epa_harness import PPOConfig, Trainer, _vn_stats, _vn_update


class ReplayTrainer(Trainer):
    def __init__(self, arm, cfg: PPOConfig, replay_k: int = 2, rho_bar: float = 1.0):
        if not arm.recurrent:
            raise ValueError("ReplayTrainer supports recurrent arms only")
        if replay_k < 1:
            raise ValueError("replay_k must be >= 1")
        self.replay_k, self.rho_bar, self.base_minibatches = replay_k, rho_bar, cfg.minibatches
        # K x more minibatches of the same size per epoch: the lr schedule counts the real number of optimiser steps
        super().__init__(arm, cfg._replace(minibatches=cfg.minibatches * replay_k))
        if not isinstance(arm.init_carry(1), jax.Array):
            raise ValueError("ReplayTrainer needs an array carry (GRU-type arm)")

    def init(self, key):
        st = super().init(key)
        K, T, N = self.replay_k, self.cfg.num_steps, self.cfg.num_envs
        obs, c0 = st["obs"], st["carry"]
        z = lambda shape, dt=jnp.float32: jnp.zeros(shape, dt)
        st["rb"] = dict(obs=z((K, T) + obs.shape), nobs=z((K, T) + obs.shape), act=z((K, T, N), jnp.int32), logp=z((K, T, N)),
                        rew=z((K, T, N)), done=z((K, T, N), jnp.bool_), carry0=z((K,) + c0.shape), last_obs=z((K,) + obs.shape))
        return st

    def _prox(self, params, obs, act, done, carry0, last_obs):
        """Current-policy log-prob of the stored actions, values and bootstrap value; no gradient."""
        arm = self.arm

        def body(h, x):
            o, a, d = x
            logits, value, h2 = arm.step(params, h, o)
            lp = jnp.take_along_axis(jax.nn.log_softmax(logits), a[:, None], 1)[:, 0]
            return arm.advance(h2, o, a, d), (lp, value)

        h, (lp, val) = jax.lax.scan(body, carry0, (obs, act, done))
        _, last_v, _ = arm.step(params, h, last_obs)
        return jax.lax.stop_gradient(lp), jax.lax.stop_gradient(val), jax.lax.stop_gradient(last_v)

    def _vtrace_gae(self, rew, done, val, last_v, log_ratio):
        cfg = self.cfg
        ratio = jnp.exp(log_ratio)
        rho, c = jnp.minimum(self.rho_bar, ratio), cfg.lam * jnp.minimum(1.0, ratio)

        def body(carry, x):
            adv_n, v_n = carry
            r, d, v, rh, cc = x
            nd = 1.0 - d.astype(jnp.float32)
            adv = rh * (r + cfg.gamma * v_n * nd - v) + cfg.gamma * nd * cc * adv_n
            return (adv, v), adv

        _, adv = jax.lax.scan(body, (jnp.zeros_like(last_v), last_v), (rew, done, val, rho, c), reverse=True)
        return adv, adv + val

    def _update(self, st):
        cfg, arm, K = self.cfg, self.arm, self.replay_k
        key, kr, ku = jax.random.split(st["key"], 3)
        env_st, obs, carry, ep_ret, traj, _ = self._rollout(st["params"], st, kr)
        new = dict(obs=traj["obs"], nobs=traj["nobs"], act=traj["act"], logp=traj["logp"], rew=traj["rew"], done=traj["done"],
                   carry0=traj["carry"][0], last_obs=obs)
        first = st["u"] == 0   # the first update fills every slot with its own rollout (K copies, only until slot K is reached)
        rb = jax.tree_util.tree_map(lambda b, x: jnp.where(first, jnp.broadcast_to(x[None], b.shape), jnp.concatenate([x[None], b[:-1]], 0)),
                                    st["rb"], new)
        # slots -> one batch of K*N env sequences, time-major (T, K*N, ...)
        seq = lambda x: jnp.moveaxis(x, 0, 1).reshape((x.shape[1], K * x.shape[2]) + x.shape[3:])
        b = {k: seq(rb[k]) for k in ("obs", "nobs", "act", "logp", "rew", "done")}
        carry0 = rb["carry0"].reshape((K * cfg.num_envs,) + rb["carry0"].shape[2:])
        last_obs = rb["last_obs"].reshape((K * cfg.num_envs,) + rb["last_obs"].shape[2:])
        logp_prox, val, last_v = self._prox(st["params"], b["obs"], b["act"], b["done"], carry0, last_obs)
        vn = st["vn"]
        if cfg.value_norm:
            mu, sd = _vn_stats(vn)
            val, last_v = val * sd + mu, last_v * sd + mu
        adv, ret = self._vtrace_gae(b["rew"], b["done"], val, last_v, logp_prox - b["logp"])
        if cfg.value_norm:
            vn = _vn_update(vn, ret, cfg.value_norm)
            mu, sd = _vn_stats(vn)
            val, ret = (val - mu) / sd, (ret - mu) / sd
        if cfg.adv_norm == "batch":
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        batch = dict(b, logp=logp_prox, val=val, adv=adv, ret=ret, carry=carry0[None])
        nmb = cfg.minibatches
        ent_coef = None
        if cfg.ent_final >= 0:
            ent_coef = cfg.ent + (cfg.ent_final - cfg.ent) * st["u"].astype(jnp.float32) / max(self.n_updates - 1, 1)
        loss_fn = lambda p, mb: self._loss_rec(p, mb, ent_coef)

        def split(x, perm):
            n = perm.shape[0]
            if x.shape[0] == 1 and x.ndim == 3:            # carry: (1, K*N, W)
                return x[:, perm].reshape(1, nmb, n // nmb, -1).transpose(1, 0, 2, 3)
            return x[:, perm].reshape((x.shape[0], nmb, n // nmb) + x.shape[2:]).swapaxes(0, 1)

        def epoch(carry_, kk):
            perm = jax.random.permutation(kk, K * cfg.num_envs)
            mbs = jax.tree_util.tree_map(lambda x: split(x, perm), batch)

            def mb_step(c, mb):
                p, o = c
                (_, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, mb)
                upd, o = self.opt.update(g, o, p)
                return (optax.apply_updates(p, upd), o), jnp.concatenate([aux, optax.global_norm(g)[None]])

            return jax.lax.scan(mb_step, carry_, mbs)

        (params, opt), aux = jax.lax.scan(epoch, (st["params"], st["opt"]), jax.random.split(ku, cfg.epochs))
        m = aux.mean((0, 1))   # same stats layout as Trainer._update (statistics of the new rollout)
        stats = jnp.concatenate([m[:4], jnp.stack([traj["fin_ret"].sum(), traj["done"].sum().astype(jnp.float32)]), m[4:]])
        return dict(params=params, opt=opt, env=env_st, obs=obs, carry=carry, ep_ret=ep_ret, key=key, vn=vn, u=st["u"] + 1, rb=rb), stats
