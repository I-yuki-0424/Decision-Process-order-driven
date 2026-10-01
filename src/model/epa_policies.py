"""Policy families for the Phase-1 comparison on Craftax-Classic (1345-d symbolic observation, 17 actions).

Every family exposes the same `Arm` interface so the PPO learner, evaluator and parameter counter are identical across
arms (roadmap P9 / P11):
    init(key)                          -> params (PyTree; "pi" = trained by the policy-gradient loss, optional "wm")
    init_carry(n)                      -> per-env carry (history window or GRU state)
    step(params, carry, obs)           -> (logits (N,17), value (N,), carry_aux)
    advance(carry_aux, obs, act, done) -> next carry (history shifted / GRU state zeroed on episode end)
    aux_loss(params, batch)            -> scalar auxiliary loss (world model) or 0
    param_counts(params)               -> (params_total, params_deployed)

Observation layout (craftax_classic renderer): obs[:1323] = 7x9 tiles x 21 channels (17 block one-hots + 4 mob planes),
obs[1323:] = 22 scalars (12 inventory, 4 intrinsics, 4 direction one-hot, light, sleeping).

Families
  mlp   : Craftax_Baselines PPO actor-critic (separate tanh MLPs, orthogonal init).      [pure RL, feed-forward]
  gru   : dense embed -> GRU -> actor/critic heads (PureJaxRL PPO-RNN pattern).           [pure RL, recurrent]
  tf    : pure Transformer over tokens {63 tiles, 1 scalar token, H history steps, 17 action candidates, CLS}; the
          logit of action a is read from its own candidate token (bidirectional over the candidate set, 4th idea).
  tf+wm : as tf, but each candidate token additionally receives features of an action-conditioned one-step world model
          (Idea 6, no noop future): per candidate action a the WM produces a hidden state h_a and a reward prediction.
          The WM is trained only on the transition that was actually executed (one action per state) and its features
          enter the policy through a stop-gradient. `wm_mode="random"` is the control: same architecture, frozen at init.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

OBS_DIM, N_ACT, N_TILE, TILE_DIM, EXTRA_DIM = 1345, 17, 63, 21, 22
MAP_DIM = N_TILE * TILE_DIM  # 1323


def n_params(tree) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(tree)))


def _ortho(key, i, o, scale):
    return {"w": jax.nn.initializers.orthogonal(scale)(key, (i, o)), "b": jnp.zeros((o,))}


def _dense(key, i, o, scale=1.0):
    return {"w": jax.random.normal(key, (i, o)) * scale / jnp.sqrt(i), "b": jnp.zeros((o,))}


def _ap(p, x):
    return x @ p["w"] + p["b"]


def _ln(x, g, b):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + 1e-5) * g + b


def _mlp(key, sizes, last_scale):
    ks = jax.random.split(key, len(sizes) - 1)
    return [_ortho(k, a, b, last_scale if i == len(sizes) - 2 else jnp.sqrt(2.0))
            for i, (k, a, b) in enumerate(zip(ks, sizes[:-1], sizes[1:]))]


def _tanh_mlp(layers, x):
    for p in layers[:-1]:
        x = jnp.tanh(_ap(p, x))
    return _ap(layers[-1], x)


class Arm:
    name = "arm"
    recurrent = False
    hist_len = 0

    def init_carry(self, n):
        return ()

    def advance(self, carry_aux, obs, act, done):
        return carry_aux

    def aux_loss(self, params, batch):
        return jnp.zeros(())

    def train_forward(self, params, mb):
        """(logits, value, auxiliary loss) for a flat minibatch; one forward pass."""
        logits, value, _ = self.step(params, mb["carry"], mb["obs"])
        return logits, value, self.aux_loss(params, mb)

    def frozen(self):
        return False


# ----------------------------------------------------------------------------------------------------------------------
# Pure RL: MLP
# ----------------------------------------------------------------------------------------------------------------------
class MLPArm(Arm):
    def __init__(self, width=512, layers=3):
        self.name, self.width, self.layers = f"mlp{width}x{layers}", width, layers

    def init(self, key):
        ka, kc = jax.random.split(key)
        sz = [OBS_DIM] + [self.width] * self.layers
        return {"pi": {"actor": _mlp(ka, sz + [N_ACT], 0.01), "critic": _mlp(kc, sz + [1], 1.0)}}

    def step(self, params, carry, obs):
        p = params["pi"]
        return _tanh_mlp(p["actor"], obs), _tanh_mlp(p["critic"], obs)[..., 0], carry

    def param_counts(self, params):
        return n_params(params), n_params(params["pi"]["actor"])


# ----------------------------------------------------------------------------------------------------------------------
# Pure RL: GRU
# ----------------------------------------------------------------------------------------------------------------------
class GRUArm(Arm):
    recurrent = True

    def __init__(self, width=256):
        self.name, self.width = f"gru{width}", width

    def init(self, key):
        k = iter(jax.random.split(key, 8))
        w = self.width
        gru = {"wi": _ortho(next(k), w, 3 * w, 1.0), "wh": _ortho(next(k), w, 3 * w, 1.0)}
        return {"pi": {"embed": _ortho(next(k), OBS_DIM, w, jnp.sqrt(2.0)), "gru": gru,
                       "actor": _mlp(next(k), [w, w, N_ACT], 0.01), "critic": _mlp(next(k), [w, w, 1], 1.0)}}

    def init_carry(self, n):
        return jnp.zeros((n, self.width))

    def _embed(self, p, obs):
        return jnp.tanh(_ap(p["embed"], obs))

    def step(self, params, h, obs):
        p = params["pi"]
        x = self._embed(p, obs)
        gi, gh = _ap(p["gru"]["wi"], x), _ap(p["gru"]["wh"], h)
        ir, iz, inn = jnp.split(gi, 3, -1)
        hr, hz, hn = jnp.split(gh, 3, -1)
        r, z = jax.nn.sigmoid(ir + hr), jax.nn.sigmoid(iz + hz)
        n = jnp.tanh(inn + r * hn)
        h2 = (1.0 - z) * n + z * h
        return _tanh_mlp(p["actor"], h2), _tanh_mlp(p["critic"], h2)[..., 0], h2

    def advance(self, h_next, obs, act, done):
        return h_next * (1.0 - done.astype(jnp.float32))[:, None]

    def param_counts(self, params):
        pi = params["pi"]
        dep = n_params(pi["embed"]) + n_params(pi["gru"]) + n_params(pi["actor"])
        return n_params(params), dep


# ----------------------------------------------------------------------------------------------------------------------
# World model (Idea 6 style, action-conditioned, step-by-step: no noop future)
# ----------------------------------------------------------------------------------------------------------------------
def init_world_model(key, hidden=256, d_act=32):
    k = jax.random.split(key, 5)
    return {"enc": _dense(k[0], OBS_DIM, hidden), "act_emb": jax.random.normal(k[1], (N_ACT, d_act)) * 0.1,
            "dyn": _dense(k[2], hidden + d_act, hidden), "head_obs": _dense(k[3], hidden, OBS_DIM, 0.1),
            "head_r": _dense(k[4], hidden, 1, 0.1)}


def wm_candidates(wm, obs):
    """Per-candidate-action hidden state and reward prediction for ONE observation: (17, hidden), (17,)."""
    h = jax.nn.gelu(_ap(wm["enc"], obs))
    x = jnp.concatenate([jnp.broadcast_to(h, (N_ACT, h.shape[0])), wm["act_emb"]], -1)
    ha = jax.nn.gelu(_ap(wm["dyn"], x))
    return ha, _ap(wm["head_r"], ha)[:, 0]


def wm_loss(wm, obs, act, rew, nobs, done):
    """One-step loss on the EXECUTED action only (one action per state): next-observation delta and reward.
    Transitions that ended an episode are dropped (the returned next observation belongs to a fresh episode)."""
    ha_all, _ = jax.vmap(lambda o: wm_candidates(wm, o))(obs)
    ha = jnp.take_along_axis(ha_all, act[:, None, None], axis=1)[:, 0]
    d_hat, r_hat = _ap(wm["head_obs"], ha), _ap(wm["head_r"], ha)[:, 0]
    w = 1.0 - done.astype(jnp.float32)
    l_obs = (((d_hat - (nobs - obs)) ** 2).mean(-1) * w).sum() / jnp.maximum(w.sum(), 1.0)
    l_rew = (((r_hat - rew) ** 2) * w).sum() / jnp.maximum(w.sum(), 1.0)
    return 100.0 * l_obs + l_rew


# ----------------------------------------------------------------------------------------------------------------------
# Pure Transformer (+ optional world-model features)
# ----------------------------------------------------------------------------------------------------------------------
class TFArm(Arm):
    def __init__(self, d=64, layers=2, heads=4, hist=4, wm_mode="none", wm_hidden=256, wm_coef=1.0, critic="shared",
                 wmq=False, gamma=0.99, aux_coef=0.0):
        assert wm_mode in ("none", "trained", "random") and critic in ("shared", "mlp")
        assert not wmq or (wm_mode != "none" and critic == "mlp"), "wmq needs a world model and a separate critic"
        self.critic, self.wmq, self.gamma, self.aux_coef = critic, wmq, gamma, aux_coef
        self.d, self.layers, self.heads, self.hist_len = d, layers, heads, hist
        self.wm_mode, self.wm_hidden, self.wm_coef = wm_mode, wm_hidden, wm_coef
        self.name = (f"tf{d}x{layers}h{hist}" + ("" if wm_mode == "none" else f"+wm_{wm_mode}")
                     + ("+sc" if critic == "mlp" else "") + ("+q" if wmq else "")
                     + (f"+aux{aux_coef:g}" if aux_coef else ""))

    def init(self, key):
        d, H = self.d, self.hist_len
        k = iter(jax.random.split(key, 16 + 4 * self.layers))
        blocks = [{"ln1_g": jnp.ones((d,)), "ln1_b": jnp.zeros((d,)), "qkv": _dense(next(k), d, 3 * d),
                   "o": _dense(next(k), d, d, 0.5), "ln2_g": jnp.ones((d,)), "ln2_b": jnp.zeros((d,)),
                   "fc1": _dense(next(k), d, 4 * d), "fc2": _dense(next(k), 4 * d, d, 0.5)}
                  for _ in range(self.layers)]
        pi = {"tile": _dense(next(k), TILE_DIM, d), "tile_pos": jax.random.normal(next(k), (N_TILE, d)) * 0.02,
              "extra": _dense(next(k), EXTRA_DIM, d), "hist": _dense(next(k), OBS_DIM, d),
              "hist_act": jax.random.normal(next(k), (N_ACT + 1, d)) * 0.02,
              "hist_pos": jax.random.normal(next(k), (max(H, 1), d)) * 0.02,
              "cand": jax.random.normal(next(k), (N_ACT, d)) * 0.02,
              "cls": jax.random.normal(next(k), (d,)) * 0.02, "type": jax.random.normal(next(k), (5, d)) * 0.02,
              "blocks": blocks, "lnf_g": jnp.ones((d,)), "lnf_b": jnp.zeros((d,)),
              "logit": _dense(next(k), d, 1, 0.01), "value": _dense(next(k), d, 1, 1.0)}
        out = {"pi": pi}
        if self.critic == "mlp":   # separate tanh-MLP critic on the raw observation, as in the MLP/GRU baselines
            pi["critic"] = _mlp(next(k), [OBS_DIM, 512, 512, 512, 1], 1.0)
            del pi["value"]
        if self.aux_coef:   # candidate-token heads: reward and next-observation delta of the EXECUTED action
            pi["aux_r"] = _dense(next(k), d, 1, 0.1)
            pi["aux_obs"] = _dense(next(k), d, OBS_DIM, 0.1)
        if self.wm_mode != "none":
            pi["wm_feat"] = _dense(next(k), self.wm_hidden + 1 + int(self.wmq), d)
            out["wm"] = init_world_model(next(k), self.wm_hidden)
        return out

    def init_carry(self, n):
        H = self.hist_len
        return (jnp.zeros((n, H, OBS_DIM)), jnp.full((n, H), N_ACT, jnp.int32), jnp.zeros((n, H), jnp.bool_))

    def advance(self, carry, obs, act, done):
        H = self.hist_len
        if H == 0:
            return carry
        ho, ha, hv = carry
        ho2 = jnp.concatenate([ho[:, 1:], obs[:, None]], 1)
        ha2 = jnp.concatenate([ha[:, 1:], act[:, None].astype(jnp.int32)], 1)
        hv2 = jnp.concatenate([hv[:, 1:], jnp.ones_like(hv[:, :1])], 1) & ~done[:, None]
        return ho2, ha2, hv2

    def _one_h(self, params, obs, ho, ha, hv):
        pi, d, H, nh = params["pi"], self.d, self.hist_len, self.heads
        tiles = obs[:MAP_DIM].reshape(N_TILE, TILE_DIM)
        typ = pi["type"]
        cand = pi["cand"]
        if self.wm_mode != "none":
            ha_wm, r_hat = wm_candidates(jax.lax.stop_gradient(params["wm"]), obs)
            feats = jnp.concatenate([ha_wm, r_hat[:, None]], -1)
            if self.wmq:   # imagined one-step lookahead: q_a = r^_a + gamma V(obs + delta^_a) - V(obs), V = the PPO critic (stop-grad)
                crit = jax.lax.stop_gradient(pi["critic"])
                d_hat = _ap(jax.lax.stop_gradient(params["wm"])["head_obs"], ha_wm)
                v_next = _tanh_mlp(crit, obs[None] + d_hat)[:, 0]
                q = r_hat + self.gamma * v_next - _tanh_mlp(crit, obs)[0]
                feats = jnp.concatenate([feats, q[:, None]], -1)
            feats = jax.lax.stop_gradient(feats)
            cand = cand + _ap(pi["wm_feat"], feats)
        toks = [_ap(pi["tile"], tiles) + pi["tile_pos"] + typ[0], (_ap(pi["extra"], obs[MAP_DIM:]) + typ[1])[None]]
        valid = [jnp.ones((N_TILE + 1,), jnp.bool_)]
        if H:
            toks.append(_ap(pi["hist"], ho) + pi["hist_act"][ha] + pi["hist_pos"] + typ[2])
            valid.append(hv)
        toks += [cand + typ[3], (pi["cls"] + typ[4])[None]]
        valid.append(jnp.ones((N_ACT + 1,), jnp.bool_))
        x, kv = jnp.concatenate(toks, 0), jnp.concatenate(valid, 0)
        L = x.shape[0]
        mask = jnp.broadcast_to(kv[None, :], (L, L))
        for b in pi["blocks"]:
            h = _ln(x, b["ln1_g"], b["ln1_b"])
            q, kk, v = jnp.split(_ap(b["qkv"], h), 3, -1)
            sp = lambda z: z.reshape(L, nh, d // nh).transpose(1, 0, 2)
            q, kk, v = sp(q), sp(kk), sp(v)
            att = jax.nn.softmax(jnp.where(mask[None], jnp.einsum("hqd,hkd->hqk", q, kk) / jnp.sqrt(d // nh), -1e9), -1)
            x = x + _ap(b["o"], jnp.einsum("hqk,hkd->hqd", att, v).transpose(1, 0, 2).reshape(L, d))
            h = _ln(x, b["ln2_g"], b["ln2_b"])
            x = x + _ap(b["fc2"], jax.nn.gelu(_ap(b["fc1"], h)))
        x = _ln(x, pi["lnf_g"], pi["lnf_b"])
        ch = x[L - 1 - N_ACT:L - 1]
        logits = _ap(pi["logit"], ch)[:, 0]
        if self.critic == "mlp":
            return logits, _tanh_mlp(pi["critic"], obs)[0], ch, x[L - 1]
        value = _ap(pi["value"], x[L - 1])[0] if "value" in pi else jnp.zeros(())   # TFGRUArm takes its value from the GRU
        return logits, value, ch, x[L - 1]

    def step(self, params, carry, obs):
        ho, ha, hv = carry
        logits, value, _, _ = jax.vmap(lambda o, a, b, c: self._one_h(params, o, a, b, c))(obs, ho, ha, hv)
        return logits, value, carry

    def train_forward(self, params, mb):
        if not self.aux_coef:
            return super().train_forward(params, mb)
        ho, ha, hv = mb["carry"]
        logits, value, ch, _ = jax.vmap(lambda o, a, b, c: self._one_h(params, o, a, b, c))(mb["obs"], ho, ha, hv)
        sel = jnp.take_along_axis(ch, mb["act"][:, None, None], axis=1)[:, 0]      # token of the executed action only
        pi = params["pi"]
        r_hat, d_hat = _ap(pi["aux_r"], sel)[:, 0], _ap(pi["aux_obs"], sel)
        w = 1.0 - mb["done"].astype(jnp.float32)   # drop transitions that ended an episode (next obs is a fresh episode)
        den = jnp.maximum(w.sum(), 1.0)
        l_obs = (((d_hat - (mb["nobs"] - mb["obs"])) ** 2).mean(-1) * w).sum() / den
        l_rew = (((r_hat - mb["rew"]) ** 2) * w).sum() / den
        return logits, value, self.aux_coef * (100.0 * l_obs + l_rew)

    def aux_loss(self, params, batch):
        if self.wm_mode != "trained":
            return jnp.zeros(())
        return self.wm_coef * wm_loss(params["wm"], batch["obs"], batch["act"], batch["rew"], batch["nobs"], batch["done"])

    def frozen(self):
        return self.wm_mode == "random"

    def param_counts(self, params):
        total = n_params(params)
        pi = params["pi"]
        deployed = n_params({k: v for k, v in pi.items() if k not in ("value", "critic")}) + n_params(params.get("wm", {}))
        return total, deployed


class TFGRUArm(TFArm):
    """Tile-token Transformer encoder per step (no history tokens) + GRU memory over the CLS embedding.
    logits = candidate-token logit + Dense(h); value from h. Recurrent: the learner trains on sequences."""
    recurrent = True

    def __init__(self, d=64, layers=2, heads=4, width=256, wm_mode="none", wm_hidden=256, wm_coef=1.0):
        super().__init__(d=d, layers=layers, heads=heads, hist=0, wm_mode=wm_mode, wm_hidden=wm_hidden, wm_coef=wm_coef,
                         critic="shared")
        self.width = width
        self.name = f"tfgru{d}x{layers}w{width}" + ("" if wm_mode == "none" else f"+wm_{wm_mode}")

    def init(self, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        out = super().init(k1)
        pi, w = out["pi"], self.width
        del pi["value"]
        pi["gru_wi"], pi["gru_wh"] = _ortho(k2, self.d, 3 * w, 1.0), _ortho(k3, w, 3 * w, 1.0)
        pi["h_logit"], pi["h_value"] = _ortho(k4, w, N_ACT, 0.01), _ortho(k5, w, 1, 1.0)
        return out

    def init_carry(self, n):
        return jnp.zeros((n, self.width))

    def step(self, params, h, obs):
        pi = params["pi"]
        z = jnp.zeros((obs.shape[0], 0, OBS_DIM)), jnp.zeros((obs.shape[0], 0), jnp.int32), jnp.zeros((obs.shape[0], 0), jnp.bool_)
        lc, _, _, cls = jax.vmap(lambda o, a, b, c: self._one_h(params, o, a, b, c))(obs, *z)
        ir, iz, inn = jnp.split(_ap(pi["gru_wi"], cls), 3, -1)
        hr, hz, hn = jnp.split(_ap(pi["gru_wh"], h), 3, -1)
        r, zg = jax.nn.sigmoid(ir + hr), jax.nn.sigmoid(iz + hz)
        h2 = (1.0 - zg) * jnp.tanh(inn + r * hn) + zg * h
        return lc + _ap(pi["h_logit"], h2), _ap(pi["h_value"], h2)[:, 0], h2

    def advance(self, h_next, obs, act, done):
        return h_next * (1.0 - done.astype(jnp.float32))[:, None]

    def train_forward(self, params, mb):   # only used by flat learners; recurrent learners call step()
        raise NotImplementedError

    def param_counts(self, params):
        pi = params["pi"]
        return n_params(params), n_params({k: v for k, v in pi.items() if k != "h_value"}) + n_params(params.get("wm", {}))


class CNNGRUArm(GRUArm):
    """Control for TFGRUArm: the same GRU/heads, but the per-step encoder is a 2-layer CNN over the 7x9x21 tile map
    (+ the 22 scalars) instead of a Transformer. Separates 'spatial encoder + memory' from 'attention'."""

    def __init__(self, width=256, ch1=32, ch2=64):
        super().__init__(width)
        self.ch1, self.ch2, self.name = ch1, ch2, f"cnngru{ch1}-{ch2}w{width}"

    def init(self, key):
        out = super().init(key)
        k = jax.random.split(jax.random.fold_in(key, 11), 3)
        pi = out["pi"]
        pi["c1"] = {"w": jax.nn.initializers.orthogonal(jnp.sqrt(2.0))(k[0], (3 * 3 * TILE_DIM, self.ch1)).reshape(3, 3, TILE_DIM, self.ch1),
                    "b": jnp.zeros((self.ch1,))}
        pi["c2"] = {"w": jax.nn.initializers.orthogonal(jnp.sqrt(2.0))(k[1], (3 * 3 * self.ch1, self.ch2)).reshape(3, 3, self.ch1, self.ch2),
                    "b": jnp.zeros((self.ch2,))}
        pi["embed"] = _ortho(k[2], 63 * self.ch2 + EXTRA_DIM, self.width, jnp.sqrt(2.0))
        return out

    def _embed(self, p, obs):
        x = obs[..., :MAP_DIM].reshape(obs.shape[:-1] + (7, 9, TILE_DIM))
        for name in ("c1", "c2"):
            x = jax.lax.conv_general_dilated(x, p[name]["w"], (1, 1), "SAME", dimension_numbers=("NHWC", "HWIO", "NHWC"))
            x = jax.nn.relu(x + p[name]["b"])
        z = jnp.concatenate([x.reshape(obs.shape[:-1] + (-1,)), obs[..., MAP_DIM:]], -1)
        return jnp.tanh(_ap(p["embed"], z))

    def param_counts(self, params):
        pi = params["pi"]
        dep = sum(n_params(pi[k]) for k in ("c1", "c2", "embed", "gru", "actor"))
        return n_params(params), dep


def make_optimizer(arm: Arm, lr, max_grad_norm):
    """Separate clip+Adam per subtree so the WM loss scale cannot interact with the policy step; a 'random' WM is frozen."""
    def chain():
        return optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr, eps=1e-5))

    def labels(params):
        return {k: ("pi" if k == "pi" else ("frozen" if arm.frozen() else "wm")) for k in params}

    return optax.multi_transform({"pi": chain(), "wm": chain(), "frozen": optax.set_to_zero()}, labels)


ARM_BUILDERS = {
    "ppo_mlp": lambda **kw: MLPArm(**kw),
    "ppo_gru": lambda **kw: GRUArm(**kw),
    "tf": lambda **kw: TFArm(wm_mode="none", **kw),
    "tf_wm": lambda **kw: TFArm(wm_mode="trained", **kw),
    "tf_wm_random": lambda **kw: TFArm(wm_mode="random", **kw),
    "tf_gru": lambda **kw: TFGRUArm(**kw),
    "cnn_gru": lambda **kw: CNNGRUArm(**kw),
    "tf_gru_wm": lambda **kw: TFGRUArm(wm_mode="trained", **kw),
    "tf_gru_wm_random": lambda **kw: TFGRUArm(wm_mode="random", **kw),
    "tf_aux": lambda **kw: TFArm(wm_mode="none", **kw),
    "tf_wmq": lambda **kw: TFArm(wm_mode="trained", critic="mlp", wmq=True, **kw),
    "tf_wmq_random": lambda **kw: TFArm(wm_mode="random", critic="mlp", wmq=True, **kw),
}
