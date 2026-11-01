"""SkillTruck: the Truck architecture (src/model/epa_policies.py::TFGRUArm, tile-token Transformer per step + GRU memory over
CLS, logits read from candidate-action tokens) carried over to the BOTSYSTEM skill-level environment (forge1201-MODED-AIPlayer
docs/proposed/RL-TRAINING-v2.md). Side track: nothing here counts toward a roadmap gate.

What changes against Truck, and why:
  tokens     81 grid-column tokens (9x9 columns x 5 heights x 11 block categories) instead of 63 Craftax tiles; one token each
             for own state, the last command's outcome, progress P and target T (DPOD's TransitionTarget channel); up to 36
             inventory and 8 entity tokens (learned embeddings of registry names + numeric features), masked when empty.
  candidates Truck has 17 fixed action tokens; here the environment lists up to K=48 concrete commands per step. A candidate
             token = embedding(command) + embedding(item/block) + embedding(entity) + Dense(features); padded ones are masked
             as keys and get logit -inf. Bidirectional attention over the whole set (4th idea), as in Truck.
  readout    Truck adds Dense(h) over the 17 actions; with a variable set the memory enters as a bilinear term
             <W h, token_k> / sqrt(d), so h can still re-rank every candidate.
  value      one head per curriculum stage (the reward changes meaning at the switch; a fresh head avoids a stale baseline).
  kind prior the logits get -log(number of offered candidates with the same command), so an untrained policy is uniform over
             command kinds first and over targets within a kind second. Without it the list's composition decided behaviour:
             up to 12 move_to and 11 mine candidates against one wait made a near-uniform policy walk and dig ~70 % of the time
             (run 2, 2026-11-01, entropy 2.83 after 11 updates). A fixed reparameterisation of the action set, no game knowledge.
Kept: d=64, 2 layers, 4 heads, GRU width 256, LayerNorm pre-norm blocks, orthogonal heads (logit scale 0.01, value 1.0).
"""
from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.epa_policies import _ap, _dense, _ln, _ortho, n_params  # noqa: E402

N_CAT, GRID_H, GRID_W = 11, 5, 9
N_TYPES = 9   # columns, self, inventory, entity, progress, target, last, candidate, cls


class SkillTruck:
    def __init__(self, sizes: dict, *, d=64, layers=2, heads=4, width=256, n_stages=2, kind_prior=True):
        self.s, self.d, self.layers, self.heads, self.width, self.n_stages = sizes, d, layers, heads, width, n_stages
        self.kind_prior = kind_prior
        self.name = f"skilltruck{d}x{layers}w{width}" + ("kp" if kind_prior else "")

    def init(self, key):
        s, d, w = self.s, self.d, self.width
        k = iter(jax.random.split(key, 40 + 4 * self.layers))
        emb = lambda n: jax.random.normal(next(k), (n, d)) * 0.02
        blocks = [{"ln1_g": jnp.ones((d,)), "ln1_b": jnp.zeros((d,)), "qkv": _dense(next(k), d, 3 * d),
                   "o": _dense(next(k), d, d, 0.5), "ln2_g": jnp.ones((d,)), "ln2_b": jnp.zeros((d,)),
                   "fc1": _dense(next(k), d, 4 * d), "fc2": _dense(next(k), 4 * d, d, 0.5)} for _ in range(self.layers)]
        return {"pi": {
            "col": _dense(next(k), GRID_H * N_CAT, d), "col_pos": emb(GRID_W * GRID_W),
            "self": _dense(next(k), s["self"], d), "item": emb(s["item"]), "ent": emb(s["entity"]), "cmd": emb(s["cmd"]),
            "reason": emb(s["reason"]), "inv_n": _dense(next(k), 1, d), "ent_f": _dense(next(k), s["ent_f"], d),
            "cand_f": _dense(next(k), s["cand_f"], d), "dt": _dense(next(k), 1, d),
            "prog": _dense(next(k), s["progress"], d), "tgt": _dense(next(k), s["progress"], d),
            "cls": jax.random.normal(next(k), (d,)) * 0.02, "type": emb(N_TYPES), "blocks": blocks,
            "lnf_g": jnp.ones((d,)), "lnf_b": jnp.zeros((d,)), "logit": _dense(next(k), d, 1, 0.01),
            "gru_wi": _ortho(next(k), d, 3 * w, 1.0), "gru_wh": _ortho(next(k), w, 3 * w, 1.0),
            "h_q": _ortho(next(k), w, d, 0.01),
            "h_value": [_ortho(next(k), w, 1, 1.0) for _ in range(self.n_stages)],
        }}

    def init_carry(self, n):
        return jnp.zeros((n, self.width))

    def _tokens(self, pi, o):
        """One observation (dict of arrays, unbatched) -> (tokens (L, d), key mask (L,), index of the first candidate)."""
        typ = pi["type"]
        grid = jax.nn.one_hot(o["grid"].reshape(GRID_H, GRID_W * GRID_W), N_CAT).transpose(1, 0, 2).reshape(GRID_W * GRID_W, -1)
        cols = _ap(pi["col"], grid) + pi["col_pos"] + typ[0]
        me = (_ap(pi["self"], o["self"]) + typ[1])[None]
        inv = pi["item"][o["inv_item"]] + _ap(pi["inv_n"], o["inv_count"][:, None]) + typ[2]
        ent = pi["ent"][o["ent_type"]] + _ap(pi["ent_f"], o["ent_feat"]) + typ[3]
        prog = (_ap(pi["prog"], o["progress"]) + typ[4])[None]
        tgt = (_ap(pi["tgt"], o["target"]) + typ[5])[None]
        last = (pi["reason"][o["last_reason"]] + _ap(pi["dt"], jnp.log1p(o["dt_ticks"])[None] / 5.0) + typ[6])[None]
        cand = pi["cmd"][o["cand_cmd"]] + pi["item"][o["cand_item"]] + pi["ent"][o["cand_ent"]] + _ap(pi["cand_f"], o["cand_feat"]) + typ[7]
        cls = (pi["cls"] + typ[8])[None]
        toks = [cols, me, inv, ent, prog, tgt, last, cand, cls]
        one = lambda n: jnp.ones((n,), jnp.bool_)
        valid = [one(cols.shape[0]), one(1), o["inv_item"] > 0, o["ent_type"] > 0, one(1), one(1), one(1), o["cand_mask"], one(1)]
        return jnp.concatenate(toks, 0), jnp.concatenate(valid, 0), sum(t.shape[0] for t in toks[:7])

    def _encode(self, pi, o):
        d, nh = self.d, self.heads
        x, kv, c0 = self._tokens(pi, o)
        L = x.shape[0]
        for b in pi["blocks"]:
            h = _ln(x, b["ln1_g"], b["ln1_b"])
            q, kk, v = jnp.split(_ap(b["qkv"], h), 3, -1)
            sp = lambda z: z.reshape(L, nh, d // nh).transpose(1, 0, 2)
            q, kk, v = sp(q), sp(kk), sp(v)
            att = jax.nn.softmax(jnp.where(kv[None, None, :], jnp.einsum("hqd,hkd->hqk", q, kk) / jnp.sqrt(d // nh), -1e9), -1)
            x = x + _ap(b["o"], jnp.einsum("hqk,hkd->hqd", att, v).transpose(1, 0, 2).reshape(L, d))
            h = _ln(x, b["ln2_g"], b["ln2_b"])
            x = x + _ap(b["fc2"], jax.nn.gelu(_ap(b["fc1"], h)))
        x = _ln(x, pi["lnf_g"], pi["lnf_b"])
        K = o["cand_mask"].shape[0]
        return x[c0:c0 + K], x[L - 1]

    def step(self, params, h, obs, stage):
        """Batched: obs = dict of (N, ...) arrays incl. 'target'; h (N, width); stage (N,) int.
        Returns logits (N, K) with -1e9 on padded candidates, value (N,) of each row's stage head, next h."""
        pi, w = params["pi"], self.width
        ch, cls = jax.vmap(lambda o: self._encode(pi, o))(obs)
        ir, iz, inn = jnp.split(_ap(pi["gru_wi"], cls), 3, -1)
        hr, hz, hn = jnp.split(_ap(pi["gru_wh"], h), 3, -1)
        r, zg = jax.nn.sigmoid(ir + hr), jax.nn.sigmoid(iz + hz)
        h2 = (1.0 - zg) * jnp.tanh(inn + r * hn) + zg * h
        logits = _ap(pi["logit"], ch)[..., 0] + jnp.einsum("nd,nkd->nk", _ap(pi["h_q"], h2), ch) / jnp.sqrt(self.d)
        if self.kind_prior:
            cmd, mask = obs["cand_cmd"], obs["cand_mask"]
            n_kind = jnp.sum((cmd[:, :, None] == cmd[:, None, :]) & mask[:, None, :], -1)
            logits = logits - jnp.log(jnp.maximum(n_kind, 1).astype(logits.dtype))
        logits = jnp.where(obs["cand_mask"], logits, -1e9)
        values = jnp.stack([_ap(v, h2)[:, 0] for v in pi["h_value"]], -1)
        value = jnp.take_along_axis(values, stage[:, None], 1)[:, 0]
        return logits, value, h2

    def param_counts(self, params):
        pi = params["pi"]
        return n_params(params), n_params({k: v for k, v in pi.items() if k != "h_value"})


OBS_KEYS = ("grid", "self", "inv_item", "inv_count", "ent_type", "ent_feat", "progress", "cand_cmd", "cand_item", "cand_ent",
            "cand_feat", "cand_mask", "last_reason", "dt_ticks")
INT_KEYS = {"grid", "inv_item", "ent_type", "cand_cmd", "cand_item", "cand_ent", "last_reason"}


def sizes_from_hello(hello: dict) -> dict:
    spec = {f["name"]: f["shape"] for f in hello["obs_spec"]["fields"]}
    missing = set(OBS_KEYS) - set(spec)
    if missing:
        raise ValueError(f"hello lacks observation fields {sorted(missing)}")
    v = hello["vocab_sizes"]
    return {"self": spec["self"][0], "item": v["item"], "entity": v["entity"], "cmd": v["cmd"], "reason": v["reason"],
            "ent_f": spec["ent_feat"][1], "cand_f": spec["cand_feat"][1], "progress": spec["progress"][0], "K": spec["cand_mask"][0]}
