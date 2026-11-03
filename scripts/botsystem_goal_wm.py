"""GoalWM: goal-conditioned policy with an explicit-state world model for the BOTSYSTEM skill-level environment (forge1201-
MODED-AIPlayer env >= v2.0). Side track (DPOD TASK-20261101-031): nothing here counts toward a roadmap gate.

Built from the architecture review of 2026-11-01 (STATE.yaml TASK-20261101-031), agreed by the operator:
  encoder    Truck's per-step Transformer over physical tokens: 81 grid columns, own state (+ episode clock), 36 inventory slots,
             8 entities, the progress flags, kills this episode, the last outcome, 32 patches of the first-hit view image (what
             the player sees: no block behind another) and up to K candidate commands. It sees no goal, so the world model and
             the reachability head below are goal-independent and can be pretrained on logged runs.
  goal       up to G goal tokens (kind: have item / kill / advancement / night; id; target delta; progress so far; age). A
             dimension that is not named is free (owner's "free values"). Goals are relative: "+1 of X since assignment".
             One cross-attention readout per goal over the encoded tokens; the mean enters the GRU and re-ranks candidates.
  world      per candidate, from its encoded token only: success probability, log duration, change of the 20 own-state numbers,
  model      change of every inventory count (an output table per item . z, plus a learned copy gate alpha(z) on the item the
             command names, as in pointer networks: crafting X raises X, placing X lowers it; E1 on run 1-3 logs showed the plain
             output table far behind a per-(command, item) lookup without it) and of every kill count (gate beta(z) on the
             entity the command names). Inventory balance is
             structural: a predicted loss can never exceed what is held (delta >= -held), plain counting, no game knowledge.
             These replace both the Hamiltonian-NN idea (inventory dynamics are discrete, non-conservative stoichiometry) and
             environment-supplied "inventory/state on success" fields (the consequence of an action handed to the model).
  policy     Truck's logit (token + bilinear memory term) + a bilinear goal term + u . stop_gradient(p_ok, log dt, predicted goal
             gain): the predicted conditions and effects of each command are an input to the choice, learned, never hard-coded.
  reach      R(s, g) = P(goal g reached within H steps | encoded state), trained with hindsight relabelling on what actually
             happened; the learner uses it to propose goals of intermediate difficulty and to skip goals that are too early.
  plan       batch generation (DPOD 4th/5th idea, owner 2026-11-02): an autoregressive decoder generates, at the start, the
             whole sequence of subgoals (goal-space dimensions) toward a target T, by beam search: causal self-attention over
             the plan so far, cross-attention into the encoded state, output tied to the goal-dimension embeddings. Concrete
             commands cannot be generated in advance (a candidate such as "mine the iron ore at x y z" exists only once seen),
             so the batch is generated at the upper level of the 5th idea and the goal-conditioned policy executes it. Trained
             on hindsight: the order in which bots actually first raised dimensions before reaching a later one (no tech tree).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.epa_policies import _ap, _dense, _ln, _ortho, n_params  # noqa: E402

N_CAT, GRID_H, GRID_W = 11, 5, 9
GOAL_KINDS = ("pad", "item", "kill", "adv", "night")
GK = {k: i for i, k in enumerate(GOAL_KINDS)}
GOAL_F = 3            # target delta (log1p), progress so far / delta (0..1), age / horizon
N_TYPES = 11          # columns, self, inventory, entity, progress, killed, last, view, candidate, cls, goal
PLAN_L = 8            # plan tokens generated at most (the last one may be END)
WM_F = 3              # policy features from the world model: p_ok, log dt / 5, predicted goal gain
WM_FEAT_SCALE = 4.0   # features x4: u ~ 1 can then dominate the choice (Adam moves u by ~lr per step; at x1 the goal control
                      # reached u_gain 0.31 after ~2,300 steps, a logit bonus too small to move an 8-way choice)
INV_SCALE = 1.0       # world-model count unit (loss in counts, Huber k=1)


def huber(x, k=1.0):
    a = jnp.abs(x)
    return jnp.where(a < k, 0.5 * x * x, k * (a - 0.5 * k))


def floor_st(x, lo):
    """max(x, lo) in the forward pass, identity gradient (straight-through): a hard floor would stop learning for every item
    not held whose prediction starts below zero (the gain of a new item could never be learned)."""
    return x + jax.lax.stop_gradient(jnp.maximum(x, lo) - x)


def held_vector(inv_item, inv_count, n_items):
    """Counts per item id from the 36 slots (count/64 on the wire)."""
    return jnp.zeros((n_items,)).at[inv_item].add(jnp.round(inv_count * 64.0) * (inv_item > 0))


class GoalWM:
    def __init__(self, sizes: dict, *, d=64, layers=2, heads=4, width=256, kind_prior=True):
        self.s, self.d, self.layers, self.heads, self.width, self.kind_prior = sizes, d, layers, heads, width, kind_prior
        self.G = sizes["G"]
        self.L = sizes.get("plan_l", PLAN_L)
        self.name = f"goalwm{d}x{layers}w{width}g{self.G}" + ("kp" if kind_prior else "") + f"pl{self.L}"

    # ------------------------------------------------------------------ parameters
    def init(self, key):
        s, d, w = self.s, self.d, self.width
        k = iter(jax.random.split(key, 64 + 4 * self.layers))
        emb = lambda n: jax.random.normal(next(k), (n, d)) * 0.02
        blocks = [{"ln1_g": jnp.ones((d,)), "ln1_b": jnp.zeros((d,)), "qkv": _dense(next(k), d, 3 * d),
                   "o": _dense(next(k), d, d, 0.5), "ln2_g": jnp.ones((d,)), "ln2_b": jnp.zeros((d,)),
                   "fc1": _dense(next(k), d, 4 * d), "fc2": _dense(next(k), 4 * d, d, 0.5)} for _ in range(self.layers)]
        enc = {
            "col": _dense(next(k), GRID_H * N_CAT, d), "col_pos": emb(GRID_W * GRID_W),
            "self": _dense(next(k), s["self"], d), "clock": _dense(next(k), s["clock"], d),
            "item": emb(s["item"]), "ent": emb(s["entity"]), "cmd": emb(s["cmd"]), "reason": emb(s["reason"]),
            "inv_n": _dense(next(k), 1, d), "ent_f": _dense(next(k), s["ent_f"], d), "cand_f": _dense(next(k), s["cand_f"], d),
            "dt": _dense(next(k), 1, d), "prog": _dense(next(k), s["progress"], d), "killed": _dense(next(k), s["entity"], d),
            "view": _dense(next(k), 4 * (N_CAT + 1), d), "view_pos": emb(max(s["view_patches"], 1)),
            "cls": jax.random.normal(next(k), (d,)) * 0.02, "type": emb(N_TYPES), "blocks": blocks,
            "lnf_g": jnp.ones((d,)), "lnf_b": jnp.zeros((d,)),
        }
        goal = {"kind": emb(len(GOAL_KINDS)), "adv": emb(s["progress"]), "feat": _dense(next(k), GOAL_F, d),
                "q": _dense(next(k), d, d), "kv": _dense(next(k), d, 2 * d), "o": _dense(next(k), d, d, 0.5),
                "ln_g": jnp.ones((d,)), "ln_b": jnp.zeros((d,))}
        wm = {"z1": _dense(next(k), d, d), "z2": _dense(next(k), d, d), "ok": _dense(next(k), d, 1, 0.1),
              "dt": _dense(next(k), d, 1, 0.1), "dself": _dense(next(k), d, s["self"], 0.1),
              "inv_out": jax.random.normal(next(k), (s["item"], d)) / jnp.sqrt(d), "kill_out": jax.random.normal(next(k), (s["entity"], d)) / jnp.sqrt(d),
              "inv_b": jnp.zeros((s["item"],)), "kill_b": jnp.zeros((s["entity"],)), "copy": _dense(next(k), d, 2, 0.1)}
        reach = {"delta": _dense(next(k), 1, d), "r1": _dense(next(k), 2 * d, 2 * d), "r2": _dense(next(k), 2 * d, 1, 0.1)}
        pi = {"logit": _dense(next(k), d, 1, 0.01), "gru_wi": _ortho(next(k), d, 3 * w, 1.0), "gru_wh": _ortho(next(k), w, 3 * w, 1.0),
              "g_in": _dense(next(k), d, d, 0.5), "h_q": _ortho(next(k), w, d, 0.01), "g_q": _dense(next(k), d, d, 0.01),
              "u": jnp.zeros((WM_F,)), "value": _ortho(next(k), w, 1, 1.0)}
        pblocks = [{"ln1_g": jnp.ones((d,)), "ln1_b": jnp.zeros((d,)), "qkv": _dense(next(k), d, 3 * d), "o": _dense(next(k), d, d, 0.5),
                    "lnc_g": jnp.ones((d,)), "lnc_b": jnp.zeros((d,)), "cq": _dense(next(k), d, d), "ckv": _dense(next(k), d, 2 * d),
                    "co": _dense(next(k), d, d, 0.5), "ln2_g": jnp.ones((d,)), "ln2_b": jnp.zeros((d,)),
                    "fc1": _dense(next(k), d, 4 * d), "fc2": _dense(next(k), 4 * d, d, 0.5)} for _ in range(2)]
        plan = {"end": emb(1), "pos": emb(self.L + 1), "task": jax.random.normal(next(k), (d,)) * 0.02, "blocks": pblocks,
                "lnf_g": jnp.ones((d,)), "lnf_b": jnp.zeros((d,)), "out": _dense(next(k), d, d)}
        return {"enc": enc, "goal": goal, "wm": wm, "reach": reach, "pi": pi, "plan": plan}

    def init_carry(self, n):
        return jnp.zeros((n, self.width))

    # ------------------------------------------------------------------ encoder (goal-independent)
    def _tokens(self, e, o):
        typ, s = e["type"], self.s
        grid = jax.nn.one_hot(o["grid"].reshape(GRID_H, GRID_W * GRID_W), N_CAT).transpose(1, 0, 2).reshape(GRID_W * GRID_W, -1)
        cols = _ap(e["col"], grid) + e["col_pos"] + typ[0]
        me = (_ap(e["self"], o["self"]) + _ap(e["clock"], o["clock"]) + typ[1])[None]
        inv = e["item"][o["inv_item"]] + _ap(e["inv_n"], o["inv_count"][:, None]) + typ[2]
        ent = e["ent"][o["ent_type"]] + _ap(e["ent_f"], o["ent_feat"]) + typ[3]
        prog = (_ap(e["prog"], o["progress"]) + typ[4])[None]
        kil = (_ap(e["killed"], o["killed"]) + typ[5])[None]
        last = (e["reason"][o["last_reason"]] + _ap(e["dt"], jnp.log1p(o["dt_ticks"])[None] / 5.0) + typ[6])[None]
        vh, vw = s["view_shape"]
        P = s["view_patches"]
        ray = jnp.concatenate([jax.nn.one_hot(o["view_cat"], N_CAT), o["view_dist"][:, None]], -1)          # (R, 12)
        pat = lambda a: a.reshape(vh // 2, 2, vw // 2, 2, -1).transpose(0, 2, 1, 3, 4).reshape(P, 4, -1)
        vit = pat(e["item"][o["view_item"]])                                                                 # (P, 4, d)
        view = _ap(e["view"], pat(ray).reshape(P, -1)) + vit.mean(1) + e["view_pos"] + typ[7]
        cand = e["cmd"][o["cand_cmd"]] + e["item"][o["cand_item"]] + e["ent"][o["cand_ent"]] + _ap(e["cand_f"], o["cand_feat"]) + typ[8]
        cls = (e["cls"] + typ[9])[None]
        toks = [cols, me, inv, ent, prog, kil, last, view, cand, cls]
        one = lambda n: jnp.ones((n,), jnp.bool_)
        vv = jnp.broadcast_to(o["view_valid"], (P,))
        valid = [one(cols.shape[0]), one(1), o["inv_item"] > 0, o["ent_type"] > 0, one(1), one(1), one(1), vv, o["cand_mask"], one(1)]
        return jnp.concatenate(toks, 0), jnp.concatenate(valid, 0), sum(t.shape[0] for t in toks[:8])

    def encode(self, e, o):
        """One observation -> (tokens (L, d), key mask (L,), candidate tokens (K, d), CLS (d,))."""
        d, nh = self.d, self.heads
        x, kv, c0 = self._tokens(e, o)
        L = x.shape[0]
        for b in e["blocks"]:
            h = _ln(x, b["ln1_g"], b["ln1_b"])
            q, kk, v = jnp.split(_ap(b["qkv"], h), 3, -1)
            sp = lambda z: z.reshape(L, nh, d // nh).transpose(1, 0, 2)
            q, kk, v = sp(q), sp(kk), sp(v)
            att = jax.nn.softmax(jnp.where(kv[None, None, :], jnp.einsum("hqd,hkd->hqk", q, kk) / jnp.sqrt(d // nh), -1e9), -1)
            x = x + _ap(b["o"], jnp.einsum("hqk,hkd->hqd", att, v).transpose(1, 0, 2).reshape(L, d))
            h = _ln(x, b["ln2_g"], b["ln2_b"])
            x = x + _ap(b["fc2"], jax.nn.gelu(_ap(b["fc1"], h)))
        x = _ln(x, e["lnf_g"], e["lnf_b"])
        K = o["cand_mask"].shape[0]
        return x, kv, x[c0:c0 + K], x[L - 1]

    # ------------------------------------------------------------------ goals
    def goal_id_emb(self, P, kind, gid):
        e, g = P["enc"], P["goal"]
        cl = lambda t: t[jnp.clip(gid, 0, t.shape[0] - 1)]
        k = kind[..., None]
        return jnp.select([k == GK["item"], k == GK["kill"], k == GK["adv"]], [cl(e["item"]), cl(e["ent"]), cl(g["adv"])],
                          jnp.zeros(gid.shape + (self.d,)))

    def goal_vec(self, P, o, x, kv):
        """Mean over the goal slots of (goal embedding + its cross-attention readout over the encoded tokens); 0 without goal."""
        g = P["goal"]
        eg = g["kind"][o["goal_kind"]] + self.goal_id_emb(P, o["goal_kind"], o["goal_id"]) + _ap(g["feat"], o["goal_feat"])
        q = _ap(g["q"], _ln(eg, g["ln_g"], g["ln_b"]))
        kk, v = jnp.split(_ap(g["kv"], x), 2, -1)
        att = jax.nn.softmax(jnp.where(kv[None, :], q @ kk.T / jnp.sqrt(self.d), -1e9), -1)
        gv = eg + _ap(g["o"], att @ v)
        m = o["goal_mask"].astype(gv.dtype)
        return (gv * m[:, None]).sum(0) / jnp.maximum(m.sum(), 1.0)

    # ------------------------------------------------------------------ world model
    def wm_z(self, w, c):
        return jax.nn.gelu(_ap(w["z2"], jax.nn.gelu(_ap(w["z1"], c))))

    def wm_heads(self, P, z):
        """z (..., d) -> ok logit, log dt, d_self (all (...,) / (..., self))."""
        w = P["wm"]
        return _ap(w["ok"], z)[..., 0], _ap(w["dt"], z)[..., 0], _ap(w["dself"], z)

    def copy_gates(self, P, z):
        """(alpha, beta): learned change of the count of the item / entity the candidate names."""
        g = _ap(P["wm"]["copy"], z)
        return g[..., 0], g[..., 1]

    def d_inv(self, P, z, held, item):
        """Predicted change of every inventory count (V,) for one candidate z (d,) naming `item`: never below -held
        (inventory balance)."""
        alpha = self.copy_gates(P, z)[0] * (item > 0)
        raw = P["wm"]["inv_out"] @ z + P["wm"]["inv_b"] + alpha * jax.nn.one_hot(item, self.s["item"])
        return floor_st(INV_SCALE * raw, -held)

    def d_kill(self, P, z, ent):
        beta = self.copy_gates(P, z)[1] * (ent > 0)
        return floor_st(INV_SCALE * (P["wm"]["kill_out"] @ z + P["wm"]["kill_b"] + beta * jax.nn.one_hot(ent, self.s["entity"])), 0.0)

    def goal_gain(self, P, o, z):
        """Predicted progress (0..1) of each candidate (K, d) towards the goal tokens, from the world model only."""
        held = held_vector(o["inv_item"], o["inv_count"], self.s["item"])
        gid, kind, m = o["goal_id"], o["goal_kind"], o["goal_mask"]
        E_i = P["wm"]["inv_out"][jnp.clip(gid, 0, self.s["item"] - 1)]      # (G, d)
        E_e = P["wm"]["kill_out"][jnp.clip(gid, 0, self.s["entity"] - 1)]
        alpha, beta = self.copy_gates(P, z)                                    # (K,)
        ci, ce = o["cand_item"][:, None], o["cand_ent"][:, None]               # (K, 1)
        di = jnp.maximum(INV_SCALE * (z @ E_i.T + P["wm"]["inv_b"][jnp.clip(gid, 0, self.s["item"] - 1)]
                                      + alpha[:, None] * ((ci == gid[None]) & (ci > 0))), -held[jnp.clip(gid, 0, self.s["item"] - 1)])
        dk = jax.nn.relu(INV_SCALE * (z @ E_e.T + P["wm"]["kill_b"][jnp.clip(gid, 0, self.s["entity"] - 1)]
                                      + beta[:, None] * ((ce == gid[None]) & (ce > 0))))
        delta = jnp.maximum(jnp.expm1(o["goal_feat"][:, 0]), 1.0)
        gain = jnp.where(kind == GK["item"], di, jnp.where(kind == GK["kill"], dk, 0.0)) / delta
        return (jnp.clip(gain, 0.0, 1.0) * m).sum(-1)

    # ------------------------------------------------------------------ reachability (hindsight)
    def reach_logit(self, P, cls, kind, gid, delta):
        """cls (d,), kind/gid/delta (M,) -> logits (M,) of P(goal reached within H steps)."""
        r, g = P["reach"], P["goal"]
        eg = g["kind"][kind] + self.goal_id_emb(P, kind, gid) + _ap(r["delta"], jnp.log1p(delta)[:, None])
        hcat = jnp.concatenate([jnp.broadcast_to(cls, eg.shape), eg], -1)
        return _ap(r["r2"], jax.nn.gelu(_ap(r["r1"], hcat)))[..., 0]

    # ------------------------------------------------------------------ plan (batch generation of subgoals)
    def n_dims(self):
        s = self.s
        return 1 + s["item"] + s["entity"] + s["progress"] + 1          # END, items, kills, advancements/progress, night

    def dim_table(self, P):
        """Embedding of every goal-space dimension (index 0 = END), the same as a goal token's kind + id embedding."""
        e, g = P["enc"], P["goal"]
        k = g["kind"]
        return jnp.concatenate([P["plan"]["end"], e["item"] + k[GK["item"]], e["ent"] + k[GK["kill"]], g["adv"] + k[GK["adv"]],
                                k[GK["night"]][None]], 0)

    def plan_logits(self, P, x, kv, task, prefix):
        """Teacher-forced decoder: x/kv encoded state, task (int) dimension index of T, prefix (L,) dimension indices.
        Output (L, n_dims): row j predicts plan token j from T and prefix[:j]."""
        d, nh, L = self.d, self.heads, self.L
        pl, tab = P["plan"], self.dim_table(P)
        inp = jnp.concatenate([(tab[task] + pl["task"])[None], tab[prefix[:L - 1]]], 0) + pl["pos"][:L]
        causal = jnp.tril(jnp.ones((L, L), jnp.bool_))
        h = inp
        sp = lambda z, n: z.reshape(n, nh, d // nh).transpose(1, 0, 2)
        for b in pl["blocks"]:
            y = _ln(h, b["ln1_g"], b["ln1_b"])
            q, kk, v = jnp.split(_ap(b["qkv"], y), 3, -1)
            att = jax.nn.softmax(jnp.where(causal[None], jnp.einsum("hqd,hkd->hqk", sp(q, L), sp(kk, L)) / jnp.sqrt(d // nh), -1e9), -1)
            h = h + _ap(b["o"], jnp.einsum("hqk,hkd->hqd", att, sp(v, L)).transpose(1, 0, 2).reshape(L, d))
            y = _ln(h, b["lnc_g"], b["lnc_b"])
            ck, cv = jnp.split(_ap(b["ckv"], x), 2, -1)
            Lx = x.shape[0]
            att = jax.nn.softmax(jnp.where(kv[None, None, :], jnp.einsum("hqd,hkd->hqk", sp(_ap(b["cq"], y), L), sp(ck, Lx)) / jnp.sqrt(d // nh), -1e9), -1)
            h = h + _ap(b["co"], jnp.einsum("hqk,hkd->hqd", att, sp(cv, Lx)).transpose(1, 0, 2).reshape(L, d))
            y = _ln(h, b["ln2_g"], b["ln2_b"])
            h = h + _ap(b["fc2"], jax.nn.gelu(_ap(b["fc1"], y)))
        h = _ln(h, pl["lnf_g"], pl["lnf_b"])
        return _ap(pl["out"], h) @ tab.T / jnp.sqrt(d)

    def plan_loss_one(self, P, o, task, seq, mask):
        x, kv, _, _ = self.encode(P["enc"], o)
        lp = jax.nn.log_softmax(self.plan_logits(P, x, kv, task, seq), -1)
        nll = -jnp.take_along_axis(lp, seq[:, None], -1)[:, 0]
        hit = (jnp.argmax(lp, -1) == seq).astype(jnp.float32)
        m = mask.astype(jnp.float32)
        return (nll * m).sum() / jnp.maximum(m.sum(), 1.0), (hit * m).sum(), m.sum()

    def plan_loss(self, P, obs, task, seq, mask):
        nll, hit, n = jax.vmap(lambda o, t, q, m: self.plan_loss_one(P, o, t, q, m))(obs, task, seq, mask)
        return nll.mean(), hit.sum() / jnp.maximum(n.sum(), 1.0)

    # ------------------------------------------------------------------ policy step
    def _one(self, P, o):
        x, kv, c, cls = self.encode(P["enc"], o)
        gvec = self.goal_vec(P, o, x, kv)
        z = self.wm_z(P["wm"], c)
        okl, ldt, _ = self.wm_heads(P, z)
        feats = WM_FEAT_SCALE * jax.lax.stop_gradient(jnp.stack([jax.nn.sigmoid(okl), ldt / 5.0, self.goal_gain(P, o, z)], -1))
        return c, cls, gvec, feats

    def recur(self, P, h, c, cls, gvec, feats, cand_cmd, cand_mask):
        """The part of a step that depends on the recurrent state h: GRU, candidate logits, value. Everything it reads from the
        observation comes from _one (the encoder does not see h), so a whole segment can be encoded at once and only this part
        run in the time loop. Batched over the leading axis: c (N, K, d), h (N, width)."""
        pi = P["pi"]
        inp = cls + _ap(pi["g_in"], gvec)
        ir, iz, inn = jnp.split(_ap(pi["gru_wi"], inp), 3, -1)
        hr, hz, hn = jnp.split(_ap(pi["gru_wh"], h), 3, -1)
        r, zg = jax.nn.sigmoid(ir + hr), jax.nn.sigmoid(iz + hz)
        h2 = (1.0 - zg) * jnp.tanh(inn + r * hn) + zg * h
        sd = jnp.sqrt(self.d)
        logits = (_ap(pi["logit"], c)[..., 0] + jnp.einsum("nd,nkd->nk", _ap(pi["h_q"], h2), c) / sd
                  + jnp.einsum("nd,nkd->nk", _ap(pi["g_q"], gvec), c) / sd + feats @ pi["u"])
        if self.kind_prior:
            n_kind = jnp.sum((cand_cmd[:, :, None] == cand_cmd[:, None, :]) & cand_mask[:, None, :], -1)
            logits = logits - jnp.log(jnp.maximum(n_kind, 1).astype(logits.dtype))
        logits = jnp.where(cand_mask, logits, -1e9)
        return logits, _ap(pi["value"], h2)[:, 0], h2

    def step(self, P, h, obs):
        """Batched: obs = dict of (N, ...) arrays incl. goal fields; h (N, width).
        Returns logits (N, K) (-1e9 on padding), value (N,), next h, encoded CLS (N, d) for the reach head."""
        c, cls, gvec, feats = jax.vmap(lambda o: self._one(P, o))(obs)
        logits, value, h2 = self.recur(P, h, c, cls, gvec, feats, obs["cand_cmd"], obs["cand_mask"])
        return logits, value, h2, cls

    def encode_many(self, P, obs, chunk=512):
        """_one over N observations (leading axis) in chunks of `chunk` under jax.checkpoint: the matrices are large (chunk x 194
        tokens) and the memory of a backward pass stays that of one chunk, however many observations there are. N is padded to a
        multiple of `chunk` with copies of the first one (dropped again)."""
        n = next(iter(obs.values())).shape[0]
        chunk = min(chunk, n)
        pad = (-n) % chunk
        if pad:
            obs = {k: jnp.concatenate([v, jnp.broadcast_to(v[:1], (pad,) + v.shape[1:])]) for k, v in obs.items()}
        grid = {k: v.reshape((-1, chunk) + v.shape[1:]) for k, v in obs.items()}
        one = jax.checkpoint(lambda o: jax.vmap(lambda q: self._one(P, q))(o))
        out = jax.lax.map(one, grid)
        return tuple(z.reshape((-1,) + z.shape[2:])[:n] for z in out)

    def unroll(self, P, h0, obs, chunk=512):
        """A batch of segments: h0 (B, width), obs of (B, T, ...). The encoder runs once over all B x T observations
        (encode_many), then a scan over T carries only the recurrence. Same values as T successive step() calls.
        Returns logits (B, T, K), value (B, T)."""
        B, T = h0.shape[0], next(iter(obs.values())).shape[1]
        flat = {k: v.reshape((B * T,) + v.shape[2:]) for k, v in obs.items()}
        enc = self.encode_many(P, flat, chunk)
        xs = tuple(jnp.swapaxes(z.reshape((B, T) + z.shape[1:]), 0, 1) for z in enc[:1] + enc[1:] + (flat["cand_cmd"], flat["cand_mask"]))

        def body(h, x):
            logits, value, h2 = self.recur(P, h, *x)
            return h2, (logits, value)
        _, (logits, value) = jax.lax.scan(body, h0, xs)
        return jnp.swapaxes(logits, 0, 1), jnp.swapaxes(value, 0, 1)

    # ------------------------------------------------------------------ auxiliary (supervised) losses
    def aux_loss_one(self, P, o, a):
        """World model + reachability losses for one logged transition (unbatched). `a` holds the action index, labels from
        the next observation (ok, dt, next inventory / self / killed) and hindsight goals (her_kind/id/delta/label/mask)."""
        x, kv, c, cls = self.encode(P["enc"], o)
        z = self.wm_z(P["wm"], c[a["act"]])
        okl, ldt, ds = self.wm_heads(P, z)
        V = self.s["item"]
        held, held2 = held_vector(o["inv_item"], o["inv_count"], V), held_vector(a["n_inv_item"], a["n_inv_count"], V)
        dinv = jnp.clip(held2 - held, -64.0, 64.0)
        dinv_hat = self.d_inv(P, z, held, o["cand_item"][a["act"]])
        dkill = (a["n_killed"] - o["killed"]) * 8.0
        wv, kvd = a["wm_valid"], a["kill_valid"]
        l_ok = wv * (jax.nn.softplus(okl) - a["ok"] * okl)
        l_dt = wv * (ldt - jnp.log1p(a["dt"])) ** 2
        l_self = wv * jnp.mean((ds - (a["n_self"] - o["self"])) ** 2)
        l_inv = wv * jnp.sum(huber((dinv_hat - dinv) / INV_SCALE))
        l_kill = wv * kvd * jnp.sum(huber((self.d_kill(P, z, o["cand_ent"][a["act"]]) - dkill) / INV_SCALE))
        rl = self.reach_logit(P, cls, a["her_kind"], a["her_id"], a["her_delta"])
        l_reach = jnp.sum(a["her_mask"] * (jax.nn.softplus(rl) - a["her_label"] * rl)) / jnp.maximum(a["her_mask"].sum(), 1.0)
        pos = dinv > 0.5
        # tie-aware: a hit when the predicted top item is one of the items with the largest gain
        gained_ok = jnp.where(pos.any(), (dinv[jnp.argmax(dinv_hat)] >= dinv.max()).astype(jnp.float32), 0.0)
        return {"ok": l_ok, "dt": l_dt, "self": l_self, "inv": l_inv, "kill": l_kill, "reach": l_reach,
                "_ok_p": jax.nn.sigmoid(okl), "_gain_has": wv * pos.any(), "_gain_hit": wv * gained_ok, "_reach_p": jax.nn.sigmoid(rl)}

    AUX_W = {"ok": 1.0, "dt": 0.25, "self": 1.0, "inv": 0.5, "kill": 0.5, "reach": 1.0}

    def aux_loss(self, P, obs, lab):
        out = jax.vmap(lambda o, a: self.aux_loss_one(P, o, a))(obs, lab)
        nv = jnp.maximum(lab["wm_valid"].sum(), 1.0)
        terms = {k: out[k].sum() / (nv if k != "reach" else out[k].shape[0]) for k in self.AUX_W}
        return sum(self.AUX_W[k] * v for k, v in terms.items()), terms, out

    def param_counts(self, P):
        return n_params(P), n_params({k: v for k, v in P.items() if k != "reach"})


def plan_beam(model, params, enc_fn, dec_fn, o, task: int, beam: int = 4, forbid=()) -> list[tuple[list[int], float]]:
    """Batch generation of one plan by beam search: up to model.L tokens (dimension indices, END = 0 stops a beam), no
    dimension twice, END not first, `forbid` (e.g. dimensions already at their goal) never. enc_fn(params, o) -> (x, kv);
    dec_fn(params, x, kv, task, prefixes (B, L)) -> logits (B, L, n_dims). Returns [(plan without END, log-prob)] best first."""
    x, kv = enc_fn(params, o)
    L, V = model.L, model.n_dims()
    beams, done = [([], 0.0)], []
    for t in range(L):
        pre = np.zeros((len(beams), L), np.int32)
        for i, (sq, _) in enumerate(beams):
            pre[i, :len(sq)] = sq
        lp = np.asarray(jax.nn.log_softmax(dec_fn(params, x, kv, task, pre)[:, t], -1))
        cand = []
        for i, (sq, sc) in enumerate(beams):
            row = lp[i].copy()
            row[list(sq) + list(forbid)] = -np.inf
            if t == 0:
                row[0] = -np.inf
            for j in np.argsort(-row)[:beam]:
                if np.isfinite(row[j]):
                    cand.append((sq + [int(j)], sc + float(row[j])))
        cand.sort(key=lambda c: -c[1])
        beams = []
        for sq, sc in cand:
            (done if sq[-1] == 0 else beams).append((sq, sc))
            if len(beams) >= beam:
                break
        if not beams:
            break
    done += beams
    done.sort(key=lambda c: -c[1])
    return [([d for d in sq if d != 0], sc) for sq, sc in done[:beam]]


OBS_KEYS = ("grid", "self", "clock", "inv_item", "inv_count", "ent_type", "ent_feat", "progress", "killed", "view_cat", "view_item",
            "view_dist", "cand_cmd", "cand_item", "cand_ent", "cand_feat", "cand_mask", "last_reason", "dt_ticks")
INT_KEYS = {"grid", "inv_item", "ent_type", "cand_cmd", "cand_item", "cand_ent", "last_reason", "view_cat", "view_item",
            "goal_kind", "goal_id"}
REQUIRED = ("grid", "self", "inv_item", "inv_count", "ent_type", "ent_feat", "progress", "cand_cmd", "cand_item", "cand_ent",
            "cand_feat", "cand_mask", "last_reason", "dt_ticks")


def sizes_from_hello(hello: dict, G: int = 2) -> dict:
    """Model sizes from the environment's hello. v1.3 hellos (no view/killed/clock) are accepted for offline pretraining: the
    missing fields are filled with zeros and masked (view_valid False, kill loss off)."""
    spec = {f["name"]: f["shape"] for f in hello["obs_spec"]["fields"]}
    missing = set(REQUIRED) - set(spec)
    if missing:
        raise ValueError(f"hello lacks observation fields {sorted(missing)}")
    v = hello["vocab_sizes"]
    vs = tuple(hello.get("view_shape", (8, 16)))
    if vs[0] % 2 or vs[1] % 2:
        raise ValueError(f"view shape {vs} must be even (2x2 patches)")
    return {"self": spec["self"][0], "item": v["item"], "entity": v["entity"], "cmd": v["cmd"], "reason": v["reason"],
            "ent_f": spec["ent_feat"][1], "cand_f": spec["cand_feat"][1], "progress": spec["progress"][0], "K": spec["cand_mask"][0],
            "clock": spec.get("clock", [2])[0], "view_shape": vs, "view_patches": vs[0] * vs[1] // 4, "G": G,
            "has_view": "view_cat" in spec, "has_killed": "killed" in spec}
