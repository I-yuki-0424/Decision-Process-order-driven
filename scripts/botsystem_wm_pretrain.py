#!/usr/bin/env python3
"""Offline pretraining and evaluation (E1) of GoalWM's encoder, world model and reach head on logged BOTSYSTEM runs
(forge1201-MODED-AIPlayer exchange/<run>/{env_to_policy,policy_to_env}.jsonl). Side track (TASK-20261101-031): nothing here
counts toward a roadmap gate.

  python scripts/botsystem_wm_pretrain.py --exchange <dir> --train RUN [RUN ...] --test RUN [RUN ...] \
      --out-params goalwm_pretrain.pkl --out-json result.json [--steps 6000] [--max-train 60000] [--seed 42]

Every logged transition (observation, chosen candidate, next observation of the same bot and episode) gives the world-model
labels the online learner uses (success, duration, own-state change, inventory change); hindsight reach labels come from the
episode's next --her-horizon observations. Older runs (env v1.x) have no view/killed/clock fields: they are filled with zeros
and masked, and the kill loss is off. K is padded to the model's K.

E1 (review of 2026-11-01): on runs held out entirely (--test), compare each head with a frequency baseline fitted on --train:
  success   AUROC of p_ok vs AUROC of the per-(command, item) success rate (per-command rate as a second baseline)
  duration  mean |log1p(dt) error| vs the per-command median of log1p(dt)
  gain      top-1 of the most-gained item (transitions with a gain) vs the per-(command, item) most frequent gained item
  reach     AUROC of R(s, g) vs the per-dimension base rate of being reached within H
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from botsystem_goal_ppo import GoalSpace, hindsight, to_arrays, wm_labels  # noqa: E402
from botsystem_goal_wm import GK, GoalWM, sizes_from_hello  # noqa: E402

V2 = {"cmd": 13, "K": 64, "view_shape": (8, 16), "view_patches": 32, "clock": 2}   # the online model's sizes (env v2.0)
COMPACT = {"grid": np.int8, "inv_item": np.int16, "ent_type": np.int16, "cand_cmd": np.int8, "cand_item": np.int16,
           "cand_ent": np.int16, "last_reason": np.int8, "view_cat": np.int8, "view_item": np.int16, "n_inv_item": np.int16}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a = p.add_argument
    a("--exchange", required=True), a("--train", nargs="+", required=True), a("--test", nargs="+", required=True)
    a("--out-params", required=True), a("--out-json", required=True), a("--steps", type=int, default=6000)
    a("--batch", type=int, default=64), a("--lr", type=float, default=1e-3), a("--max-train", type=int, default=60000)
    a("--max-test", type=int, default=15000), a("--her-horizon", type=int, default=32), a("--seed", type=int, default=42)
    a("--commit", default="", help="DPOD commit of this code (the container has no git)")
    return p.parse_args(argv)


def pad_k(o: dict, K: int) -> dict:
    n = len(o["cand_mask"])
    if n == K:
        return o
    o = dict(o)
    for k, z in (("cand_cmd", 0), ("cand_item", 0), ("cand_ent", 0), ("cand_mask", False)):
        o[k] = list(o[k]) + [z] * (K - n)
    o["cand_feat"] = list(o["cand_feat"]) + [[0.0] * len(o["cand_feat"][0])] * (K - n)
    return o


def load_run(d: Path, sizes: dict, keep: float, rng, gs: GoalSpace, H: int):
    """Transitions of one run: (obs arrays, labels incl. hindsight, meta (cmd, item, ok, log dt, gained item))."""
    acts = {}
    with open(d / "policy_to_env.jsonl") as f:
        for line in f:
            m = json.loads(line)
            if m.get("type") == "act":
                acts[(m["env_id"], m["episode"], m["step"])] = m["action"]["candidate"]
    rows, eps = [], defaultdict(list)      # eps[(env, episode)] = [(vals, row index or None)]
    last = {}
    def close(key):
        seq = eps.pop(key, [])
        arch = list(gs.archive)
        for t, (vals, ri) in enumerate(seq):
            if ri is not None:
                rows[ri][1].update(hindsight(vals, [v for v, _ in seq[t + 1:t + 1 + H]], arch, rng))
    with open(d / "env_to_policy.jsonl") as f:
        for line in f:
            if '"type":"obs"' not in line.replace(" ", ""):
                continue
            m = json.loads(line)
            key, st = (m["env_id"], m["episode"]), m["step"]
            o = pad_k(m["obs"], sizes["K"])
            vals = gs.values(o)
            prev = last.get(m["env_id"])
            if prev is not None and prev["key"] != key:
                close(prev["key"])
            ri = None
            if prev is not None and prev["key"] == key and prev["step"] == st - 1 and (key + (st - 1,)) in acts:
                gs.note_raised(prev["vals"], vals)
                a = acts[key + (st - 1,)]
                if prev["o"]["cand_mask"][a] and rng.random() < keep:
                    x, x2 = to_arrays(prev["o"], sizes), to_arrays(o, sizes)
                    lab = wm_labels(x2, o, bool(m["terminated"]), a, False)
                    held = Counter({i: round(c * 64) for i, c in zip(prev["o"]["inv_item"], prev["o"]["inv_count"]) if i})
                    held2 = Counter({i: round(c * 64) for i, c in zip(o["inv_item"], o["inv_count"]) if i})
                    gain = {i: held2[i] - held[i] for i in held2 if held2[i] > held[i]}
                    meta = (int(x["cand_cmd"][a]), int(x["cand_item"][a]), float(lab["ok"]), float(np.log1p(lab["dt"])),
                            max(gain, key=gain.get) if gain and not m["terminated"] else -1)
                    small = lambda dct: {k: np.asarray(v).astype(COMPACT.get(k, np.asarray(v).dtype)) for k, v in dct.items()}
                    rows.append((small(x), small(lab), meta))
                    ri = len(rows) - 1
                if ri is not None and eps[key]:
                    eps[key][-1] = (eps[key][-1][0], ri)      # the transition starts at the previous observation
            eps[key].append((vals, None))
            last[m["env_id"]] = {"key": key, "step": st, "o": o, "vals": vals}
            if m["terminated"] or m["truncated"]:
                close(key)
                last.pop(m["env_id"], None)
    for key in list(eps):
        close(key)
    for _, lab, _ in rows:          # transitions never closed (cannot happen after the loop above; kept as a guard)
        if "her_kind" not in lab:
            lab.update(hindsight({}, [], [], rng))
    return rows


def stack(rows):
    out_o, out_l = {}, {}
    for k in rows[0][0]:
        out_o[k] = np.stack([r[0][k] for r in rows]).astype(COMPACT.get(k, rows[0][0][k].dtype))
    for k in rows[0][1]:
        out_l[k] = np.stack([r[1][k] for r in rows]).astype(COMPACT.get(k, np.asarray(rows[0][1][k]).dtype))
    meta = np.array([r[2] for r in rows], np.float64)
    return out_o, out_l, meta


def widen(d: dict) -> dict:
    return {k: (v.astype(np.int32) if v.dtype in (np.int8, np.int16) else v) for k, v in d.items()}


def auroc(score, y):
    score, y = np.asarray(score, np.float64), np.asarray(y) > 0.5
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s_sorted = score[order]
    i = 0
    while i < len(score):                       # average ranks of ties
        j = i
        while j + 1 < len(score) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main(argv=None):
    a = parse_args(argv)
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    ex = Path(a.exchange)
    hello = json.loads(open(ex / a.train[0] / "env_to_policy.jsonl").readline())
    sizes = {**sizes_from_hello(hello), **V2}
    gs = GoalSpace(sizes, hello["milestones"])
    counts = {r: sum(1 for _ in open(ex / r / "policy_to_env.jsonl")) for r in a.train + a.test}
    n_tr, n_te = sum(counts[r] for r in a.train), sum(counts[r] for r in a.test)
    train = [row for r in a.train for row in load_run(ex / r, sizes, min(1.0, a.max_train / n_tr), rng, gs, a.her_horizon)]
    test = [row for r in a.test for row in load_run(ex / r, sizes, min(1.0, a.max_test / n_te), rng, gs, a.her_horizon)]
    print(f"loaded {len(train)} train / {len(test)} test transitions in {time.time() - t0:.0f}s; archive {len(gs.archive)} dims", flush=True)
    tr_o, tr_l, tr_m = stack(train)
    te_o, te_l, te_m = stack(test)
    del train, test

    model = GoalWM(sizes)
    params = model.init(jax.random.PRNGKey(a.seed))
    opt = optax.chain(optax.clip_by_global_norm(5.0), optax.adam(optax.cosine_decay_schedule(a.lr, a.steps, 0.1)))
    ost = opt.init(params)

    @jax.jit
    def step(p, s, o, l):
        (loss, terms), g = jax.value_and_grad(lambda q: model.aux_loss(q, o, l)[:2], has_aux=True)(p)
        u, s = opt.update(g, s, p)
        return optax.apply_updates(p, u), s, loss, terms

    evalf = jax.jit(lambda p, o, l: model.aux_loss(p, o, l))
    curve = []
    for i in range(a.steps):
        idx = rng.integers(0, len(tr_m), a.batch)
        params, ost, loss, terms = step(params, ost, widen({k: v[idx] for k, v in tr_o.items()}), widen({k: v[idx] for k, v in tr_l.items()}))
        if i % max(1, a.steps // 50) == 0 or i == a.steps - 1:
            curve.append({"step": i, "loss": float(loss), **{k: float(v) for k, v in terms.items()}})
            print(json.dumps(curve[-1]), flush=True)

    # ---- E1 on the held-out runs
    outs = defaultdict(list)
    for i in range(0, len(te_m), 256):
        sl = slice(i, i + 256)
        _, _, out = evalf(params, widen({k: v[sl] for k, v in te_o.items()}), widen({k: v[sl] for k, v in te_l.items()}))
        for k in ("_ok_p", "_gain_has", "_gain_hit", "_reach_p"):
            outs[k].append(np.asarray(out[k]))
    o_ok = np.concatenate(outs["_ok_p"])
    valid = te_l["wm_valid"] > 0.5
    cmd, item, ok, ldt, gained = te_m[:, 0].astype(int), te_m[:, 1].astype(int), te_m[:, 2], te_m[:, 3], te_m[:, 4].astype(int)
    tcmd, titem, tok, tldt, tgained = tr_m[:, 0].astype(int), tr_m[:, 1].astype(int), tr_m[:, 2], tr_m[:, 3], tr_m[:, 4].astype(int)
    rate_ci, rate_c = defaultdict(list), defaultdict(list)
    for c, it, y in zip(tcmd, titem, tok):
        rate_ci[(c, it)].append(y), rate_c[c].append(y)
    glob = float(np.mean(tok))
    b_ci = np.array([np.mean(rate_ci[(c, it)]) if (c, it) in rate_ci else (np.mean(rate_c[c]) if c in rate_c else glob) for c, it in zip(cmd, item)])
    b_c = np.array([np.mean(rate_c[c]) if c in rate_c else glob for c in cmd])
    # duration: model prediction re-run cheaply on the chosen candidate
    dt_hat = []
    zf = jax.jit(jax.vmap(lambda o, act: model.wm_heads(params, model.wm_z(params["wm"], model.encode(params["enc"], o)[2][act]))[1]))
    for i in range(0, len(te_m), 256):
        sl = slice(i, i + 256)
        dt_hat.append(np.asarray(zf(widen({k: v[sl] for k, v in te_o.items()}), te_l["act"][sl].astype(np.int32))))
    dt_hat = np.concatenate(dt_hat)
    med_c = {c: float(np.median(tldt[tcmd == c])) for c in set(tcmd)}
    b_dt = np.array([med_c.get(c, float(np.median(tldt))) for c in cmd])
    # gained item: model top-1 vs per-(command, item) most frequent gained item
    g_ci = defaultdict(Counter)
    for c, it, g in zip(tcmd, titem, tgained):
        if g >= 0:
            g_ci[(c, it)][g] += 1
    has = (gained >= 0) & valid
    b_gain = np.array([(g_ci[(c, it)].most_common(1)[0][0] if g_ci[(c, it)] else -2) for c, it in zip(cmd, item)])
    m_gain_hit = np.concatenate(outs["_gain_hit"])
    # reach: per-dimension base rate from train hindsight labels
    base = defaultdict(list)
    for k, g, y, mk in zip(tr_l["her_kind"].ravel(), tr_l["her_id"].ravel(), tr_l["her_label"].ravel(), tr_l["her_mask"].ravel()):
        if mk:
            base[(int(k), int(g))].append(y)
    rm = te_l["her_mask"].ravel() > 0
    rk, rg, ry = te_l["her_kind"].ravel()[rm], te_l["her_id"].ravel()[rm], te_l["her_label"].ravel()[rm]
    r_hat = np.concatenate(outs["_reach_p"]).ravel()[rm]
    gb = float(np.mean([y for v in base.values() for y in v])) if base else 0.5
    b_reach = np.array([np.mean(base[(int(k), int(g))]) if (int(k), int(g)) in base else gb for k, g in zip(rk, rg)])

    res = {
        "task_id": "TASK-20261101-031", "kind": "offline pretraining + E1 (side track, no gate)", "protocol": "none (BOTSYSTEM side track)",
        "commit": a.commit or None,
        "model": model.name, "params_total": int(model.param_counts(params)[0]), "seed": a.seed, "steps": a.steps, "batch": a.batch, "lr": a.lr,
        "her_horizon": a.her_horizon, "train_runs": a.train, "test_runs": a.test, "n_train": int(len(tr_m)), "n_test": int(len(te_m)),
        "archive_dims": len(gs.archive),
        "success": {"model_auroc": auroc(o_ok[valid], ok[valid]), "baseline_cmd_item_auroc": auroc(b_ci[valid], ok[valid]),
                    "baseline_cmd_auroc": auroc(b_c[valid], ok[valid]), "test_success_rate": float(ok[valid].mean())},
        "duration": {"model_mae_log1p": float(np.mean(np.abs(dt_hat - ldt)[valid])), "baseline_cmd_median_mae_log1p": float(np.mean(np.abs(b_dt - ldt)[valid]))},
        "gain": {"n_with_gain": int(has.sum()), "model_top1": float(m_gain_hit[has].mean()) if has.any() else None,
                 "baseline_cmd_item_top1": float((b_gain[has] == gained[has]).mean()) if has.any() else None},
        "reach": {"n_labels": int(rm.sum()), "positive_rate": float(ry.mean()) if len(ry) else None,
                  "model_auroc": auroc(r_hat, ry), "baseline_dim_rate_auroc": auroc(b_reach, ry)},
        "curve": curve, "wall_s": round(time.time() - t0, 1),
    }
    Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out_json).write_text(json.dumps(res, indent=1))
    Path(a.out_params).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_params, "wb") as f:
        pickle.dump({"params": jax.device_get(params), "sizes": sizes, "model": model.name, "result": {k: v for k, v in res.items() if k != "curve"}}, f)
    print(json.dumps({k: v for k, v in res.items() if k != "curve"}, indent=1))


if __name__ == "__main__":
    main()
