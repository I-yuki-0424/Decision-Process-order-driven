"""Localise NaN/inf in the MDP-search candidates (variant_5_1/5_4/mdp_branch) on real Craftax observations.
Runs real env steps (CraftaxObsAdapter, uniform-random actions) and real forward passes; reports the first
stage at which a non-finite value appears and the magnitude growth of Q per value-iteration step."""
import sys
sys.path.insert(0, ".")
import jax, jax.numpy as jnp, numpy as np
from src.environment.craftax_obs_adapter import CraftaxObsAdapter
from src.pipeline.candidate_benchmark import CANDIDATE_REGISTRY
from src.model.candidates import shared_stages as ss, variant_5_4_single_shot as v54, variant_5_1_shared_attention as v51
from src.model.channel_encoder import encode_channel_independent

ad = CraftaxObsAdapter(60, None, 8)
def collect(seed, T=40):
    k = jax.random.PRNGKey(seed); inp, st, ad_ = ad.reset(k); out = [inp]
    for t in range(T):
        k, a, e = jax.random.split(k, 3)
        act = jax.random.randint(a, (), 0, ad.num_actions)
        inp, st, r, d, _ = ad.step(e, st, act, ad_, step_count=t, prev_history=inp.history); out.append(inp)
    return out

def value_iter_trace(arr, k, n):
    S = jnp.clip(arr[:,0].astype(jnp.int32),0,n-1); S1 = jnp.clip(arr[:,1].astype(jnp.int32),0,n-1)
    A = jnp.clip(arr[:,2].astype(jnp.int32),0,n-1); P,R,G = arr[:,3],arr[:,4],arr[:,5]
    Q = jnp.zeros((n,n)); tr=[]
    for it in range(n):
        nxt = jnp.mean(jax.lax.top_k(Q[S1],k)[0],-1)
        Q = jax.ops.segment_sum(P*(R+G*nxt), S*n+A, num_segments=n*n).reshape(n,n)
        tr.append(float(jnp.max(jnp.abs(Q))))
    return tr

import functools
for name, fwd, init in [("variant_5_4", v54.forward_variant_5_4, v54.init_variant_5_4_parameters),
                        ("variant_5_1", v51.forward_variant_5_1, v51.init_variant_5_1_parameters),
                        ("variant_5_1_stable", functools.partial(v51.forward_variant_5_1, stable_mdp=True), v51.init_variant_5_1_parameters),
                        ("variant_5_4_stable", functools.partial(v54.forward_variant_5_4, stable_mdp=True), v54.init_variant_5_4_parameters)]:
    stats = dict(n=0, nonfinite_logits=0, nonfinite_entropy=0, nonfinite_heads=0, id0_next=0, maxQ=0.0)
    for seed in range(4):
        params = init(jax.random.PRNGKey(seed), d_model=32, num_actions=ad.num_actions, action_feat_dim=ad.action_feat_dim,
                      num_costs=ad.num_costs, num_resources=ad.num_resources, target_dim=8)
        for inp in collect(seed):
            d = fwd(params, inp)
            lg = d.action_logits; stats["n"] += 1
            stats["nonfinite_logits"] += int(not bool(jnp.all(jnp.isfinite(lg))))
            lp = jax.nn.log_softmax(lg); ent = -jnp.sum(jnp.exp(lp)*lp)
            stats["nonfinite_entropy"] += int(not bool(jnp.isfinite(ent)))
            stats["nonfinite_heads"] += int(not all(bool(jnp.all(jnp.isfinite(x))) for x in
                (d.estimated_costs, d.predicted_next_state, d.progress_rate_pred, d.validity_score)))
            fin = lg[lg > -1e8]; stats["maxQ"] = max(stats["maxQ"], float(jnp.max(jnp.abs(jnp.where(jnp.isfinite(fin), fin, 0.)))) if fin.size else 0.)
    print(name, stats, flush=True)

# Q growth trace on a synthetic id table typical of the code path (next id = 0 => gradient clipped to 1e4)
n = ad.num_actions
ids = jnp.arange(n)
arr = ss.build_mdp_transition_array(jnp.full((n,), 5), jnp.zeros((n,), jnp.int32), ids, 0.1, 0.5, 0.9)
print("gradient@next_id=0 row:", np.asarray(arr[0]))
print("max|Q| per value-iteration step (next_id=0):", value_iter_trace(arr, 3, n))
arr2 = ss.build_mdp_transition_array(jnp.full((n,), 5), jnp.full((n,), 5), ids, 0.1, 0.5, 0.9)
print("max|Q| per step (next_id==state, gradient=100):", value_iter_trace(arr2, 3, n))
