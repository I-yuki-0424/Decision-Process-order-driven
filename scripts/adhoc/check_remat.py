"""Ad-hoc: remat gives the same update as no remat; peak memory / time for batch shapes (run in the GPU container)."""
import sys, time
sys.path.insert(0, ".")
import jax, jax.numpy as jnp, numpy as np
from src.pipeline.epa_harness import PPOConfig, Trainer
from src.model.epa_policies import ARM_BUILDERS

kw = dict(mem_token=True, prev_act=True, head_skip=True)
def run(n, t, mb, remat, upd=2):
    cfg = PPOConfig(total_steps=n * t * upd, num_envs=n, num_steps=t, epochs=3, minibatches=mb, lr=1.5e-3, ent=0.003, lam=0.625, vf=1.0,
                    max_grad_norm=0.5, value_norm=0.99, adv_norm="batch", remat=remat)
    tr = Trainer(ARM_BUILDERS["tf_gru"](**kw), cfg)
    st = tr.init(jax.random.PRNGKey(0))
    t0 = time.time()
    for _ in range(upd):
        st, s = tr.update(st)
    jax.block_until_ready(st["params"])
    dt = time.time() - t0
    ms = jax.devices()[0].memory_stats() or {}
    return st["params"], dt, ms.get("peak_bytes_in_use", 0) / 2**20, np.asarray(s)

mode = sys.argv[1]
if mode == "equal":
    pa, _, _, sa = run(16, 32, 4, False)
    pb, _, _, sb = run(16, 32, 4, sys.argv[2] == "1")
    d = max(float(jnp.abs(x - y).max()) for x, y in zip(jax.tree_util.tree_leaves(pa), jax.tree_util.tree_leaves(pb)))
    print("max abs param diff remat vs not:", d, "stats", sa[:4], sb[:4])
else:
    n, t, mb, remat = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] == "1"
    p, dt, peak, s = run(n, t, mb, remat, upd=3)
    print(f"n={n} t={t} mb={mb} remat={remat} 3 updates {dt:.1f}s peak {peak:.0f} MiB  stats(kl,clip,gnorm)={s[6:]}")
