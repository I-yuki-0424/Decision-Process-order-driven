import sys
sys.path.insert(0, ".")
import jax, jax.numpy as jnp
from src.pipeline.epa_harness import PPOConfig, Trainer
from src.model.epa_policies import ARM_BUILDERS
kw = dict(mem_token=True, prev_act=True, head_skip=True)
res = {}
for remat in (False, True):
    cfg = PPOConfig(total_steps=16 * 32, num_envs=16, num_steps=32, epochs=1, minibatches=2, remat=remat, adv_norm="batch", value_norm=0.99)
    tr = Trainer(ARM_BUILDERS["tf_gru"](**kw), cfg)
    st = tr.init(jax.random.PRNGKey(0))
    env_st, obs, carry, ep_ret, traj, last_v = tr._rollout(st["params"], st, jax.random.PRNGKey(1))
    adv, ret = tr._gae(traj, last_v)
    mb = dict(traj, adv=adv, ret=ret, carry=traj["carry"][0:1])
    (l, aux), g = jax.value_and_grad(tr._loss_rec, has_aux=True)(st["params"], mb)
    res[remat] = (l, g)
a, b = res[False], res[True]
print("loss", float(a[0]), float(b[0]))
num = max(float(jnp.abs(x - y).max()) for x, y in zip(jax.tree_util.tree_leaves(a[1]), jax.tree_util.tree_leaves(b[1])))
den = max(float(jnp.abs(x).max()) for x in jax.tree_util.tree_leaves(a[1]))
print("max grad abs diff", num, "max grad abs", den)
