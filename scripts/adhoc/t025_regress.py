"""TASK-025 regression: 2 PPO updates of E1 / G4 with new options off -> hash of final params (compare across commits); smoke of tp/ent_final."""
import hashlib, json, sys
sys.path.insert(0, ".")
import jax, numpy as np
from src.model.epa_policies import ARM_BUILDERS
from src.pipeline.epa_harness import PPOConfig, Trainer

def run(arm_key, kw, **cf):
    arm = ARM_BUILDERS[arm_key](**kw)
    cfg = PPOConfig(total_steps=2 * 8 * 16, num_envs=8, num_steps=16, epochs=2, minibatches=2, lr=1e-3, gamma=0.97, lam=0.625,
                    vf=1.0, max_grad_norm=0.5, value_norm=0.99, adv_norm="batch", warmup=0.05, **cf)
    tr = Trainer(arm, cfg)
    params, curve, _ = tr.train(5)
    h = hashlib.sha256(b"".join(np.asarray(x).tobytes() for x in jax.tree_util.tree_leaves(params))).hexdigest()[:16]
    print(arm.name, cf, h, "wm/aux col:", np.round(curve[:, 3], 3).tolist(), "params:", arm.param_counts(params))

if __name__ == "__main__":
    run("tf_gru", dict(mem_token=True, prev_act=True, head_skip=True, width=128))
    run("ppo_gru", dict(ln=True, skip=True))
    if len(sys.argv) > 1:
        run("tf_gru", dict(mem_token=True, prev_act=True, head_skip=True, width=128, tp_coef=0.1))
        run("tf_gru", dict(mem_token=True, prev_act=True, head_skip=True, width=128, tp_coef=0.1, tp_src="head"))
        run("ppo_gru", dict(ln=True, skip=True, tp_coef=0.1))
        run("tf_gru", dict(mem_token=True, prev_act=True, head_skip=True, width=128), ent_final=0.0)
        run("tf_gru", dict(mem_token=True, prev_act=True, head_skip=True, width=128, ev_coef=0.3))
        run("tf_gru", dict(mem_token=True, prev_act=True, head_skip=True, width=128, ev_coef=0.3, tp_coef=0.1))
        run("ppo_gru", dict(ln=True, skip=True, ev_coef=0.3))
