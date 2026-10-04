"""Behaviour diagnostics for a trained Phase-1 policy (TASK-20261004-023, proposal 2). Never used for training or selection.

Replays EXACTLY the trajectories of `epa_harness.evaluate_first_episodes` (same keys, chunking, sampled policy, first episode
of each fresh env) but steps the env with `step_env` (no auto-reset, with the same per-step key that `step()` passes to it),
so the terminal state of each first episode stays visible. Per episode it records:
  * death: step, light level, asleep or not, food/drink/energy/health before the fatal step, and cause flags
    (lava = standing on lava; zombie = a zombie attacked this step; necessity = food, drink or energy was 0;
    arrow = health loss not explained by the other sources). Flags are a best-effort reading of the env rules
    (craftax_classic/game_logic.py), not a metric.
  * crafting readiness: steps spent next to a crafting table (8-neighbourhood, the env's own is_near_block) and how many of
    them with >= 1 wood, >= 1 stone, or both (both + table are what the stone pickaxe/sword need), max wood/stone held.
  * achievements incl. the death tick (the headline evaluator cannot see that tick, see masked_achievements).
Simulator state is read only here, after training, for analysis (P3 concerns decision time; nothing flows back).
"""
import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_classic.constants import BlockType
from craftax.craftax_classic.game_logic import is_near_block

from src.environment.craftax_env_adapter import NUM_ACHIEVEMENTS, masked_achievements
from src.pipeline.epa_harness import _ENV, _PARAMS, EPISODE_LIMIT, env_reset, summarize_achievements

CAUSES = ("lava", "zombie", "arrow", "necessity", "timeout", "other", "alive")


def _step_no_reset(keys, st, act):
    """`step_env` with the key that the auto-reset `step()` would give it -> identical transition, terminal state kept."""
    ks = jax.vmap(lambda k: jax.random.split(k)[0])(keys)
    obs, st2, rew, done, _ = jax.vmap(_ENV.step_env, in_axes=(0, 0, 0, None))(ks, st, act, _PARAMS)
    return obs.astype(jnp.float32), st2, done


def _death_flags(s, s2, timeout):
    """Cause flags for the step s -> s2 (single env)."""
    pos = s2.player_position
    lava = s2.map[pos[0], pos[1]] == BlockType.LAVA.value
    zpos = s2.zombies.position
    zhit = s2.zombies.mask & (jnp.abs(zpos - pos[None]).sum(-1) == 1) & (s2.zombies.attack_cooldown == 5)   # cooldown reset on attack
    n_z = zhit.sum()
    nec = (s.player_food <= 0) | (s.player_drink <= 0) | ((s.player_energy <= 0) & ~s.is_sleeping)
    dmg = (s.player_health - s2.player_health).astype(jnp.float32)
    explained = n_z * jnp.where(s.is_sleeping, 7.0, 2.0) + nec.astype(jnp.float32)
    arrow = (dmg - explained >= 2.0) & ~lava
    zombie = (n_z > 0) & ~lava
    necessity = nec & ~lava & ~zombie & ~arrow
    other = ~(lava | zombie | arrow | necessity | timeout)
    return jnp.stack([lava, zombie, arrow, necessity, timeout, other])


def make_diag_chunk(arm, chunk):
    near_table = jax.vmap(lambda s: is_near_block(s, BlockType.CRAFTING_TABLE.value))

    @jax.jit
    def diag_chunk(params, carry, env_st, obs, acc, key):
        def body(c, k):
            carry, env_st, obs, acc = c
            k1, k2 = jax.random.split(k)
            logits, _, caux = arm.step(params, carry, obs)
            a = jax.random.categorical(k1, logits)
            nobs, st2, d = _step_no_reset(jax.random.split(k2, obs.shape[0]), env_st, a)
            alive = acc["alive"]
            af = alive.astype(jnp.float32)
            inv = env_st.inventory
            wood, stone = inv.wood >= 1, inv.stone >= 1
            nt = near_table(env_st) & alive
            timeout = st2.timestep >= _PARAMS.max_timesteps
            flags = jax.vmap(_death_flags)(env_st, st2, timeout)             # (n, 6)
            dying = alive & d
            acc = dict(
                alive=alive & ~d,
                length=acc["length"] + alive.astype(jnp.int32),
                ach_masked=jnp.maximum(acc["ach_masked"], masked_achievements(st2.achievements, d[:, None], alive[:, None])),
                ach_full=jnp.maximum(acc["ach_full"], st2.achievements.astype(jnp.float32) * af[:, None]),
                near_table=acc["near_table"] + nt,
                near_table_wood=acc["near_table_wood"] + (nt & wood),
                near_table_stone=acc["near_table_stone"] + (nt & stone),
                near_table_both=acc["near_table_both"] + (nt & wood & stone),
                held_both=acc["held_both"] + (alive & wood & stone),
                max_wood=jnp.maximum(acc["max_wood"], jnp.where(alive, inv.wood, 0)),
                max_stone=jnp.maximum(acc["max_stone"], jnp.where(alive, inv.stone, 0)),
                death_flags=jnp.where(dying[:, None], flags, acc["death_flags"]),
                death_light=jnp.where(dying, env_st.light_level, acc["death_light"]),
                death_asleep=jnp.where(dying, env_st.is_sleeping, acc["death_asleep"]),
                death_vitals=jnp.where(dying[:, None], jnp.stack([env_st.player_health, env_st.player_food, env_st.player_drink,
                                                                  env_st.player_energy], -1).astype(jnp.float32),
                                       acc["death_vitals"]),
            )
            return (arm.advance(caux, obs, a, d), st2, nobs, acc), None

        c, _ = jax.lax.scan(body, (carry, env_st, obs, acc), jax.random.split(key, chunk))
        return c

    return diag_chunk


def diagnose_first_episodes(arm, params, key, n_envs=256, chunk=250, max_steps=EPISODE_LIMIT + 1):
    """Same key schedule as evaluate_first_episodes (k0 -> resets, k1 -> one subkey per chunk). Returns per-episode arrays."""
    fn = make_diag_chunk(arm, chunk)
    k0, k1 = jax.random.split(key)
    obs, env_st = env_reset(jax.random.split(k0, n_envs))
    carry = arm.init_carry(n_envs)
    z_i, z_f = jnp.zeros((n_envs,), jnp.int32), jnp.zeros((n_envs,), jnp.float32)
    acc = dict(alive=jnp.ones((n_envs,), jnp.bool_), length=z_i, ach_masked=jnp.zeros((n_envs, NUM_ACHIEVEMENTS)),
               ach_full=jnp.zeros((n_envs, NUM_ACHIEVEMENTS)), near_table=z_i, near_table_wood=z_i, near_table_stone=z_i,
               near_table_both=z_i, held_both=z_i, max_wood=z_i, max_stone=z_i,
               death_flags=jnp.zeros((n_envs, len(CAUSES) - 1), jnp.bool_), death_light=z_f,
               death_asleep=jnp.zeros((n_envs,), jnp.bool_), death_vitals=jnp.zeros((n_envs, 4)))
    steps = 0
    while steps < max_steps and bool(acc["alive"].any()):
        k1, sub = jax.random.split(k1)
        carry, env_st, obs, acc = fn(params, carry, env_st, obs, acc, sub)
        steps += chunk
    return {k: np.asarray(v) for k, v in acc.items()}


def cause_of_death(flags, alive):
    """First matching flag in CAUSES order (lava > zombie > arrow > necessity > timeout > other); 'alive' if censored."""
    out = []
    for f, a in zip(flags, alive):
        out.append("alive" if a else CAUSES[int(np.argmax(f))] if f.any() else "other")
    return out


def summarize(diag):
    """Aggregate per-episode diagnostics into the numbers the proposals need (all computed from the arrays)."""
    n = len(diag["length"])
    causes = cause_of_death(diag["death_flags"], diag["alive"])
    dead = ~diag["alive"]
    night = dead & (diag["death_light"] < 0.5)
    steps = np.maximum(diag["length"], 1)
    out = dict(
        episodes=int(n), censored=int(diag["alive"].sum()),
        length_mean=float(diag["length"].mean()), length_quantiles=[float(q) for q in np.quantile(diag["length"], [0.1, 0.25, 0.5, 0.75, 0.9])],
        cause_counts={c: int(sum(x == c for x in causes)) for c in CAUSES},
        death_at_night_frac=float(night.sum() / max(dead.sum(), 1)),
        death_asleep_frac=float((dead & diag["death_asleep"]).sum() / max(dead.sum(), 1)),
        death_vitals_mean=dict(zip(("health", "food", "drink", "energy"), [float(x) for x in diag["death_vitals"][dead].mean(0)]))
        if dead.any() else {},
        near_table_step_frac=float((diag["near_table"] / steps).mean()),
        near_table_with_both_frac=float(diag["near_table_both"].sum() / max(diag["near_table"].sum(), 1)),
        episodes_near_table_with_both=int((diag["near_table_both"] > 0).sum()),
        episodes_ever_held_wood_and_stone=int((diag["held_both"] > 0).sum()),
        max_wood_mean=float(diag["max_wood"].mean()), max_stone_mean=float(diag["max_stone"].mean()),
        headline_masked=summarize_achievements(diag["ach_masked"]), incl_death_tick=summarize_achievements(diag["ach_full"]),
    )
    return out, causes
