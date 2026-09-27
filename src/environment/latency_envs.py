"""Latency decision process: the world keeps moving while the agent computes (pure JAX, jit/vmap-safe).

Cycle semantics (fixed tick dt, latency `delta` ticks, commit length `k` >= delta):
  Cycle c starts at tick t_c = c*k with an observation snapshot o_c. The plan P_c (k actions) computed from o_c only
  reaches the actuator at t_c + delta. During [t_c, t_c + delta) the env executes the committed prefix
  u_c = P_{c-1}[k-delta:k] (the tail of the previous plan; the reference action right after a reset). Then
  P_c[0:k-delta] runs until t_{c+1}, and P_c[k-delta:k] becomes u_{c+1}.
  delta = 0 is ordinary k-step action chunking; k = delta = 1 is the classic one-step-delay MDP.
  (o_c, u_c) is a Markov state of the resulting macro-step MDP (Katsikopoulos & Engelbrecht 2003), so "augment"
  (policy sees u_c) is the information-complete baseline that any forecasting arm must be compared against.

Oracle: `oracle_forecast` replays u_c on a copy of the true state with the SAME per-tick keys `cycle` will use, i.e. the
exact state at t_c + delta including future exogenous noise (perfect foresight = zero-latency information). It exists
only for the upper-bound arm and is never an input of any other arm.

Cores (CoreEnv): `pendulum_goal` (passive drift = gravity; torque-limited so reaching a goal angle needs a pumping
sequence), `intercept` (exogenous randomly turning target -- the case where the world moves independently of the agent),
`craftax` (Craftax-Classic symbolic, made real-time: mobs keep moving during the thinking ticks; also used with
delta=0, k=1 as the plain-PPO calibration environment).
"""
import math
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp


def tree_where(cond, a, b):
    return jax.tree_util.tree_map(lambda x, y: jnp.where(cond, x, y), a, b)


class CoreEnv(NamedTuple):
    """Tick-level environment. All callables are pure; `step` must NOT auto-reset (the wrapper resets)."""
    name: str
    obs_dim: int
    goal_dim: int
    num_actions: int
    ref_action: int          # reference continuation: what runs when nothing is committed (noop / zero torque / stay)
    dt: float                # seconds per tick
    max_ticks: int           # time limit per episode
    reset: Callable          # key -> core_state
    step: Callable           # (key, core_state, action) -> (core_state, reward, terminated)
    obs: Callable            # core_state -> (obs_dim,)
    goal: Callable           # core_state -> (goal_dim,)
    achieved_goal: Optional[Callable] = None  # obs -> (goal_dim,), for hindsight relabelling; None = not goal-conditioned


# ----------------------------------------------------------------------------------------------------------------------
# Cores
# ----------------------------------------------------------------------------------------------------------------------

class PendulumState(NamedTuple):
    theta: jnp.ndarray       # 0 = hanging, pi = upright
    p: jnp.ndarray           # angular velocity
    goal_theta: jnp.ndarray


def make_pendulum_goal(max_ticks: int = 200, noise: float = 0.1, goal_mode: str = "uniform",
                       n_torques: int = 5) -> CoreEnv:
    """Same plant as scripts/run_wm_planning.py (theta'' = -sin(theta) - 0.05 theta' + u, |u| <= 0.7, dt = 0.05),
    with discretised torque, process noise on theta' and a goal angle (goal_mode 'uniform' or 'upright')."""
    dt, umax = 0.05, 0.7
    torques = jnp.linspace(-umax, umax, n_torques)

    def field(x, u):
        return jnp.stack([x[1], -jnp.sin(x[0]) - 0.05 * x[1] + u])

    def rk4(x, u):
        k1 = field(x, u); k2 = field(x + dt / 2 * k1, u); k3 = field(x + dt / 2 * k2, u); k4 = field(x + dt * k3, u)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def reset(key):
        k1, k2, k3 = jax.random.split(key, 3)
        theta = jax.random.uniform(k1, (), minval=-0.3, maxval=0.3)
        p = jax.random.uniform(k2, (), minval=-0.3, maxval=0.3)
        g = jnp.pi if goal_mode == "upright" else jax.random.uniform(k3, (), minval=-jnp.pi, maxval=jnp.pi)
        return PendulumState(theta, p, jnp.asarray(g, jnp.float32))

    def step(key, s, a):
        u = torques[a]
        x = rk4(jnp.stack([s.theta, s.p]), u)
        p = x[1] + noise * math.sqrt(dt) * jax.random.normal(key)
        theta = jnp.arctan2(jnp.sin(x[0]), jnp.cos(x[0]))
        reward = -(1.0 - jnp.cos(theta - s.goal_theta)) - 0.01 * u ** 2
        return PendulumState(theta, p, s.goal_theta), reward, jnp.array(False)

    def obs(s):
        return jnp.stack([jnp.cos(s.theta), jnp.sin(s.theta), s.p / 2.0]).astype(jnp.float32)

    def goal(s):
        return jnp.stack([jnp.cos(s.goal_theta), jnp.sin(s.goal_theta)]).astype(jnp.float32)

    return CoreEnv("pendulum_goal", 3, 2, n_torques, n_torques // 2, dt, max_ticks, reset, step, obs, goal,
                   achieved_goal=lambda o: o[:2])


class InterceptState(NamedTuple):
    agent: jnp.ndarray       # (2,)
    target: jnp.ndarray      # (2,)
    heading: jnp.ndarray     # () target heading (radians)


def make_intercept(max_ticks: int = 200, agent_speed: float = 0.03, target_speed: float = 0.02,
                   turn_noise: float = 0.3, radius: float = 0.05, step_cost: float = 0.01) -> CoreEnv:
    """Unit arena; agent moves {stay, +x, -x, +y, -y}; the target moves on its own (random heading walk, reflecting
    walls). Catching (distance < radius) gives +1 and ends the episode. With latency the agent must lead the target."""
    moves = jnp.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], jnp.float32) * agent_speed

    def reset(key):
        k1, k2, k3 = jax.random.split(key, 3)
        return InterceptState(jax.random.uniform(k1, (2,), minval=0.1, maxval=0.9),
                              jax.random.uniform(k2, (2,), minval=0.1, maxval=0.9),
                              jax.random.uniform(k3, (), minval=-jnp.pi, maxval=jnp.pi))

    def step(key, s, a):
        agent = jnp.clip(s.agent + moves[a], 0.0, 1.0)
        heading = s.heading + turn_noise * jax.random.normal(key)
        vel = target_speed * jnp.stack([jnp.cos(heading), jnp.sin(heading)])
        pos = s.target + vel
        lo, hi = pos < 0.0, pos > 1.0
        pos = jnp.where(lo, -pos, jnp.where(hi, 2.0 - pos, pos))
        vel = jnp.where(lo | hi, -vel, vel)
        caught = jnp.linalg.norm(agent - pos) < radius
        reward = jnp.where(caught, 1.0, -step_cost)
        return InterceptState(agent, pos, jnp.arctan2(vel[1], vel[0])), reward, caught

    def obs(s):
        return jnp.concatenate([s.agent, s.target, jnp.stack([jnp.cos(s.heading), jnp.sin(s.heading)])]).astype(
            jnp.float32)

    return CoreEnv("intercept", 6, 0, 5, 0, 1.0, max_ticks, reset, step, obs, lambda s: jnp.zeros((0,), jnp.float32))


def make_craftax_classic(max_ticks: int = 10000) -> CoreEnv:
    """Craftax-Classic symbolic (1345-d obs, 17 actions, noop = 0). Uses step_env (no internal auto-reset)."""
    from craftax.craftax_classic.envs.craftax_symbolic_env import CraftaxClassicSymbolicEnv

    env = CraftaxClassicSymbolicEnv()
    params = env.default_params

    def reset(key):
        return env.reset_env(key, params)[1]

    def step(key, s, a):
        _, s2, r, d, _ = env.step_env(key, s, a, params)
        return s2, jnp.asarray(r, jnp.float32), jnp.asarray(d)

    obs_dim = int(env.get_obs(reset(jax.random.PRNGKey(0))).shape[0])
    return CoreEnv("craftax", obs_dim, 0, 17, 0, 1.0, max_ticks, reset, step,
                   lambda s: env.get_obs(s).astype(jnp.float32), lambda s: jnp.zeros((0,), jnp.float32))


def make_core_env(name: str, **kw) -> CoreEnv:
    makers = {"pendulum_goal": make_pendulum_goal, "intercept": make_intercept, "craftax": make_craftax_classic}
    return makers[name](**kw)


# ----------------------------------------------------------------------------------------------------------------------
# Latency wrapper
# ----------------------------------------------------------------------------------------------------------------------

class LatencyState(NamedTuple):
    core: Any
    prefix: jnp.ndarray      # (delta,) int32 committed actions for [t_c, t_c + delta)
    tick: jnp.ndarray        # () int32 ticks since episode start
    hist_obs: jnp.ndarray    # (M, D) observations of the last M executed ticks before t_c (oldest first)
    hist_act: jnp.ndarray    # (M,) actions executed at those ticks
    hist_valid: jnp.ndarray  # (M,) bool
    ep_return: jnp.ndarray   # () running episode return


class CycleOutput(NamedTuple):
    tick_obs: jnp.ndarray         # (k+1, D) obs before each tick; last = obs after the final tick (before any reset)
    tick_act: jnp.ndarray         # (k,) actions actually executed
    tick_reward: jnp.ndarray      # (k,) 0 on ticks after the episode ended
    tick_alive: jnp.ndarray       # (k,) tick executed inside the episode
    tick_end: jnp.ndarray         # (k,) episode ended (terminated or time limit) at this tick
    done: jnp.ndarray             # () episode ended during this cycle (the wrapper then reset the env)
    terminated: jnp.ndarray       # () ...by a terminal state rather than the time limit
    finished_return: jnp.ndarray  # () return of the episode that ended in this cycle, 0 if none


def charged_latency(base_delta: int, sequential_calls: int, calls_per_tick: Optional[float]) -> int:
    """Compute-as-latency: planner cost (sequential forward passes) converted to ticks of the moving world."""
    if not calls_per_tick:
        return int(base_delta)
    return int(base_delta) + int(math.ceil(sequential_calls / calls_per_tick))


class LatencyEnv:
    def __init__(self, core: CoreEnv, delta: int, k: int, hist_len: int = 0):
        if not (k >= 1 and 0 <= delta <= k):
            raise ValueError(f"need k >= 1 and 0 <= delta <= k, got delta={delta}, k={k}")
        self.core, self.delta, self.k, self.hist_len = core, int(delta), int(k), int(hist_len)

    def reset(self, key) -> LatencyState:
        c = self.core
        return LatencyState(
            core=c.reset(key),
            prefix=jnp.full((self.delta,), c.ref_action, jnp.int32),
            tick=jnp.array(0, jnp.int32),
            hist_obs=jnp.zeros((self.hist_len, c.obs_dim), jnp.float32),
            hist_act=jnp.full((self.hist_len,), c.ref_action, jnp.int32),
            hist_valid=jnp.zeros((self.hist_len,), jnp.bool_),
            ep_return=jnp.array(0.0, jnp.float32),
        )

    def observe(self, s: LatencyState):
        return self.core.obs(s.core)

    def goal(self, s: LatencyState):
        return self.core.goal(s.core)

    def _run_ticks(self, keys, core_state, tick, actions):
        c = self.core

        def body(carry, x):
            cs, t, alive = carry
            key, a = x
            obs_before = c.obs(cs)
            nxt, r, term = c.step(key, cs, a)
            t2 = t + 1
            end = alive & (term | (t2 >= c.max_ticks))
            carry = (tree_where(alive, nxt, cs), jnp.where(alive, t2, t), alive & ~end)
            return carry, (obs_before, a, jnp.where(alive, r, 0.0).astype(jnp.float32), alive, end, alive & term)

        return jax.lax.scan(body, (core_state, tick, jnp.array(True)), (keys, actions))

    def oracle_forecast(self, key, s: LatencyState):
        """Exact observation at t_c + delta (upper-bound arm only). `key` must be the key later passed to `cycle`."""
        if self.delta == 0:
            return self.observe(s)
        keys = jax.random.split(key, self.k + 1)[: self.delta]
        (cs, _, _), _ = self._run_ticks(keys, s.core, s.tick, s.prefix)
        return self.core.obs(cs)

    def _push_history(self, s, obs_b, act, alive):
        M = self.hist_len
        if M == 0:
            return s.hist_obs, s.hist_act, s.hist_valid
        n = alive.sum()  # alive is a prefix mask: the first n ticks were executed
        cat = lambda old, new: jnp.concatenate([old, new], axis=0)
        sl = lambda x: jax.lax.dynamic_slice_in_dim(x, n, M, axis=0)
        return sl(cat(s.hist_obs, obs_b)), sl(cat(s.hist_act, act)), sl(cat(s.hist_valid, alive))

    def cycle(self, key, s: LatencyState, chunk):
        """Execute one decision cycle. chunk: (k,) int plan P_c. Returns (next LatencyState, CycleOutput)."""
        k, d = self.k, self.delta
        keys = jax.random.split(key, k + 1)
        actions = jnp.concatenate([s.prefix, chunk[: k - d].astype(jnp.int32)])
        (cs, tick, _), (obs_b, act, rew, alive, end, term) = self._run_ticks(keys[:k], s.core, s.tick, actions)
        tick_obs = jnp.concatenate([obs_b, self.core.obs(cs)[None]], axis=0)
        done = end.any()
        ep_ret = s.ep_return + rew.sum()
        h_obs, h_act, h_val = self._push_history(s, obs_b, act, alive)
        cont = LatencyState(cs, chunk[k - d:].astype(jnp.int32), tick, h_obs, h_act, h_val, ep_ret)
        nxt = tree_where(done, self.reset(keys[k]), cont)
        out = CycleOutput(tick_obs, act, rew, alive, end, done, term.any(), jnp.where(done, ep_ret, 0.0))
        return nxt, out
