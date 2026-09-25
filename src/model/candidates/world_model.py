"""Idea 6 (docs/DPOD.ipynb cells 21-24): `FormulaPresets` + `HamiltonianOps` +
`WorldModel`. Source: predictive-world-model-overview.md (Approaches C, D, E).

NOT WIRED IN. This module is intentionally standalone and is not connected to
InputContextN / DecisionVectorD or to action selection anywhere in this
codebase. Per the critical-review decision recorded in docs/core/STATE.yaml
(task TASK-20260925-001), it is implemented and preserved because it is
structurally sound Hamiltonian-NN code on its own terms, but wiring it into
the Craftax training/action-selection pipeline is deliberately deferred:

1. It requires canonical coordinates (q, p). None of the existing A/S/H/T
   channel encoders (src/model/channel_encoder.py) produce them, and how such
   an encoder would be trained is unspecified.
2. Oracle-leakage risk: `FormulaPresets` must encode independent, textbook
   physics. If its presets were instead copied from the Craftax simulator
   that also generates the evaluation data, the "model" would just reproduce
   the environment -- this module's presets below are generic Newtonian
   mechanics (gravity, drag, buoyancy, ground contact), not derived from or
   fitted to Craftax internals, but that independence has to be maintained by
   whoever eventually wires this in.
3. Event jumps (mode-to-mode transitions that change state dimension) are not
   implemented.
4. Nothing here connects to `MDPSolver` / the candidate branches in this
   package; that link is unspecified in the source material.
5. The mode mask table (which presets each mode uses) is hand-specified, not
   learned or derived from data.

Params here are plain dicts (matching the source notebook), not the
NamedTuple PyTree convention used by src/model/*.py -- an explicit exception,
justified by this module's deliberate isolation from the rest of the
trainable pipeline.
"""

import functools
from typing import Callable, Tuple

import jax
import jax.numpy as jnp


class FormulaPresets:
    """Fixed, known physics-equation library (Approach C). Not learned.

    Two kinds, kept separate to avoid double counting:
      * POTENTIALS V(q): conservative, go INSIDE the Hamiltonian
      * FORCES F(x): non-conservative (drag, buoyancy, ground contact), added
        to dp/dt outside the Hamiltonian
    A mode selects which presets apply through a boolean mask row. The last
    axis of q is 'up'. State x = (q, p), p = m v.
    """
    ETA_NAMES = ('m', 'g', 'rho', 'CdA', 'V', 'k', 'c')
    ETA = {name: i for i, name in enumerate(ETA_NAMES)}
    ETA_DIM = len(ETA_NAMES)

    @staticmethod
    def gravity_potential(q, eta):
        return eta[FormulaPresets.ETA['m']] * eta[FormulaPresets.ETA['g']] * q[-1]

    POTENTIALS: Tuple[Callable, ...] = (gravity_potential.__func__,)

    @staticmethod
    def drag_force(x, eta, wind):
        n = x.shape[0] // 2
        m, rho, cda = (eta[FormulaPresets.ETA[k]] for k in ('m', 'rho', 'CdA'))
        rel = x[n:] / m - wind
        return -0.5 * rho * cda * jnp.linalg.norm(rel) * rel

    @staticmethod
    def buoyancy_force(x, eta, wind):
        n = x.shape[0] // 2
        rho, g, vol = (eta[FormulaPresets.ETA[k]] for k in ('rho', 'g', 'V'))
        return jnp.zeros(n).at[-1].set(rho * g * vol)

    @staticmethod
    def ground_force(x, eta, wind):
        """Spring-damper below z=0 (penalty form; hard contact / restitution
        belongs to event switching, not here)."""
        n = x.shape[0] // 2
        m, k, c = (eta[FormulaPresets.ETA[i]] for i in ('m', 'k', 'c'))
        z, vz = x[n - 1], x[-1] / m
        f = jnp.where(z < 0.0, -k * z - c * vz, 0.0)
        return jnp.zeros(n).at[-1].set(f)

    FORCES: Tuple[Callable, ...] = (drag_force.__func__, buoyancy_force.__func__, ground_force.__func__)
    N_POT = len(POTENTIALS)
    N_FORCE = len(FORCES)

    @staticmethod
    def total_potential(q, eta, mask):
        vals = jnp.stack([f(q, eta) for f in FormulaPresets.POTENTIALS])
        return jnp.sum(jnp.where(mask, vals, 0.0))

    @staticmethod
    def total_force(x, eta, wind, mask):
        vals = jnp.stack([f(x, eta, wind) for f in FormulaPresets.FORCES])  # (F, n)
        return jnp.sum(jnp.where(mask[:, None], vals, 0.0), axis=0)


class HamiltonianOps:
    """Hamiltonian Neural Network pieces (Approach D) and the dissipative
    port-Hamiltonian extension (Approach E).

    The network outputs ONE scalar H(q, p); the vector field comes from
    autodiff, so structure (antisymmetric J, PSD R) is built in, not learned.
        H(q,p)  = |p|^2 / 2m + sum_active potentials(q) + MLP(q)
        x_dot   = (J - R) grad H,   J = [[0, I], [-I, 0]],   R = L L^T
    Limits: needs canonical (q, p); 'conserves something' != 'physically
    correct' (the MLP term may conserve a wrong quantity); fixed state dim.
    """

    @staticmethod
    def init_parameters(key, n, hidden=64):
        k1, k2, k3 = jax.random.split(key, 3)
        return dict(
            w1=jax.random.normal(k1, (n, hidden)) * 0.1, b1=jnp.zeros((hidden,)),
            w2=jax.random.normal(k2, (hidden, hidden)) * 0.1, b2=jnp.zeros((hidden,)),
            w3=jnp.zeros((hidden, 1)), b3=jnp.zeros((1,)),  # zero init -> V_nn starts at 0
        )

    @staticmethod
    def mlp_scalar(params, q):
        h = jnp.tanh(q @ params['w1'] + params['b1'])
        h = jnp.tanh(h @ params['w2'] + params['b2'])
        return (h @ params['w3'] + params['b3'])[0]

    @staticmethod
    def hamiltonian(params, x, eta, potential_mask):
        n = x.shape[0] // 2
        q, p = x[:n], x[n:]
        m = eta[FormulaPresets.ETA['m']]
        kinetic = jnp.sum(p ** 2) / (2.0 * m)
        known = FormulaPresets.total_potential(q, eta, potential_mask)
        return kinetic + known + HamiltonianOps.mlp_scalar(params, q)

    @staticmethod
    def symplectic_matrix(n):
        eye, zero = jnp.eye(n), jnp.zeros((n, n))
        return jnp.block([[zero, eye], [-eye, zero]])

    @staticmethod
    def dissipation_matrix(L_raw):
        """R = L L^T with L lower-triangular: PSD by construction."""
        L = jnp.tril(L_raw)
        return L @ L.T

    @staticmethod
    def port_field(params, L_raw, x, eta, potential_mask):
        """(J - R) grad_x H. With R=0 this is the plain HNN field."""
        n = x.shape[0] // 2
        grad_h = jax.grad(HamiltonianOps.hamiltonian, argnums=1)(params, x, eta, potential_mask)
        J = HamiltonianOps.symplectic_matrix(n)
        R = HamiltonianOps.dissipation_matrix(L_raw)
        return (J - R) @ grad_h

    @staticmethod
    def leapfrog(params, x, eta, potential_mask, dt):
        """Symplectic (Stormer-Verlet) step for the conservative separable
        H = T(p) + V(q). Do NOT swap for RK4 -- energy drifts."""
        n = x.shape[0] // 2
        dH = jax.grad(HamiltonianOps.hamiltonian, argnums=1)
        q, p = x[:n], x[n:]
        p_half = p - 0.5 * dt * dH(params, jnp.concatenate([q, p]), eta, potential_mask)[:n]
        q_new = q + dt * dH(params, jnp.concatenate([q, p_half]), eta, potential_mask)[n:]
        p_new = p_half - 0.5 * dt * dH(params, jnp.concatenate([q_new, p_half]), eta, potential_mask)[:n]
        return jnp.concatenate([q_new, p_new])

    @staticmethod
    def energy_drift(params, traj, eta, potential_mask):
        """H(x_t) - H(x_0) along a trajectory (T, 2n): a diagnostic for the
        conservation claim, valid only when R=0 and u=0."""
        H = jax.vmap(lambda x: HamiltonianOps.hamiltonian(params, x, eta, potential_mask))(traj)
        return H - H[0]


class WorldModel:
    """Idea 6 (Approach E): mode gate W_omega + per-mode port-Hamiltonian +
    preset forces + near-zero residual, integrated semi-implicitly.

        m_t   = gate(phi(x_t, e_t))                    (hard, straight-through)
        x_dot = (J - R_m) grad H_m(x) + G u + F_m(x, eta, e) + r_theta(x, e, m)
        x_t+1 = semi_implicit_euler(x_t, x_dot, dt)

    All K modes are evaluated every step (static shapes) and combined by the
    one-hot mode vector, so the whole thing is jit/vmap/scan-safe. Event
    detection / fragmentation jumps (Delta_{m->m'}) are NOT implemented here:
    they change state dimension and need a design decision first.
    """

    @staticmethod
    def init_parameters(key, n, n_u, n_wind, num_modes, hidden=64):
        ks = jax.random.split(key, 6)
        d = 2 * n
        keys_k = jax.random.split(ks[0], num_modes)
        hnn = jax.vmap(lambda k: HamiltonianOps.init_parameters(k, n, hidden))(keys_k)
        return dict(
            gate_w=jax.random.normal(ks[1], (d + n_wind, num_modes)) * 0.1,
            gate_b=jnp.zeros((num_modes,)),
            hnn=hnn,
            R_L=jax.random.normal(ks[2], (num_modes, d, d)) * 0.01,
            G=jax.random.normal(ks[3], (num_modes, d, n_u)) * 0.01,
            res_w1=jax.random.normal(ks[4], (d + n_wind + num_modes, hidden)) * 0.1,
            res_b1=jnp.zeros((hidden,)),
            res_w2=jnp.zeros((hidden, d)),  # zero init -> residual starts at 0
            res_b2=jnp.zeros((d,)),
        )

    @staticmethod
    def mode_vector(params, x, wind, tau=1.0, gumbel_key=None):
        """Hard one-hot mode with soft gradient (straight-through). Gumbel
        noise only when a key is given (training); plain argmax at inference."""
        logits = jnp.concatenate([x, wind]) @ params['gate_w'] + params['gate_b']
        if gumbel_key is not None:
            logits = logits + jax.random.gumbel(gumbel_key, logits.shape)
        soft = jax.nn.softmax(logits / tau)
        hard = jax.nn.one_hot(jnp.argmax(soft), soft.shape[-1])
        return hard + soft - jax.lax.stop_gradient(soft), soft

    @staticmethod
    def _vector_field(params, x, u, wind, eta, mode_mask, m):
        n = x.shape[0] // 2
        mp = FormulaPresets

        def per_mode(hnn_k, L_k, G_k, mask_k):
            return HamiltonianOps.port_field(hnn_k, L_k, x, eta, mask_k[:mp.N_POT]) \
                + G_k @ u \
                + jnp.concatenate([jnp.zeros(n), mp.total_force(x, eta, wind, mask_k[mp.N_POT:])])

        fields = jax.vmap(per_mode)(params['hnn'], params['R_L'], params['G'], mode_mask)  # (K, 2n)
        dx = jnp.einsum('k,kd->d', m, fields)

        h = jnp.tanh(jnp.concatenate([x, wind, m]) @ params['res_w1'] + params['res_b1'])
        return dx + h @ params['res_w2'] + params['res_b2']

    @staticmethod
    def step(params, x, u, wind, eta, mode_mask, dt, tau=1.0, gumbel_key=None):
        """Semi-implicit Euler: p first (from the full field), then q from
        the field re-evaluated at the new p. Returns (x_next, soft_mode)."""
        n = x.shape[0] // 2
        m, soft = WorldModel.mode_vector(params, x, wind, tau, gumbel_key)
        dx = WorldModel._vector_field(params, x, u, wind, eta, mode_mask, m)
        p_new = x[n:] + dt * dx[n:]
        dq = WorldModel._vector_field(
            params, jnp.concatenate([x[:n], p_new]), u, wind, eta, mode_mask, m)[:n]
        return jnp.concatenate([x[:n] + dt * dq, p_new]), soft

    @staticmethod
    @functools.partial(jax.jit, static_argnames=('dt',))
    def rollout(params, x0, u_seq, wind_seq, eta, mode_mask, dt):
        """x0 (2n,), u_seq (T, n_u), wind_seq (T, w) -> traj (T, 2n),
        soft_modes (T, K). Open-loop; re-assimilate real observations and
        replan (MPC-style) rather than trusting long rollouts."""
        def body(x, inp):
            u, w = inp
            x_next, soft = WorldModel.step(params, x, u, w, eta, mode_mask, dt)
            return x_next, (x_next, soft)
        _, (traj, soft) = jax.lax.scan(body, x0, (u_seq, wind_seq))
        return traj, soft

    @staticmethod
    @functools.partial(jax.jit, static_argnames=('dt', 'num_samples'))
    def monte_carlo(params, key, x0, u_seq, wind_seq, eta, eta_log_std, mode_mask, dt, num_samples):
        """N rollouts with lognormal-perturbed physical parameters eta
        (ensemble/Monte Carlo of Approach E). Returns (N, T, 2n); use it for
        P(event), mean and scatter. Spread here only reflects the assumed
        eta_log_std -- it is not calibrated uncertainty until validated."""
        etas = eta * jnp.exp(eta_log_std * jax.random.normal(key, (num_samples,) + eta.shape))
        return jax.vmap(
            lambda e: WorldModel.rollout(params, x0, u_seq, wind_seq, e, mode_mask, dt)[0]
        )(etas)
