"""
Simulation of C. Elegans.
"""
from functools import partial
import random

import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid


def _theta(t, s, params):
    phase_1 = params["phase_1"]
    phase_2 = params["phase_2"]
    phase_3 = params["phase_3"]

    A = params["A"]
    T = params["T"]
    kw = params["kw"]
    ku = params["ku"]

    s = s ** (1 + 0.5 * (params["dr"] - 0.5))

    r = 0.5 + jnp.abs(jnp.sin(2 * jnp.pi * t)) * 0.5
    w = 2 * jnp.pi / T

    wave_motion = r * A * jnp.cos(w * t + kw * s + phase_1)
    u_motion = A * jnp.cos(w * t + phase_2) * jnp.cos(ku * s + phase_3)

    motion = u_motion + wave_motion
    return motion


@partial(jax.vmap, in_axes=(None, 0, None))
def theta(t, s, params):
    inc = params["inc"]
    x = jnp.cos(_theta(t, s, params) + inc)
    y = jnp.sin(_theta(t, s, params) + inc)
    return x, y


@partial(jax.vmap, in_axes=(None, 0, None))
def dtheta(t, s, params):
    inc = params["inc"]
    _dtheta = jax.jacfwd(_theta)
    dx = -jnp.sin(_theta(t, s, params) + inc) * _dtheta(t, s, params)
    dy = jnp.cos(_theta(t, s, params) + inc) * _dtheta(t, s, params)
    return dx, dy


def _qpos(t, params, kpoints):
    s = jnp.linspace(0, 1, kpoints)
    ds = 1 / kpoints
    dq = jnp.stack(theta(t, s, params), axis=-1)
    q = jnp.cumsum(dq, axis=0) * ds
    q = q - q.sum(0) * ds
    return q * params["L"]


qpos = jax.vmap(_qpos, in_axes=(0, None, None))


@partial(jax.vmap, in_axes=(0, None, None))
def qtan(t, params, kpoints):
    s = jnp.linspace(0, 1, kpoints)
    q = jnp.stack(theta(t, s, params), axis=-1)
    return q


@partial(jax.vmap, in_axes=(0, None, None))
def qvel(t, params, kpoints):
    s = jnp.linspace(0, 1, kpoints)
    ds = 1 / kpoints
    dq = jnp.stack(dtheta(t, s, params), axis=-1)
    q = jnp.cumsum(dq, axis=0) * ds
    q = q - q.sum(0) * ds
    return q * params["L"]


def worm_simulation(params, duration, snapshots, kpoints):
    """
    Simulates the whole trajectory/skeleton of a single worm.
    """
    time = jnp.linspace(0, duration, snapshots)
    X = qpos(time, params, kpoints)
    u = qvel(time, params, kpoints)
    t = qtan(time, params, kpoints)

    ds = params["L"] / kpoints
    V, Omega = solve(t, u, X, ds, params["alpha"])

    dt = duration / snapshots
    inc = jnp.cumsum(Omega * dt / 2, axis=0)
    V = rotate(V, inc)
    Uc = jnp.cumsum(V * dt, axis=0)[:, jnp.newaxis]

    X0 = jnp.array([params["x0"], params["y0"]])[jnp.newaxis, jnp.newaxis]
    X = rotate(X, inc)
    return X + X0 + Uc


@jax.vmap
def rotate(x, angle):
    """Rotates x by angle."""
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    rotation_matrix = jnp.array([[c, -s], [s, c]])
    return x @ rotation_matrix


@partial(jax.vmap, in_axes=(0, 0, 0, None, None))
def solve(t, u, X, ds, alpha):
    tx, ty = t[:, 0], t[:, 1]
    nx, ny = ty, -tx

    x, y = X[:, 0], X[:, 1]
    cx, cy = -y, x

    ux, uy = u[:, 0], u[:, 1]

    Ut = jnp.array([ux * tx + uy * ty, tx, ty, cx * tx + cy * ty])
    Un = jnp.array([ux * nx + uy * ny, nx, ny, cx * nx + cy * ny])

    fx = Ut * tx[jnp.newaxis] + alpha * Un * nx[jnp.newaxis]
    fy = Ut * ty[jnp.newaxis] + alpha * Un * ny[jnp.newaxis]

    Fx = trapezoid(fx, dx=ds)
    Fy = trapezoid(fy, dx=ds)
    Tau = trapezoid(x * fy - y * fx, dx=ds)

    b = -jnp.array([Fx[0], Fy[0], Tau[0]])
    A = jnp.array([Fx[1:], Fy[1:], Tau[1:]])

    V = jnp.linalg.solve(A, b)
    return V[:2], -V[2]


def sampling_params(key, nworms, box_size):
    @partial(jax.jit, static_argnums=(1, 2))
    def normal(key, loc=0.0, scale=1.0):
        key, sample_key = jax.random.split(key)
        samples = loc + jax.random.normal(sample_key, shape=(nworms,)) * scale
        return key, samples

    @partial(jax.jit, static_argnums=(1, 2))
    def uniform(key, low=0.0, high=1.0):
        key, sample_key = jax.random.split(key)
        samples = jax.random.uniform(sample_key, shape=(nworms,), minval=low, maxval=high)
        return key, samples

    params = {}
    key, params["L"] = uniform(key, low=30, high=45)

    # Motion related parameters
    key, params["A"] = normal(key, loc=1, scale=0.1)
    key, params["T"] = normal(key, loc=0.8, scale=0.1)
    key, params["kw"] = uniform(key, low=0, high=2 * jnp.pi)
    key, params["ku"] = normal(key, loc=jnp.pi, scale=1)

    # Non-motion related parameters
    key, params["inc"] = uniform(key, low=0, high=2 * jnp.pi)
    key, params["dr"] = uniform(key, low=0.2, high=0.8)
    key, params["phase_1"] = uniform(key, low=0, high=2 * jnp.pi)
    key, params["phase_2"] = uniform(key, low=0, high=2 * jnp.pi)
    key, params["phase_3"] = normal(key, loc=0, scale=0.1)
    key, params["alpha"] = normal(key, loc=4, scale=4)
    params["alpha"] = jnp.abs(params["alpha"] + 1.0)

    half_box = box_size // 2
    key, params["x0"] = uniform(key, low=-half_box, high=half_box)
    key, params["y0"] = uniform(key, low=-half_box, high=half_box)
    return params


def drop_param(params, dropout):
    """
    Use the first value of one of the parameters to be on all the worms.
    """
    if random.random() <= dropout:
        dropped_key = random.choice(list(params.keys()))
        params[dropped_key] = jnp.broadcast_to(
            params[dropped_key][0], shape=params[dropped_key].shape
        )
    return params


def simulate(
    key,
    nworms,
    duration,
    snapshots,
    box_size=128,
    kpoints=42,
    params=None,
    dropout: float = 0,
):
    """
    Simulates the movement of 'num_worms' across a simulation box of 'box_size'.
    """
    if params is None:
        params = sampling_params(key, nworms, box_size)

    if dropout > 0:
        params = drop_param(params, dropout)

    sim_fn = partial(
        worm_simulation,
        duration=duration,
        snapshots=snapshots,
        kpoints=kpoints,
    )
    X = jax.vmap(sim_fn, out_axes=1)(params)
    X = X + box_size // 2
    return X


def sim_pca(key, nworms, kpoints):
    """
    Simulates the undulations of nworms on different timesteps between t=0s
    and t=10s.
    """
    params_rng_key, time_rng_key = jax.random.split(key, 2)
    params = sampling_params(params_rng_key, nworms, 0)

    timesteps = jax.random.uniform(time_rng_key, shape=(nworms,), minval=0, maxval=10)
    qpos_parallel = jax.vmap(_qpos, in_axes=(0, 0, None))
    W = qpos_parallel(timesteps, params, kpoints)
    return W
