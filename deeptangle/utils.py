from collections import namedtuple

import jax
import jax.numpy as jnp


NetState = namedtuple('NetState', ['params', 'state', 'opt_state'])
ExperimentID = namedtuple('ExperimentID', 'uid dir')

def mid_value(a, axis):
    """Return the middle value in that axis."""
    indices = jnp.array([a.shape[axis] // 2])
    return jnp.take(a, indices, axis=axis).squeeze(axis)


def mid_values(a, axis):
    for i, ax in enumerate(sorted(axis)):
        a = mid_value(a, ax - i)
    return a

def remove_leading_dim(a):
    # FIXME(albert): allow for multiple arrays as inputs.
    return a[0]

def broadcast_sharded(x, num_devices):
    """Broadcast x to the number of devices on its leading dimension."""

    def broadcast(v):
        new_shape = (num_devices, *v.shape)
        broadcasted_array = jax.numpy.broadcast_to(v, new_shape)
        return broadcasted_array

    return jax.tree_util.tree_map(broadcast, x)


def single_from_sharded(x):
    return jax.tree_util.tree_map(lambda v: v[0], x)
