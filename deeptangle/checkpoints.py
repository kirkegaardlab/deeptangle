"""
Manages experiments' checkpoints (saving to file and restoring from file).
"""
from pathlib import Path
import pickle

import jax
import jax.numpy as jnp

from deeptangle import utils


def save(checkpoint_dir: str, state, broadcast: bool = False) -> None:
    """
    Saves the given state into the checkpoint_dir. It splits the tree definition
    and the arrays into different files for efficiency reasons.
    """
    path = Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)

    if broadcast:
        state = utils.single_from_sharded(state)

    with path.joinpath('arrays.npy').open('wb') as f:
        for x in jax.tree_util.tree_leaves(state):
            jnp.save(f, x, allow_pickle=False)

    tree_struct = jax.tree_util.tree_map(lambda _: 0, state)
    with path.joinpath('tree.pkl').open('wb') as f:
        pickle.dump(tree_struct, f)


def restore(experiment_dir: str, broadcast: bool = False):
    path = Path(experiment_dir)

    if not path.exists():
        raise FileNotFoundError(f'{experiment_dir} does not exist.')

    with path.joinpath('tree.pkl').open('rb') as f:
        tree_struct = pickle.load(f)

    leaves, treedef = jax.tree_util.tree_flatten(tree_struct)
    with path.joinpath('arrays.npy').open('rb') as f:
        flat_state = [jnp.load(f) for _ in leaves]

    state = jax.tree_util.tree_unflatten(treedef, flat_state)

    if broadcast:
        num_devices = jax.local_device_count()
        state = utils.broadcast_sharded(state, num_devices)

    return state


def save_pca_matrix(experiment_dir: str, A: jnp.ndarray) -> None:
    path = Path(experiment_dir)
    path.mkdir(parents=True, exist_ok=True)

    with path.joinpath('eigenworms_transform.npy').open('wb') as f:
        jnp.save(f, A, allow_pickle=False)


def load_pca_matrix(experiment_dir: str) -> jnp.ndarray:
    path = Path(experiment_dir)

    if not path.exists():
        raise FileNotFoundError(f'{experiment_dir} does not exist.')

    with path.joinpath('eigenworms_transform.npy').open('rb') as f:
        A = jnp.load(f)
    return A
