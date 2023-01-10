"""
Wrappers to build the model from scratch or from a checkpoint.
"""
import haiku as hk
import jax
import jax.numpy as jnp

from deeptangle import checkpoints
from deeptangle.dataset.pca import points_from_pca
from deeptangle.logger import recover_experiment_parameters
from deeptangle.model import Detector
from deeptangle.predict import Predictions


def build_model(A: jnp.ndarray, num_suggestions: int, latent_dim: int, num_frames: int):
    """
    Builds a model with the function that translated the outputted eigenvalues to coordinates.

    Parameters:
        A: Transformation Matrix of the PCA.
        num_suggestions: Number of suggestion C per cell in the final layer.
        latent_dim: Dimension of the latent space.
        num_frames: Number of stacked frames in the Input clip.

    Returns:
        A Haiku transform tuple with an init and an apply function.
    """
    num_components, kpoints2 = A.shape
    J = jnp.flip(jnp.identity(kpoints2), axis=1)
    B = A @ J @ jnp.transpose(A)

    def _forward_fn(batch, is_training):
        predictor = Detector(num_components, num_suggestions, latent_dim, num_frames)

        # Transpose the temporal frames to be last channel on the input.
        batch = jnp.transpose(batch, axes=(0, 2, 3, 1))

        # Predict
        S, H, P = predictor(batch, is_training, B)

        # Transform confidence score to be in [0,1] range.
        S = jax.nn.sigmoid(S)

        # Convert eigenvalues to coordinates.
        CM, H = H[..., :2], H[..., 2:]
        W = points_from_pca(H, A, CM)
        return Predictions(w=W, s=S, p=P)

    forward_fn = hk.without_apply_rng(hk.transform_with_state(_forward_fn))
    return forward_fn


def load_model(origin_dir: str, broadcast: bool = False):
    """
    Builds a model using the weights and the transformation matrix found at the directory.

    Parameters:
        origin_dir: Path to the folder where the weights are.
        broadcast: Whether to broadcast the weights to the number of devices.

    Returns:
        The forward function and the state of the model.
    """
    experiment_parameters = recover_experiment_parameters(origin_dir)
    A = checkpoints.load_pca_matrix(origin_dir)
    forward = build_model(
        A=A,
        num_suggestions=experiment_parameters["n_suggestions"],
        latent_dim=experiment_parameters["latent_dim"],
        num_frames=experiment_parameters["nframes"],
    )
    state = checkpoints.restore(origin_dir, broadcast=broadcast)
    return forward, state
