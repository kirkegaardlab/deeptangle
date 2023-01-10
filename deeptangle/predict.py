from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from numba import njit
import numpy as np

from deeptangle.utils import NetState


class Predictions(NamedTuple):
    """Holds the predictions of the model.

    Attributes:
        w: K coordinates of the skeleton in euclidian space (x,y).
        s: Confidence score of each prediction.
        p: Latent point representation of the prediction.
    """

    # Shape: ([Batch], num_predictions, num_frames_predicted, K, 2)
    w: jnp.ndarray

    # Shape: ([Batch], num_predictions, 1)
    s: jnp.ndarray

    # Shape: ([Batch], num_predictions, Latent space dimension)
    p: jnp.ndarray


@partial(jax.jit, static_argnums=0)
def predict(forward_fn, netstate: NetState, inputs: jnp.ndarray) -> Predictions:
    """
    Predicts the objects in the inputs.

    Parameters:
        forward_fn: A function that takes in a NetState object and input tensor, and returns the model's prediction.
        netstate: An object representing the state of a neural network.
        inputs: A tensor of inputs to the neural network.

    Returns:
        A Predictions tuple with the results of the NN.
    """
    params, state, _ = netstate
    predictions, _ = forward_fn.apply(params, state, inputs, is_training=False)
    return predictions


@njit
def _suppress(x, p, i, cutoff, threshold):
    mask = np.full(len(x), True)
    visible = np.sum((x[i] - x) ** 2, -1) < cutoff**2
    mask[visible] = np.sum((p[i] - p[visible]) ** 2, -1) >= threshold
    mask[i] = True
    return mask


@njit
def non_max_suppression(
    predictions: Predictions, threshold: float, overlap_threshold: float, cutoff: int
):
    """
    Non-maximum supression function that uses the latent space encoding.

    Parameters:
        predictions: The output of the forward function.
        threshold: The score threshold to remove bad predictions, ranges must be [0, 1].
        overlap_threshold: Equivalent to exclusion radius. How close must predicitons be at the
                           latent space to be considered to be targetting the same label.
        cutoff: Physical cuttoff for the lantent space, in pixels.

    Returns:
        A mask with 1s on the remaining predictions and 0s on the suppressed ones.
    """
    x, s, p = predictions
    n = len(x)

    remaining_ind = np.arange(n)

    # Remove low score predictions
    valid = s > threshold
    x, s, p = x[valid], s[valid], p[valid]
    remaining_ind = remaining_ind[valid]
    xcm = x[:, x.shape[1] // 2, x.shape[2] // 2, :]

    # Sort the arrays by scores.
    sorted_idxs = np.flip(np.argsort(s))

    xcm, p = xcm[sorted_idxs], p[sorted_idxs]
    remaining_ind = remaining_ind[sorted_idxs]

    # From probability to square distance
    threshold_p = -np.log(overlap_threshold)

    for i in range(n):
        idx = _suppress(xcm, p, i, cutoff, threshold_p)
        xcm, p = xcm[idx], p[idx]
        remaining_ind = remaining_ind[idx]
        if i >= len(remaining_ind):
            break

    non_suppressed_mask = np.full(n, False)
    non_suppressed_mask[remaining_ind] = True
    return non_suppressed_mask


def clean_predictions(p, threshold, overlap_threshold, cutoff=48):
    chosen_id = non_max_suppression(p, threshold, overlap_threshold, cutoff=cutoff)
    filter_preds = jax.tree_util.tree_map(lambda x: x[chosen_id], p)
    return filter_preds


def detect(
    forward_fn,
    netstate: NetState,
    inputs: jnp.ndarray,
    threshold: float,
    overlap_threshold: float,
    cutoff: int = 48,
):
    """
    Detects and removes overlapping predictions in the output of a neural network.

    Parameters:
        forward_fn: A function that takes in a NetState object and input tensor, and returns the model's prediction.
        netstate: An object representing the state of a neural network.
        inputs: A tensor of inputs to the neural network.
        threshold: A float indicating the minimum probability value for a prediction to be considered.
        overlap_threshold: A float indicating the minimum overlap for two predictions to be considered overlapping.
        cutoff: An optional integer indicating the maximum number of predictions to consider (default is 48).

    Returns:
        A Predictions representing the final predictions after removing overlaps.
    """
    predictions = predict(forward_fn, netstate, inputs)
    predictions = jax.tree_util.tree_map(lambda x: x[0], predictions)
    predictions = jax.tree_util.tree_map(np.asarray, predictions)
    best_predictions_idx = non_max_suppression(predictions, threshold, overlap_threshold, cutoff)
    final_predictions = jax.tree_util.tree_map(lambda x: x[best_predictions_idx], predictions)
    return final_predictions
