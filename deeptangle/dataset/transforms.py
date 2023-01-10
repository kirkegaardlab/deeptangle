""" "
Image transformations that can get applied to the Input frames before feeding them into the
Network.
"""

import jax.random as jr
import jax.numpy as jnp


def normalize(image, mean=0.0, std=1.0):
    return (image - image.mean()) / (image.std() + 1e-8) * std + mean


def invert(image, maxval=1):
    return maxval - image


def add_channel(image):
    """Adds a tailing dimension to the image."""
    return image[..., jnp.newaxis]


def remove_channel(image):
    """Removes the last dimension of the image."""
    return image[..., 0]


def _apply_noise(key, image, noise, p=1.0):
    prob = jr.uniform(key, shape=image.shape)
    return jnp.where(prob <= p, image + noise, image)


def apply_white_noise(key, image, mu, std, p):
    noise_key, apply_key = jr.split(key, 2)
    noise = mu + std * jr.normal(noise_key, shape=image.shape)
    return _apply_noise(apply_key, image, noise, p)


def apply_random_white_noise(key, image, mu, p, maxstd=0.2):
    noise_key, std_key = jr.split(key, 2)
    std = jr.uniform(std_key, shape=(), minval=0, maxval=maxstd)
    return apply_white_noise(noise_key, image, mu, std, p)
