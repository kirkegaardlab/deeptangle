import jax
import jax.numpy as jnp
import dm_pix as pix


def add_channel(image):
    """Adds a tailing dimension to the image."""
    return image[..., jnp.newaxis]


def remove_channel(image):
    """Removes the last dimension of the image."""
    return image[..., 0]


def normal_intesity_gradient(key, image, size):
    """
    Applies a gradient intensity modifier to the pixel values.
    """
    ii, jj = jnp.meshgrid(jnp.arange(size), jnp.arange(size))
    i, j = jax.random.randint(key, shape=(2,), minval=0, maxval=size)
    N = jnp.exp(-((ii - i) ** 2 + (jj - j) ** 2) / (0.5 * (size / 2) ** 2)) + 1.0
    return image * N[jnp.newaxis, ..., jnp.newaxis]


def add_static_objects(key, image, size, threshold=1.0):
    noise = jax.random.normal(key, shape=(size, size, 1)) ** 3
    noise = pix.gaussian_blur(noise, 3, kernel_size=3)
    noise = 1.5 * (noise > threshold) * jax.nn.sigmoid(noise)
    noise = pix.gaussian_blur(noise, 3, kernel_size=3) * 0.5
    return jnp.maximum(image, noise)


def gaussian_blur(image, sigma, kernel_size=3):
    """Applies a gaussian blur to I."""
    return pix.gaussian_blur(image, sigma=sigma, kernel_size=kernel_size)


def add_background(image, value=0.1):
    """Adds a constant background to all the images."""
    return image + value


def normalise(image, mean=0.5, std=0.5):
    if jnp.any(std == 0):
        raise ValueError("Normalizing can not use std=0")

    image = image - image.mean()
    return std / image.std() * image + mean


def _apply_noise(key, image, noise, p=1.0):
    prob = jax.random.uniform(key, shape=image.shape)
    return jnp.where(prob <= p, image + noise, image)


def apply_white_noise(key, image, mu, std, p):
    noise_key, apply_key = jax.random.split(key, 2)
    noise = mu + std * jax.random.normal(noise_key, shape=image.shape)
    return _apply_noise(apply_key, image, noise, p)


def apply_random_white_noise(key, image, mu, p, maxstd=0.2):
    noise_key, std_key = jax.random.split(key, 2)
    std = jax.random.uniform(std_key, shape=(), minval=0, maxval=maxstd)
    return apply_white_noise(noise_key, image, mu, std, p)


def invert(image, maxval=1.0):
    return maxval - image


def random_gamma(
    key, image, lower: float, upper: float, loc: float = 0.0, scale: float = 1.0
):
    alpha = loc + scale * jax.random.normal(key, shape=())
    gamma = jnp.clip(2**alpha, lower, upper)
    std = image.std()
    image = (image / std) ** gamma
    image = std * (image / image.std())
    return image
