from jax import lax, vmap
import jax.numpy as jnp
import jax.random as jr

from celegans import transforms


def convert_to_clip(w, size, R=0.8, eps=0.3, px_spine=0.9):
    """
    Converts the given worms coordinates w of shape:
    (timesteps, num_worms, coord_points, 2) to a collection of 2D frames (clip)
    of shape (size, size).

    Args:
        w: coordinates of the worms.
        size: frame height and width.
        R: radius of the worm.
        eps: Antialiasing smoothing distance(?).

    Returns:
        clip: The simulation of w into consecutive frames.
    """
    w = jnp.round(w).astype(int)
    time, nworms, K, ndim = w.shape

    i = jnp.arange(K)
    r = R * jnp.abs(jnp.sin(jnp.arccos((i - K / 2) / (K / 2 + 0.2))))
    RL = int(3 * R)
    ii, jj = jnp.meshgrid(jnp.arange(-RL, RL + 1), jnp.arange(-RL, RL + 1))

    @vmap
    def draw_circle(r):
        reps = r + eps
        px_value = r / R * px_spine
        d = jnp.sqrt(ii**2 + jj**2)
        return jnp.where(d <= reps, jnp.where(d < r, px_spine, (reps - d) / eps) * px_value, 0)

    circles = draw_circle(r)

    @vmap
    def draw_frame(wt):
        frame = jnp.zeros((size, size), dtype=jnp.float32)

        def place_circle(i, im):
            circle = circles[i % K]
            cx, cy = wt[i]
            current_px = im[cy + jj, cx + ii]
            larger_px = jnp.maximum(circle, current_px)
            negative = ((cy + jj) < 0) | ((cx + ii) < 0)
            new_px = jnp.where(negative, current_px, larger_px)
            im = im.at[cy + jj, cx + ii].set(new_px)
            return im

        frame = lax.fori_loop(0, len(wt), place_circle, init_val=frame)
        return frame

    w_flatten = w.reshape(time, nworms * K, ndim)
    clip = draw_frame(w_flatten)
    return clip


def video_synthesis(key, w, size):
    """
    Converts the simulated coordinates of the worms to (size,size) arrays of pixels.

    It applies some normalizations and adds noise to make them feel as real
    images.

    Args:
        key: PRNGKey to use for the random noise.
        w: Cordinates of the worms with shape (timesteps, nworms, kpoints, 2).
        size: Frame size (size, size).
        normalise: Whether to normalise the frames or not.
    Returns:
        clip: Frames of the clip with the worms drawn (timesteps, size, size).
    """
    bg_rng, fg_rng = jr.split(key, 2)
    bg_rngs = jr.split(bg_rng, 4)
    fg_rngs = jr.split(fg_rng, 2)

    # Create a background for the plate.
    background_value = jr.uniform(bg_rngs[0], shape=(), minval=0.01, maxval=0.2)
    object_threshold = jr.normal(bg_rngs[1], shape=()) * 2 + 8

    transformations = [
        lambda image: transforms.add_channel(image),
        lambda image: transforms.add_static_objects(
            bg_rngs[2], image, size, threshold=object_threshold
        ),
        lambda image: transforms.gaussian_blur(image, sigma=1.5, kernel_size=3),
        lambda image: transforms.apply_white_noise(fg_rngs[0], image, mu=0, std=0.01, p=1),
        lambda image: transforms.remove_channel(image),
    ]

    background = jnp.full(shape=(size, size), fill_value=background_value)
    clip = convert_to_clip(w, size, px_spine=0.7)
    clip = jnp.maximum(clip, background)
    for transform in transformations:
        clip = transform(clip)
    return clip
