from functools import partial
from typing import Any, Callable, NamedTuple, Optional, Union

import jax
from jax import lax, pmap, random, vmap
import jax.numpy as jnp


class SyntheticGenerator(NamedTuple):
    """Holds the function used to generate synthethic datasets.


    Attributes:
        init_pca: A pure function that creates samples to generate a good PCA.
        simulate: A pure function that generates the coordinates of the objects.
        video_synthesize: A pure function that transforms the output of `simulate` to video clips.
    """

    # Args: [PRNGKey] -> PCA transformation Matrix (A).
    init_pca: Union[Callable[..., jnp.ndarray], None]

    # Args: [PRNGKey] -> Coordinates of the objects (W).
    simulate: Union[Callable[..., jnp.ndarray], Any]

    # Args: [PRNGKey, W] -> Sequence of frames (X).
    video_synthesize: Union[Callable[..., jnp.ndarray], Any]


def synthetic_dataset(
    key,
    generator: SyntheticGenerator,
    batch_size: int = 64,
    augmentations: Optional[list] = None,
    augmentation_rates: Optional[jnp.ndarray] = None,
    entire_clip: bool = False,
):
    @partial(pmap, axis_name="i")
    def generate_batch(rng):
        @partial(vmap)
        def generate_pair(pair_key):
            sim_rng, frames_rng, pix_rng = random.split(pair_key, num=3)

            W = generator.simulate(sim_rng)
            temporal_window = slice(len(W) // 2 - 1, len(W) // 2 + 2, 1)
            label = (
                W if entire_clip else W[temporal_window, ...].transpose((1, 0, 2, 3))
            )
            X = generator.video_synthesize(frames_rng, W)

            if augmentations:
                choose_rng, trans_rng = random.split(pix_rng, 2)
                idx = random.choice(
                    choose_rng, len(augmentations), shape=(), p=augmentation_rates
                )
                X = lax.switch(idx, augmentations, trans_rng, X)
            return X, label

        batch_rngs = random.split(rng, num=batch_size)
        return generate_pair(batch_rngs)

    num_devices = jax.local_device_count()

    while True:
        key, batch_gen_key = random.split(key, num=2)
        yield generate_batch(random.split(batch_gen_key, num_devices))
