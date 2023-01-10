from typing import Optional, Sequence

import haiku as hk
from haiku._src.nets.resnet import BlockGroup
import jax
import jax.numpy as jnp


class Detector(hk.Module):
    """
    Composed Neural Netork of CNN backbone and latent space encoder.
    """

    def __init__(
        self,
        npoints: int,
        n_suggestion_per_feature: int,
        latent_space_dim: int,
        nframes: int = 11,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.neigen = npoints
        self.temporal_window = 3
        self.npoints = self.temporal_window * (npoints + 2) + 1  # (pca values + CM) + score
        self.n_suggestion = n_suggestion_per_feature
        self.offset = None

        bn_config = {"decay_rate": 0.9, "eps": 1e-5, "create_scale": True, "create_offset": True}

        # Feature extraction
        init_channels = 64 + sum([nframes // 5 * 2**i for i in range(6)])
        self.backbone = ResNet(
            blocks_per_group=(2, 4, 4, 2), bottleneck=False, init_channels=init_channels
        )

        # Layers to predict positions
        self.fc_w1 = hk.Linear(512)
        self.bn_w1 = hk.BatchNorm(**bn_config)
        self.fc_w2 = hk.Linear(self.n_suggestion * self.npoints)

        # Layers to predict latent space (Dimensionality reduction)
        self.encoder = LatentSpaceEncoder(latent_space_dim, name="LS_encoder")

    def __call__(self, D, is_training, B):
        batch_size, height, width, _ = D.shape

        z = self.backbone(D, is_training)

        # Predicting positions (eigenvalues)
        w = jax.nn.relu(self.fc_w1(z))
        w = self.bn_w1(w, is_training)
        w = self.fc_w2(w)

        nrows = w.shape[1]
        ncols = w.shape[2]
        w = w.reshape(batch_size, nrows, ncols, self.n_suggestion, self.npoints)

        if self.offset is None:
            s_y = (height / nrows / 2) + jnp.arange(0, height, height / nrows)
            s_x = width / ncols / 2 + jnp.arange(0, width, width / ncols)
            self.offset = jnp.stack(jnp.meshgrid(s_x, s_y), axis=-1)

        # Separate the score prediction from the positional
        w, s = w[..., :-1], w[..., -1]

        # Split npoints to tw predictions (past-present-future)
        w = w.reshape(*w.shape[:-1], self.temporal_window, 2 + self.neigen)
        w = self.add_offset(w)

        # Flatten the prediction from grid to 1D
        w = w.reshape(batch_size, -1, self.temporal_window, 2 + self.neigen)
        s = s.reshape(batch_size, -1)

        # Predict latent space from the positions (only see the temporal self, not other predictions)
        w = self.align_eigenvalues(w, B)
        p = self.encoder(w, B, is_training=is_training)
        return s, w, p

    def add_offset(self, x):
        return x.at[..., (0, 1)].add(self.offset[None, ..., None, None, :])  # pyright: ignore

    def align_eigenvalues(self, x, B):
        w = x[..., 2:]
        wf = jnp.matmul(w, B)
        dist_keep = jnp.mean((w[..., 1:2, :] - w) ** 2, axis=-1, keepdims=True)
        dist_flip = jnp.mean((w[..., 1:2, :] - wf) ** 2, axis=-1, keepdims=True)
        w = jnp.where(dist_keep > dist_flip, wf, w)
        x = x.at[..., 2:].set(w)
        return x


class LatentSpaceEncoder(hk.Module):
    """
    Orientational invariant latent space encoder.
    """

    def __init__(self, latent_dim: int, name: Optional[str] = None):
        super().__init__(name)

        bn_config = {"decay_rate": 0.9, "eps": 1e-5, "create_scale": True, "create_offset": True}
        self.fc_p1 = hk.Linear(128)
        self.bn_p1 = hk.BatchNorm(**bn_config)
        self.fc_p2 = hk.Linear(latent_dim)

    def __call__(self, x, B, is_training):
        x = jax.lax.stop_gradient(x)
        xf = x.at[..., 2:].set(jnp.matmul(x[..., 2:], B))

        p = jax.nn.relu(self.fc_p1(x.reshape(*x.shape[:2], -1)))
        pf = jax.nn.relu(self.fc_p1(xf.reshape(*x.shape[:2], -1)))
        p = self.bn_p1(p + pf, is_training)
        p = self.fc_p2(p)
        return p


class ResNet(hk.Module):
    """
    ResNet Network with average pooling instead of max-pooling.

    Code based on:
    https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/nets/resnet.py
    """

    def __init__(
        self,
        blocks_per_group: Sequence[int],
        bottleneck: bool = True,
        channels_per_group: Sequence[int] = (64, 128, 256, 512),
        use_projection: Sequence[bool] = (True, True, True, True),
        name: Optional[str] = None,
        strides: Sequence[int] = (1, 2, 1, 2),
        init_channels: int = 64,
    ):
        super().__init__(name=name)

        bn_config = {"decay_rate": 0.9, "eps": 1e-5, "create_scale": True, "create_offset": True}

        self.initial_conv = hk.Conv2D(init_channels, 7, stride=2, with_bias=False)
        self.initial_batchnorm = hk.BatchNorm(name="initial_batchnorm", **bn_config)

        self.block_groups = []
        for i, stride in enumerate(strides):
            self.block_groups.append(
                BlockGroup(
                    channels=channels_per_group[i],
                    num_blocks=blocks_per_group[i],
                    stride=stride,
                    bn_config=bn_config,
                    resnet_v2=False,
                    bottleneck=bottleneck,
                    use_projection=use_projection[i],
                    name="block_group_%d" % (i),
                )
            )

    def __call__(self, inputs, is_training):
        out = self.initial_conv(inputs)
        out = self.initial_batchnorm(out, is_training)
        out = jax.nn.relu(out)
        out = hk.avg_pool(out, window_shape=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding="SAME")

        for block_group in self.block_groups:
            out = block_group(out, is_training, test_local_stats=False)
        return out
