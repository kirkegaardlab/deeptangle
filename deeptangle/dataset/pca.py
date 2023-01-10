from chex import Array
import jax.numpy as jnp

from sklearn.decomposition import PCA


def init_pca(key, synth_generator, n_components: int):
    X = synth_generator.init_pca(key)
    X = X.reshape(len(X), -1)
    pca = PCA(n_components=n_components)
    pca.fit(X)
    A = pca.components_
    A -= A.mean(1)[:, None]
    A = jnp.array(A)
    return A


def transform(X: Array, A: Array) -> Array:
    """
    Apply dimensionality reduction to X.

    X is projected on the first principal components previously extracted
    from a training set.
    """
    return jnp.dot(X, jnp.transpose(A))


def inverse_transform(X: Array, A: Array) -> Array:
    """
    Transform data back to its original space.
    """
    return jnp.dot(X, A)


def points_from_pca(X, A, X_cm):
    """
    Transforms the predictions of CM and PCA values to
    the physical positions of the skeletons points.
    """
    # Recover the coordinates from the PCA values.
    X_recovered = inverse_transform(X, A)

    # Reshape to 2D and add the CM coordinates.
    Xn = X_recovered.reshape(*X.shape[:-1], -1, 2)
    Xn = Xn + X_cm[..., jnp.newaxis, :]
    return Xn
