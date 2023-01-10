from collections import namedtuple
from functools import partial

from absl import app, flags
import dm_pix as pix
import jax
from jax import jit, pmap, vmap, grad, lax
from jax.tree_util import tree_map, tree_reduce
import jax.numpy as jnp
import jax.random as jr
import optax

from celegans import sim_pca, simulate, video_synthesis
from celegans.transforms import random_gamma
from deeptangle import checkpoints, logger, utils
from deeptangle import SyntheticGenerator, build_model, load_model, synthetic_dataset
from deeptangle.dataset import pca, transforms

TrainState = utils.NetState
Losses = namedtuple("Losses", ["w", "s", "p"])

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("batch_size", 40, "Size of the training batch.")
flags.DEFINE_float("learning_rate", 0.001, "Optimizer learning rate.")
flags.DEFINE_integer("train_steps", 100_000_000, "Number of training steps.")
flags.DEFINE_integer("eval_interval", 100, "Number of steps between evaluations.")
flags.DEFINE_integer("nframes", 11, "Number of frames in a clip.")
flags.DEFINE_integer("size", 256, "Size of the frame for training.")
flags.DEFINE_list("nworms", [5, 10, 50, 100, 150, 200, 250], "Number of worms.")
flags.DEFINE_float("clip_duration", 0.55, "seconds of simulation that the clip should last.")
flags.DEFINE_float("sim_dropout", 0.0, "Parameter dropout")
flags.DEFINE_integer("n_suggestions", 8, "Number of suggestions for cell in last layer.")
flags.DEFINE_integer("latent_dim", 8, "Dimension of the latent space.")
flags.DEFINE_integer("kpoints", 49, "Number of points in the skeleton simulation.")
flags.DEFINE_integer("npca", 12, "Number of components of the pca.")
flags.DEFINE_integer("nworms_pca", 100_000, "Number of worms used to find the pca.")
flags.DEFINE_integer("save_interval", -1, "Interval of epochs to save weights.")
flags.DEFINE_integer("warmup", 100, "Steps to wait until saving model.")
flags.DEFINE_bool("augmentation", True, "")
flags.DEFINE_float("sigma", 10, "Smoothness of the score")
flags.DEFINE_float("wloss_w", 1, "Weight on the coordinates loss.")
flags.DEFINE_float("wloss_s", 1e2, "Weight on the score/confidence loss.")
flags.DEFINE_float("wloss_p", 1e5, "Weight on the latent space loss.")
flags.DEFINE_string("load", None, "Path to where the weights are read from.")
flags.DEFINE_bool("save", False, "Path to where the weights are saved.")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Path to the dir containing checkpoints.")
flags.DEFINE_bool("merge", False, "Override init params.")
flags.DEFINE_integer("cutoff", 48, "")


def normalization(image):
    lower = jnp.percentile(image, 1)
    upper = jnp.percentile(image, 99)
    return (image - lower) / (upper - lower)


def make_datasets(key):
    # Define the augmentations to apply after the frames have been synthesized.
    augments = [
        lambda key, image: pix.random_brightness(key, image, max_delta=0.2),
        lambda key, image: pix.random_contrast(key, image, lower=0.2, upper=0.5),
        lambda key, image: random_gamma(
            key, image, lower=0.5, upper=2.5, loc=-0.2, scale=1.0
        ),
    ]

    # Increase the noise of the frames regardless of augmentation.
    def augmentation_wrapper(f):
        def wrapper(key, frame):
            key, noise_key = jax.random.split(key)
            frame = transforms.add_channel(frame)
            frame = transforms.apply_random_white_noise(noise_key, frame, mu=0, p=0.2, maxstd=0.1)
            frame = jnp.clip(frame, 0.0, None)
            frame = f(key, frame)
            frame = transforms.remove_channel(frame)
            frame = normalization(frame)
            return frame

        return wrapper

    augments = list(map(augmentation_wrapper, augments))
    augments_p = jnp.array([0.2, 0.2, 0.6])

    # Define a synthethic dataset for each amount of worms on the trainnig set.
    keys = jax.random.split(key, len(FLAGS.nworms))
    datasets = []
    for key, nworms in zip(keys, list(map(int, FLAGS.nworms))):
        pca_fn = lambda key: sim_pca(key, nworms=FLAGS.nworms_pca, kpoints=FLAGS.kpoints)
        sim_fn = lambda key: simulate(
            key, nworms, FLAGS.clip_duration, FLAGS.nframes, FLAGS.size, FLAGS.kpoints
        )
        video_fn = lambda key, w: video_synthesis(key, w, size=FLAGS.size)

        generator = SyntheticGenerator(pca_fn, sim_fn, video_fn)
        dataset = synthetic_dataset(key, generator, FLAGS.batch_size, augments, augments_p)

        # For compiling and to run the function inside and avoid new variables (generator gradients
        # overriden if the code is not executed)
        _ = next(dataset)

        datasets.append(dataset)
    return datasets, generator  # type: ignore


def _optimizer():
    return optax.adamw(learning_rate=FLAGS.learning_rate)


@partial(jit, static_argnames="forward")
def init_network(rng_key, forward) -> TrainState:
    """
    Initialises the weights of the network with dummy data to map the shapes
    of the hidden layers.
    """
    X_init = jnp.ones((1, FLAGS.nframes, FLAGS.size, FLAGS.size))
    params, state = forward.init(rng_key, X_init, is_training=True)
    opt_state = _optimizer().init(params)
    return TrainState(params, state, opt_state)


def _importance_weights(n: int) -> jnp.ndarray:
    weights = 1 / (jnp.abs(jnp.arange(-n // 2 + 1, n // 2 + 1)) + 1)
    return weights / weights.sum()


def multi_loss_fn(Y_pred, Y_label):
    X_pred, S_pred, P_pred = Y_pred

    inside = jnp.all((Y_label >= 0) & (Y_label < FLAGS.size), axis=(-1, -2, -3))

    @vmap
    def distace_matrix(a, b):
        A = a[None, ...]
        B = b[:, None, ...]
        return jnp.sum((A - B) ** 2, axis=-1)

    # Compute the distance matrix for direct and flip versions
    distance = distace_matrix(X_pred, Y_label).mean(-1)
    flip_distance = distace_matrix(X_pred, jnp.flip(Y_label, axis=-2)).mean(-1)
    distances = jnp.minimum(distance, flip_distance)

    # Reduce the distance to be weighted by the importance of each frame.
    num_frames = X_pred.shape[2]
    temporal_weights = _importance_weights(num_frames)
    distances = jnp.average(distances, axis=-1, weights=temporal_weights)

    # Compute the loss of the points only taking into consideration only those
    # predictions that are inside.
    inside_count = jnp.sum(inside) + 1e-6
    masked_distances = distances * inside[:, :, None]
    Loss_X = jnp.sum(jnp.min(masked_distances, axis=2)) / inside_count

    # Before computing the score and latent space losses,
    # we stop gradients for of the distances.
    distances = jax.lax.stop_gradient(distances)
    X = jax.lax.stop_gradient(X_pred)

    # Compute the confidence score of each prediction as S = exp(-d2/sigma)
    # and perform L2 loss.
    scores = jnp.exp(-jnp.min(distances, axis=1) / FLAGS.sigma)
    Loss_S = jnp.mean((scores - S_pred) ** 2)

    # Find out which target is closests to each prediction.
    # ASSUMPTION: That is the one they are predicting.
    T = jnp.argmin(distances, axis=1)

    # Compute which permutations are targeting the same index on a matrix.
    # T(i,j) = T(j, i) = 1 if i,j 'target' the same label, 0 otherwise
    same_T = T[:, None, :] == T[:, :, None]

    # Visibility mask for far predictions that not should not share latent
    # space.
    distance_ls = distace_matrix(P_pred, P_pred)
    K = X.shape[3]
    Xcm = X[:, :, num_frames // 2, K // 2, :] # [B N Wt K 2]
    visible = distace_matrix(Xcm, Xcm) < FLAGS.cutoff**2
    factor = visible / visible.sum(axis=2)[:, :, None]

    # Compute the cross entropy loss depending on whether they aim to predict
    # the same target. P(i targets k| j targets k) ~= e^(-d^2)
    # https://jax.readthedocs.io/en/latest/faq.html#gradients-contain-nan-where-using-where
    safe_log = lambda x: jnp.log(jnp.where(x > 0.0, x, 1.0))
    atraction = distance_ls # log(exp(d2))
    repulsion = -safe_log(1 - jnp.exp(-distance_ls))
    Loss_P = factor * jnp.where(same_T, atraction, repulsion)

    # Only take into account the predictions that are actually preddicting.
    # Bad prediction should not be close to actual predictions in the latent
    # space.
    scores_matrix = scores[:, :, None] * scores[:, None, :]
    Loss_P = jnp.sum(scores_matrix * Loss_P) / scores_matrix.sum()
    return Losses(Loss_X, Loss_S, Loss_P)


def loss_fn(forward, params, state, inputs, targets):
    preds, state = forward.apply(params, state, inputs, is_training=True)

    # Compute the losses and average over the batch.
    total_losses = multi_loss_fn(preds, targets)

    # Weight the losses with the given HP and
    # compute total loss as a sum of losses.
    weights = Losses(FLAGS.wloss_w, FLAGS.wloss_s, FLAGS.wloss_p)
    total_losses = tree_map(jnp.multiply, weights, total_losses)
    loss = tree_reduce(jnp.add, total_losses)
    return loss, (state, total_losses)


grad_fn = jit(grad(loss_fn, argnums=1, has_aux=True), static_argnames="forward")


@partial(pmap, axis_name="i", static_broadcasted_argnums=0, donate_argnums=1)
def train_step(forward, train_state, inputs, targets):
    # Unpack the train state and compute gradients w.r.t the parameters.
    params, state, opt_state = train_state
    grads, (state, losses) = grad_fn(forward, params, state, inputs, targets)

    # Use the mean of the gradient across replicas if the model is in
    # a distributed training.
    grads = lax.pmean(grads, axis_name="i")

    # Update the parameters by using the optimizer.
    updates, opt_state = _optimizer().update(grads, opt_state, params=params)
    params = optax.apply_updates(params, updates)

    # Loss should be the mean of all the losses too (only affects logging.)
    losses = lax.pmean(losses, axis_name="i")

    new_train_state = TrainState(params, state, opt_state)
    return losses, new_train_state


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    experiment_id = logger.start_logging(FLAGS, FLAGS.checkpoint_dir)

    # Generate 3 random number generators from the given seed.
    init_key, data_key, train_key = jr.split(jr.PRNGKey(FLAGS.seed), num=3)

    # Create the datasets with the combinations of worms from FLAGS.nworms.
    with logger.time_activity("Compiling synthethic datasets"):
        datasets, synth_generator = make_datasets(data_key)

    if FLAGS.load:
        A = checkpoints.load_pca_matrix(FLAGS.load)
        forward_fn, state = load_model(FLAGS.load, broadcast=True)
    else:
        net_key, pca_key = jr.split(init_key, 2)
        A = pca.init_pca(pca_key, synth_generator, n_components=FLAGS.npca)

        forward_fn = build_model(A, FLAGS.n_suggestions, FLAGS.latent_dim, FLAGS.nframes)
        state = init_network(net_key, forward_fn)
        state = utils.broadcast_sharded(state, jax.local_device_count())

    if FLAGS.save:
        checkpoints.save_pca_matrix(experiment_id.dir, A)

    # Initial high values for the loss that will be overriden -- hopefully.
    saved_loss = 1e9
    losses = [Losses(w=1e9, s=1e9, p=1e9)] * len(datasets)

    # Training loop
    for step in range(FLAGS.train_steps):
        train_key, dataset_key = jax.random.split(train_key, 2)
        idx = jax.random.choice(dataset_key, len(datasets))
        X, y = next(datasets[idx])

        train_loss, state = train_step(forward_fn, state, X, y)
        losses[idx] = train_loss

        if ((step + 1) % FLAGS.eval_interval) == 0:
            losses = tree_map(jnp.mean, jax.device_get(losses))
            loss = tree_reduce(jnp.add, losses) / len(losses)
            avg = tree_map(lambda *a: jnp.average(jnp.array(a)), *losses)
            logl = {"loss": loss, "w": avg.w, "s": avg.s, "p": avg.p}

            if save := (loss < saved_loss and step > FLAGS.warmup and FLAGS.save):
                checkpoints.save(experiment_id.dir, state, broadcast=True)
                saved_loss = loss
            logger.log_step(step, experiment_id.uid, logl, save)


if __name__ == "__main__":
    app.run(main)
