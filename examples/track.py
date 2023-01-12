import os
import pickle
import subprocess

from absl import app, flags
import jax
import jax.numpy as jnp
from matplotlib import Path
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import equalize_adapthist
import skvideo.io

import deeptangle as dt


flags.DEFINE_string("input", default=None, required=True, help="Path to the video.")
flags.DEFINE_string("model", default=None, required=True, help="Path to the weights")
flags.DEFINE_string("output", default="out/", help="Location to store the frames.")
flags.DEFINE_float("correction_factor", default=1, help="Value of the correction_factor.")
flags.DEFINE_float("score_threshold", default=0.5, help="Score threshold to prune bad predictions.")
flags.DEFINE_float("overlap_threshold", default=0.5, help="Overlap score threshold to suppress predictions.")
flags.DEFINE_integer("initial_frame", default=5, help="First target frame to start tracking.")
flags.DEFINE_integer("num_frames", default=0, help="Number of frames to perform tracking.")
flags.DEFINE_integer("num_batches", default=10, help="Maximum number of batches to do simultaneously.")

FLAGS = flags.FLAGS


def main(args):
    del args

    with dt.time_activity("Loading Model"):
        forward_fn, state = dt.load_model(FLAGS.model)

    with dt.time_activity("Reading input video from video file"):
        frames_to_load = FLAGS.initial_frame + FLAGS.num_frames + 6 if FLAGS.num_frames > 0 else 0
        initial_frame = FLAGS.initial_frame - 5
        video = skvideo.io.vread(FLAGS.input, num_frames=frames_to_load, as_grey=True)[initial_frame:, ..., 0]

    with dt.time_activity("Pre-processing the video"):
        video = 255 - video
        video = equalize_adapthist(video)
        video = FLAGS.correction_factor * video

    with dt.time_activity("Converting video into clips"):
        clips = dt.video_to_clips(video, num_frames=11)

    with dt.time_activity("Predicting splines"):

        def predict_in_batches(x):
            trim_frames = int(len(x) - len(x) % FLAGS.num_batches)
            new_shape = (FLAGS.num_batches, -1, *x[0].shape)
            batched_X = jnp.reshape(x[:trim_frames], new_shape)
            scan_predict_fn = lambda _, u: (None, dt.predict(forward_fn, state, u))
            _, y = jax.lax.scan(scan_predict_fn, init=None, xs=batched_X)
            y = jax.tree_util.tree_map(lambda u: jnp.reshape(u, (-1, *u.shape[2:])), y)
            y = jax.tree_util.tree_map(np.array, y)
            predictions = list(map(dt.Predictions, *y))
            return predictions

        def non_max_suppression(p):
            p = jax.tree_util.tree_map(lambda x: np.array(x), p)
            non_suppressed_p = dt.non_max_suppression(p, threshold=0.5, overlap_threshold=0.5, cutoff=48)
            return jax.tree_util.tree_map(lambda x: x[non_suppressed_p], p)

        predictions_list = predict_in_batches(clips)
        predictions_list = [non_max_suppression(p) for p in predictions_list]

    with dt.time_activity("Tracking"):
        identities_list, splines_list = dt.identity_assignment(predictions_list)
        identities_list, splines_list = dt.merge_tracks(identities_list, splines_list, framesize=video.shape[1])

    with dt.time_activity(f"Plotting the results and saving them at {FLAGS.output}"):
        plt.style.use("fast")
        outdir_path = Path(FLAGS.output)
        outdir_path.mkdir(exist_ok=True, parents=True)
        for t, (identities, splines) in enumerate(zip(identities_list, splines_list)):
            fig = plt.figure(figsize=(10.42, 10.42))
            plt.ylim(0, video.shape[1])
            plt.xlim(0, video.shape[2])
            plt.imshow(video[5 + t], cmap="binary")
            for i, x in zip(identities, splines):
                color = f"C{i%5:02d}"
                plt.plot(x[5:-5, 0], x[5:-5, 1], "-", color=color)

            figname = outdir_path.joinpath(f"{t:04d}.png")
            fig.savefig(figname, pad_inches=0, bbox_inches="tight")
            plt.close(fig)

    with dt.time_activity("Converting frames to movie using ffmpeg"):
        cwd = os.path.dirname(os.path.realpath(__file__))
        cmd = f"ffmpeg -framerate 20 -y  -hide_banner -loglevel error -pattern_type glob -i '{str(outdir_path)}/*.png' -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -c:v libx264 -pix_fmt yuv420p '{str(outdir_path)}/../tracking.mp4' "
        subprocess.run(cmd, cwd=cwd, shell=True)



if __name__ == "__main__":
    app.run(main)
