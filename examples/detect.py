from absl import app, flags
import deeptangle as dt
import matplotlib.pyplot as plt
from skimage.exposure import equalize_adapthist
import skvideo.io


flags.DEFINE_string("input", default=None, required=True, help="Path to the video.")
flags.DEFINE_string("output", default="out.png", help="File where the output is saved.")
flags.DEFINE_float("correction_factor", default=1, help="Value of the correction_factor.")
flags.DEFINE_float("score_threshold", default=0.5, help="Score threshold to prune bad predictions.")
flags.DEFINE_float("overlap_threshold", default=0.5, help="Overlap score threshold to suppress predictions.")
flags.DEFINE_integer("frame", default=5, help="Target frame to detect")
flags.DEFINE_string("model", default="ckpt", help="Path to the weights")
FLAGS = flags.FLAGS


def main(args):
    del args

    with dt.time_activity("Loading Model"):
        forward_fn, state = dt.load_model(FLAGS.model)

    with dt.time_activity("Reading input clip from video file"):
        frames_to_load = FLAGS.frame + 6
        initial_frame = FLAGS.frame - 5
        clip = skvideo.io.vread(FLAGS.input, num_frames=frames_to_load, as_grey=True)[initial_frame:, ..., 0]

    with dt.time_activity("Pre-processing the clip"):
        clip = 255 - clip
        clip = equalize_adapthist(clip)
        clip = FLAGS.correction_factor * clip
        clip = clip[None, ...]

    with dt.time_activity("Predicting splines"):
        predictions = dt.detect(
            forward_fn,
            state,
            clip,
            threshold=FLAGS.score_threshold,
            overlap_threshold=FLAGS.overlap_threshold,
        )

    plt.style.use("fast")
    with dt.time_activity("Plotting the results"):
        plt.figure(figsize=(10, 10))
        plt.xlim(0, clip.shape[3])
        plt.ylim(0, clip.shape[2])
        plt.imshow(clip[0, 5], cmap="binary")
        for x in predictions.w[:, 1]:
            plt.plot(x[5:-5, 0], x[5:-5, 1], "-")
    plt.figsave(fig, FLAGS.output, dpi=300)


if __name__ == "__main__":
    app.run(main)
