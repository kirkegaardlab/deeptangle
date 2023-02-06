from collections import defaultdict
import contextlib
from datetime import datetime
import subprocess
import time

import jax
import jax.numpy as jnp
from matplotlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.exposure import equalize_adapthist
import skvideo.io

import deeptangle as dt
from gooey import Gooey, GooeyParser


@contextlib.contextmanager
def time_activity(activity_name: str, current: int, total: int):
    print(f"[Starting] {activity_name}")
    start = time.perf_counter()
    yield
    duration = time.perf_counter() - start
    print(f"[Timing] {activity_name} finished: {duration:.4f} s.")
    print("progress: {}/{}".format(current, total))
    print()

def save_splines_as_csv(fname, identities, splines):
    data = defaultdict(list)
    num_points = splines[0].shape[1]
    for t, (ii, ss) in enumerate(zip(identities, splines)):
        data['frame'].extend([t]*len(ii))
        data['identity'].extend(ii)
        for i in range(num_points):
            data[f'x{i}'].extend(ss[:,i,0])
            data[f'y{i}'].extend(ss[:,i,1])
    df = pd.DataFrame(data)
    df.to_csv(fname)


@Gooey(
    advanced=True,
    program_name="de(ep)tangle",
    default_size=(900, 700),
    progress_regex=r"^progress: (?P<current>\d+)/(?P<total>\d+)$",
    progress_expr="current / total * 100",
    hide_progress_msg=True,
)
def main():
    parser = GooeyParser(description="Many splines tracking")

    parser.add_argument("--input", default=None, required=True, help="Path to the video.", widget="FileChooser")
    parser.add_argument("--model", default="weights/", required=True, help="Path to the weights", widget="DirChooser")
    parser.add_argument("--output", default=str(datetime.today().isoformat().split('.')[0].replace(':','-')), help="Location to store the video.", widget="DirChooser")
    parser.add_argument("--correction_factor", type=float, default=1.2, help="Value of the correction_factor μ.", widget="DecimalField")
    parser.add_argument("--score_threshold", type=float, default=0.5, help="Score threshold to prune bad predictions.", widget="DecimalField")
    parser.add_argument("--overlap_threshold", type=float, default=0.5, help="Overlap score threshold to suppress predictions.", widget="DecimalField")
    parser.add_argument("--initial_frame", type=int, default=5, help="First target frame to start tracking.", widget="IntegerField")
    parser.add_argument("--num_frames", type=int, default=100, help="Number of frames to perform tracking. (0 to do whole video)", widget="IntegerField")
    parser.add_argument("--num_batches", type=int, default=10, help="Maximum number of batches to do simultaneously.", widget="IntegerField")

    args = parser.parse_args()

    with time_activity("Loading Model", 1, 7):
        forward_fn, state = dt.load_model(args.model)

    with time_activity("Reading input video from video file", 2, 8):
        frames_to_load = args.initial_frame + args.num_frames + 6 if args.num_frames > 0 else 0
        initial_frame = args.initial_frame - 5
        video = skvideo.io.vread(args.input, num_frames=frames_to_load, as_grey=True)[initial_frame:, ..., 0]

    with time_activity("Pre-processing the video", 3, 8):
        video = 255 - video
        video = equalize_adapthist(video)
        video = args.correction_factor * video

    with time_activity("Converting video into clips", 4, 8):
        clips = dt.video_to_clips(video, num_frames=11)

    with time_activity("Predicting splines", 5, 8):

        def predict_in_batches(x):
            trim_frames = int(len(x) - len(x) % args.num_batches)
            new_shape = (args.num_batches, -1, *x[0].shape)
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

    with time_activity("Tracking", 6, 8):
        identities_list, splines_list = dt.identity_assignment(predictions_list)
        identities_list, splines_list = dt.merge_tracks(identities_list, splines_list, framesize=video.shape[1])

    with time_activity(f"Saving results as csv at {args.output}/splines.csv", 7, 8):
        outdir_path = Path(args.output)
        outdir_path.mkdir(exist_ok=True, parents=True)
        save_splines_as_csv(outdir_path/'splines.csv', identities_list, splines_list)

    if len(args.output) > 0:
        with time_activity(f"Plotting the results and saving them at {args.output}", 7, 8):
            frames_dir = outdir_path / 'frames'
            frames_dir.mkdir(exist_ok=True, parents=True)
            plt.style.use("fast")
            for t, (identities, splines) in enumerate(zip(identities_list, splines_list)):
                fig = plt.figure(figsize=(10.42, 10.42))
                plt.ylim(0, video.shape[1])
                plt.xlim(0, video.shape[2])
                plt.imshow(video[5 + t], cmap="binary")
                for i, x in zip(identities, splines):
                    color = f"C{i%5:02d}"
                    plt.plot(x[5:-5, 0], x[5:-5, 1], "-", color=color)

                figname = frames_dir / f"{t:04d}.png"
                print(f"[IO] Saving frame {t:03d}/{len(identities_list)} at {figname}")
                fig.savefig(figname, pad_inches=0, bbox_inches="tight")
                plt.close(fig)

        with time_activity("Converting frames to movie using ffmpeg", 8, 8):
            cmd = f"ffmpeg -framerate 20 -y  -hide_banner -loglevel error -pattern_type glob -i '{str(frames_dir)}/*.png' -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -c:v libx264 -pix_fmt yuv420p '{str(outdir_path)}/tracking.mp4'"
            subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    main()
