from skimage import color
import skvideo.io

import numpy as np
import jax.numpy as jnp


def read_clip_from_video(filepath, start_time, end_time, fps, size, origin):
    video = skvideo.io.vreader(filepath)

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    clip_frames = jnp.arange(start_frame, end_frame)

    x0, y0 = (int(i) for i in origin)
    mask = (clip_frames, slice(y0, y0 + size, 1), slice(x0, x0 + size, 1))

    # Skip the frames until the start frame.
    for _ in range(start_frame):
        next(video)

    frames = np.array(
        [color.rgb2gray(next(video)) for _ in range(end_frame - start_frame)]
    )
    frames = frames[mask]
    return jnp.asarray(frames)


def video_to_clips(video, num_frames):
    assert num_frames % 2 == 1, "num_frames must have a middle frame"

    N = len(video)
    slices = np.arange(num_frames)
    clips = np.stack([video[slices + i] for i in range(N - num_frames)])
    return clips
