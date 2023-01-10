# Example scripts to use with de(ep)tangle

Here are some example scripts that make use of this repository.

## Requirements
Before running the scripts, install `deeptangle` locally from the root directory
```install
pip install -e .
```
 as well as the `requirements.txt` file using pip.
```requirements
pip install -r requirements.txt
```

You also need to download the weights and the input videos.

## Detection
The main purpose of the model is to detect splines. Thus, `detect.py` provides a simple script to show how to do so.

An example usage would be:
```usage-train
python3 detect.py --model=weights/ --input=/path/to/video.avi
```

The flags are:
```flags
  --correction_factor: Value of the correction_factor.
    (default: '1.0')
    (a number)
  --frame: Target frame to detect
    (default: '5')
    (an integer)
  --input: Path to the video.
  --model: Path to the weights
    (default: 'ckpt')
  --output: File where the output is saved.
    (default: 'out.png')
  --overlap_threshold: Overlap score threshold to suppress predictions.
    (default: '0.5')
    (a number)
  --score_threshold: Score threshold to prune bad predictions.
    (default: '0.5')
    (a number)
```

## Tracking
Likewise, a `track.py` example is included, where the batching trick to increase performance is shown.

An example usage would be:
```usage-track
python3 track.py --model=weights/ --input=/path/to/video.avi
```

The flags are
```track-flags
  --correction_factor: Value of the correction_factor.
    (default: '1.0')
    (a number)
  --initial_frame: First target frame to start tracking.
    (default: '5')
    (an integer)
  --input: Path to the video.
  --model: Path to the weights
  --num_batches: Maximum number of batches to do simultaneously.
    (default: '10')
    (an integer)
  --num_frames: Number of frames to perform tracking.
    (default: '0')
    (an integer)
  --overlap_threshold: Overlap score threshold to suppress predictions.
    (default: '0.5')
    (a number)
  --score_threshold: Score threshold to prune bad predictions.
    (default: '0.5')
    (a number)
  --output: Location to store the frames.
    (default: 'out/')
```
