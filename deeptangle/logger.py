import contextlib
from datetime import datetime
import json
from pathlib import Path
import subprocess
import timeit
import uuid

from absl import logging

from deeptangle.utils import ExperimentID

logging.get_absl_handler().setFormatter(None)  # type: ignore


def start_logging(flags, model_dir: str):
    path = Path(model_dir)

    # Define some parameters to characterise the experiment
    info_flags = {
        'uid': str(uuid.uuid4())[:8],
        'hash': _get_git_revision_hash(),
        'date': datetime.today(),
    }

    # Get the parameters from train.py
    training_flags = [v for k, v in flags.flags_by_module_dict().items() if 'train' in k]
    if len(training_flags) > 0:
        training_flags = {v.name: v.value for v in training_flags[0]}
    experiment_flags = {**info_flags, **training_flags}

    if flags.save:
        experiment_dir = path.joinpath(f"{experiment_flags['uid']}")
        experiment_dir.mkdir(parents=True, exist_ok=True)
        filepath = experiment_dir.joinpath('experiment.json')

        flags.alsologtostderr = True
        logging.get_absl_handler().use_absl_log_file('logs', experiment_dir)   # type: ignore

        # Create a symlink to a log file in the parents folder to ease of finding.
        log_symlink = path.joinpath('logs.INFO')
        log_symlink.unlink(missing_ok=True)
        log_symlink.symlink_to(Path(f"{experiment_flags['uid']}").joinpath('logs.INFO'))

        # Save the experiment parameters into a json file.
        with open(filepath, 'w') as f:
            json.dump(experiment_flags, f, indent=4, default=str)
        logging.info('Experiment parameters stored at %s.', filepath)
        logging.info('Experiment logs stored at %s/%s.INFO', experiment_dir, info_flags['uid'])
    else:
        experiment_dir = None
        logging.info('\n'.join(f'- {k}: {v}' for k, v in experiment_flags.items()))
    experiment_id = ExperimentID(info_flags['uid'], experiment_dir)
    return experiment_id


@contextlib.contextmanager
def time_activity(activity_name: str):
    start = timeit.default_timer()
    yield
    duration = timeit.default_timer() - start
    logging.info('[Timing] %s finished: %.4f s.', activity_name, duration)


def _get_git_revision_hash() -> str:
    cmd = ['git', 'rev-parse', 'HEAD']
    return subprocess.check_output(cmd).decode('ascii').strip()


def store_logs(flags, stderr=True):
    flags.alsologtostderr = stderr
    logging.get_absl_handler().use_absl_log_file()  # type: ignore
    logging.info('Logs stored in %s.', flags.log_dir)


def log_step(step, uid, losses, save=False):
    time = datetime.now().strftime('%d/%m/%Y|%H:%M:%S')
    header = f'[{uid}|{time}][{step+1}] '
    loss_text = ' | '.join(f'{k}:{v:8.6g}' for k, v in losses.items())
    text = header + loss_text
    if save:
        text += ' (model saved)'
    logging.info(text)


def recover_experiment_parameters(experiment_dir: str) -> dict:
    experiment_path = Path(experiment_dir)
    filepath = experiment_path.joinpath('experiment.json')
    with open(filepath, 'r') as f:
        exp_data = json.load(f)
    return exp_data
