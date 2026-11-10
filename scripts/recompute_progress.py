"""
Recompute a corrected progress.csv for an old training run, using its saved
policy checkpoints (itr_N.pkl / params.pkl).

Why this exists: train_swimmer_es.py originally computed DistanceFromTarget and
MeanVelocity incorrectly (DistanceFromTarget pooled every timestep of the
episode instead of just the final one; MeanVelocity pooled timesteps across
n_evals episodes unweighted instead of averaging per-episode first). Both bugs
are fixed in train_swimmer_es.py now, but that fix can't retroactively repair
a progress.csv already written by a finished run -- this script re-evaluates
that run's SAVED weights with the corrected logic instead.

Important limitations, inherent to what training actually saves to disk:
  - Only epochs with a saved itr_N.pkl (every `snapshot_gap` epochs, per
    `snapshot_mode='gap_and_last'`) can be recomputed -- NOT every epoch like
    the original progress.csv. The output is necessarily sparser.
  - PopulationAvgReturn/Min/Max can't be recovered at all: those describe the
    256 perturbed candidates evaluated that epoch, and only the resulting
    center policy's weights (policy_dict) were ever saved, not the candidates
    themselves.
  - The reward/return value for each snapshot is a FRESH re-measurement (a new
    deterministic rollout of those exact weights), not the original historical
    CurrentReturn number -- the environment's random stream (initial joint
    angles, start heading) can't be rewound to reproduce the exact original
    run bit-for-bit, so expect a close but not identical value.

Must be run in an environment with garage/dm_control/torch/neurobo installed
(the same environment training itself runs in) -- not a plain analysis env.

Usage:
    python scripts/recompute_progress.py <run_dir> [--n_evals N] [--output PATH]

Example:
    python scripts/recompute_progress.py D:\\ncap-swimmer\\outputs\\2026-10-14_16-49-14_2
"""
import argparse
import glob
import io
import json
import os
import pickle
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'garage', 'src'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))


def _lazy_imports():
    """Deferred so --help works without the heavy training dependencies installed."""
    global np, pd, torch, suite, rollout, DMControlEnv
    import numpy as np
    import pandas as pd
    import torch
    from dm_control import suite
    from garage import rollout
    from garage.envs.dm_control import DMControlEnv
    import neurobo.ncap.envs  # noqa: F401  (registers swim/navigate tasks with the dm_control suite)


class CPU_Unpickler(pickle.Unpickler):
    """Same pattern as scripts/play.py: load CUDA tensors onto CPU."""

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)


def load_checkpoint(path):
    with open(path, 'rb') as f:
        return CPU_Unpickler(f).load()


def find_snapshots(run_dir):
    """Returns a sorted list of (epoch, path) for every itr_N.pkl in run_dir,
    plus params.pkl (treated as the final epoch, from variant.json's n_epochs)
    if that epoch isn't already covered by an itr_N.pkl file."""
    snapshots = []
    for path in glob.glob(os.path.join(run_dir, 'itr_*.pkl')):
        match = re.search(r'itr_(\d+)\.pkl$', os.path.basename(path))
        if match:
            snapshots.append((int(match.group(1)), path))
    snapshots.sort()

    variant_path = os.path.join(run_dir, 'variant.json')
    params_path = os.path.join(run_dir, 'params.pkl')
    if os.path.isfile(variant_path) and os.path.isfile(params_path):
        with open(variant_path) as f:
            n_epochs = json.load(f).get('n_epochs')
        if n_epochs and n_epochs not in dict(snapshots):
            snapshots.append((n_epochs, params_path))
    snapshots.sort()
    return snapshots


def evaluate_checkpoint(ckpt, task, seed, n_evals):
    """Rolls the checkpoint's policy out n_evals times and returns
    (mean_return, final_distance, mean_velocity), using the corrected
    aggregation: distance from the LAST timestep of each episode, velocity
    averaged within each episode first (matching the fixed Agent.rollout in
    train_swimmer_es.py)."""
    policy_cls = ckpt['policy_cls']
    policy_kwargs = ckpt['policy_kwargs']
    policy = policy_cls(**policy_kwargs)
    policy.set_param_values(ckpt['policy_dict'])

    env = DMControlEnv(suite.load('swimmer', task, task_kwargs={'random': seed}))

    total_reward = 0.
    distances = []
    velocities = []
    for _ in range(n_evals):
        data = rollout(env, policy, deterministic=True)
        total_reward += sum(data['rewards'])

        # Only the navigate task observes 'to_target'; swim has no target.
        if 'to_target' in data['env_infos']:
            final_to_target = data['env_infos']['to_target'][-1]
            distances.append(np.linalg.norm(final_to_target))

        velocities.append(np.mean(-data['env_infos']['body_velocities'][:, 1]))

    mean_return = total_reward / n_evals
    mean_distance = np.median(distances) if distances else np.nan
    mean_velocity = np.median(velocities)
    return mean_return, mean_distance, mean_velocity


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'run_dir', help='Training run folder containing itr_*.pkl / params.pkl / variant.json.')
    parser.add_argument(
        '--n_evals', type=int, default=None,
        help="Rollouts to average per checkpoint. Defaults to the run's own n_evals from variant.json.")
    parser.add_argument(
        '--seed', type=int, default=1,
        help="Seed for these new rollouts. Can't reproduce the exact original random stream either way.")
    parser.add_argument(
        '--output', default=None,
        help='Output CSV path. Defaults to <run_dir>/progress_recomputed.csv.')
    args = parser.parse_args()

    _lazy_imports()

    with open(os.path.join(args.run_dir, 'variant.json')) as f:
        variant = json.load(f)
    task = variant['task']
    n_evals = args.n_evals or variant.get('n_evals', 1)

    snapshots = find_snapshots(args.run_dir)
    if not snapshots:
        raise SystemExit(f'No itr_*.pkl or params.pkl snapshots found in {args.run_dir}')

    rows = []
    for epoch, path in snapshots:
        print(f'Evaluating epoch {epoch} ({os.path.basename(path)})...')
        ckpt = load_checkpoint(path)
        mean_return, mean_distance, mean_velocity = evaluate_checkpoint(
            ckpt, task=task, seed=args.seed, n_evals=n_evals)
        rows.append(dict(
            Epoch=epoch,
            Return=mean_return,
            DistanceFromTarget=mean_distance,
            MeanVelocity=mean_velocity,
        ))
        print(f'  Return={mean_return:.2f}  DistanceFromTarget={mean_distance:.3f}  '
              f'MeanVelocity={mean_velocity:.4f}')

    output_path = args.output or os.path.join(args.run_dir, 'progress_recomputed.csv')
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f'Wrote {len(rows)} rows to {output_path}')


if __name__ == '__main__':
    main()
