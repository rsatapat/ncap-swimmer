"""Recompute a correct DistanceFromTarget curve for runs logged before the
bearing/distance mixup in train_swimmer_es.py's rollout() was fixed.

Background: that bug computed np.linalg.norm([bearing, distance]) instead of
just distance, so progress.csv's DistanceFromTarget is sqrt(bearing^2 +
distance^2) for every run logged before the fix -- contaminated by the final
heading error, not a clean distance measurement. The raw bearing/distance
pair was never saved separately, so the old progress.csv rows can't be
corrected after the fact.

What this script does instead: for each saved itr_N.pkl checkpoint in a run,
re-roll the actual saved policy weights through fresh deterministic episodes
(same code path as play.py) and take the correctly-separated final distance.
This gives an accurate replacement curve, but only at checkpoint epochs (not
every epoch progress.csv has), and it won't exactly reproduce the original
training-time episodes (that RNG stream is gone -- consumed by the whole ES
population every epoch, not reproducible from outside). The policy itself is
deterministic (no sampling), so the only difference from the original
training-time eval is which heading the fresh episodes happen to draw.

Usage (run where dm_control is importable, e.g. WSL):
    python scripts/fix_distance_metric.py <run_dir> [<run_dir> ...] \
        --target=-2,1 --n_evals=8

Writes distance_fix.csv (columns: itr, distance) into each run_dir.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'garage', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from play import load_checkpoint, rollout_episode  # noqa: E402


def recompute_distance_curve(run_dir, target, n_evals=None, seed_base=1, max_steps=1000):
    """Re-roll every itr_N.pkl checkpoint in run_dir and return the corrected
    median final distance at each one.

    Args:
        run_dir: training run folder containing itr_<N>.pkl checkpoints and
            variant.json.
        target: (x, y) the fixed target this run was actually trained on
            (there's no --target_pos CLI flag, so this can't be read back
            from variant.json -- the caller has to know it).
        n_evals: number of fresh episodes per checkpoint, matching the run's
            own --n_evals so the median is computed over the same number of
            samples. Defaults to variant.json's own n_evals.
        seed_base: first seed used; seeds seed_base..seed_base+n_evals-1 are
            used for every checkpoint, so differences across checkpoints
            reflect the policy, not the heading draw.
        max_steps: episode length cap.

    Returns:
        list of (itr, corrected_distance) tuples, sorted by itr.
    """
    with open(os.path.join(run_dir, 'variant.json')) as f:
        variant = json.load(f)
    if n_evals is None:
        n_evals = variant.get('n_evals', 8)
    if 'initial_heading_mode' in variant:
        initial_heading_mode = variant['initial_heading_mode']
    else:
        initial_heading_mode = 'uniform' if variant.get('no_bias_initial_heading', False) else 'toward_target'

    itr_files = [f for f in os.listdir(run_dir) if re.match(r'itr_\d+\.pkl$', f)]
    itrs = sorted(int(re.match(r'itr_(\d+)\.pkl$', f).group(1)) for f in itr_files)

    results = []
    for itr in itrs:
        policy, task = load_checkpoint(os.path.join(run_dir, f'itr_{itr}.pkl'))
        if task != 'navigate':
            continue
        distances = []
        for seed in range(seed_base, seed_base + n_evals):
            result = rollout_episode(policy, task, seed=seed, target=target, max_steps=max_steps,
                                      initial_heading_mode=initial_heading_mode)
            # obs_distances is the policy's own to_target observation (bearing,
            # distance separated) across the episode -- last entry is the
            # final-step distance, with no bearing contamination.
            distances.append(result['obs_distances'][-1])
        corrected = float(np.median(distances))
        results.append((itr, corrected))
        print(f"{os.path.basename(run_dir)} itr={itr}: corrected distance={corrected:.4f}")

    return results


def write_distance_fix(run_dir, results):
    out_path = os.path.join(run_dir, 'distance_fix.csv')
    with open(out_path, 'w') as f:
        f.write('itr,distance\n')
        for itr, distance in results:
            f.write(f'{itr},{distance}\n')
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dirs', nargs='+', help='One or more training run folders to fix.')
    parser.add_argument('--target', default='-2,1', help="Target this run was trained on, as 'x,y'.")
    parser.add_argument('--n_evals', type=int, default=None,
                         help="Episodes per checkpoint. Defaults to each run's own variant.json n_evals.")
    parser.add_argument('--seed_base', type=int, default=1, help='First seed; n_evals consecutive seeds are used.')
    parser.add_argument('--n_steps', type=int, default=1000, help='Max steps per episode.')
    args = parser.parse_args()

    target = tuple(float(v) for v in args.target.split(','))
    for run_dir in args.run_dirs:
        results = recompute_distance_curve(
            run_dir, target, n_evals=args.n_evals, seed_base=args.seed_base, max_steps=args.n_steps)
        write_distance_fix(run_dir, results)
