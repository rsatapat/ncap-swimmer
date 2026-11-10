"""Roll out a trained swimmer policy, either to save a video or to sweep
over seeds/targets and report how well it does.
"""
import argparse
import csv
import io
import json
import os
import sys
import pickle

import imageio
import numpy as np

# Add repo directories to Python's path, resolved relative to this file (repo/scripts/).
# Must happen before importing garage/neurobo so the local copies are found.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'garage', 'src'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))

from dm_control import suite
from garage.envs.dm_control import DMControlEnv
import neurobo.ncap.envs  # registers the 'swim'/'navigate' tasks with the suite
import torch

# Edit these to change what run_sweep() (and the default video target) use.
# TARGETS is only used for navigate checkpoints; swim checkpoints just run
# each seed once.
SEEDS = [1, 2, 3, 4, 5]
TARGETS = [(-3.0, 0.0), (3.0, 0.0), (0.0, 3.0), (0.0, -3.0)]


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)


def load_checkpoint(path):
    """Load a policy from a params.pkl snapshot.

    Also recovers the task ('swim' or 'navigate') it was trained on from
    variant.json next to the checkpoint, since params.pkl itself doesn't
    record which task made it, falling back to 'swim' if that file is missing.

    Returns:
        (policy, task)
    """
    with open(path, 'rb') as f:
        data = CPU_Unpickler(f).load()

    policy = data['policy_cls'](**data['policy_kwargs'])
    policy.set_param_values(data['best_policy_dict'])

    task = 'swim'
    variant_path = os.path.join(os.path.dirname(path), 'variant.json')
    if os.path.isfile(variant_path):
        with open(variant_path) as f:
            task = json.load(f).get('task', task)

    return policy, task


def rollout_episode(policy, task, seed, target=None, max_steps=300, render=False):
    """Run one episode with a fixed seed (and target, for navigate), and
    collect its trajectory, summary metrics, and (if requested) video frames.

    This is the single stepping loop used by both run_sweep() and
    make_video(), so a video frame and a sweep's trajectory point always
    come from the exact same rollout logic.

    Args:
        policy: loaded policy, e.g. from load_checkpoint().
        task: 'swim' or 'navigate'.
        seed: random seed for this episode (controls initial heading, etc).
        target: (x, y) target position (navigate only).
        max_steps: episode length cap.
        render: if True, also render and collect an RGB frame per step.

    Returns:
        dict with seed, target, total_reward, mean_reward, mean_velocity,
        final_distance (None for swim), trajectory (head xy per step, shape
        (T+1, 2)), velocities (per-step, shape (T,)), rewards (per-step,
        shape (T,)), and frames (list of RGB arrays, only if render=True).
    """
    task_kwargs = dict(random=seed)
    if task == 'navigate' and target is not None:
        task_kwargs['target_pos'] = target
    env = DMControlEnv(suite.load('swimmer', task, task_kwargs=task_kwargs))
    physics = env._env.physics

    obs, _ = env.reset()
    trajectory = [physics.named.data.geom_xpos['head'][:2].copy()]
    rewards = []
    velocities = []
    frames = [physics.render(height=480, width=640, camera_id=0)] if render else None

    for _ in range(max_steps):
        action, _ = policy.get_action(obs)
        step = env.step(action)
        obs = step.observation
        rewards.append(step.reward)
        velocities.append(-step.env_info['body_velocities'][1])
        trajectory.append(physics.named.data.geom_xpos['head'][:2].copy())
        if render:
            frames.append(physics.render(height=480, width=640, camera_id=0))
        if step.last:
            break

    trajectory = np.array(trajectory)
    # Resolve the actual target used, even when the caller left it to the
    # task's own default (target=None) -- read back from the task itself
    # rather than trusting the possibly-None argument.
    resolved_target = tuple(env._env.task._target_pos) if task == 'navigate' else None
    final_distance = (
        float(np.linalg.norm(np.array(resolved_target) - trajectory[-1]))
        if resolved_target is not None else None
    )

    return {
        'seed': seed,
        'target': resolved_target,
        'total_reward': float(np.sum(rewards)),
        'mean_reward': float(np.mean(rewards)),
        'mean_velocity': float(np.mean(velocities)),
        'final_distance': final_distance,
        'trajectory': trajectory,
        'velocities': np.array(velocities),
        'rewards': np.array(rewards),
        'frames': frames,
    }


def run_sweep(policy, task, seeds, targets, max_steps=300, out_csv=None):
    """Roll out the policy across every (seed, target) combination and
    report the results, without rendering any video.

    For swim checkpoints, targets don't apply and each seed is run once.

    Args:
        policy: loaded policy, e.g. from load_checkpoint().
        task: 'swim' or 'navigate'.
        seeds: list of random seeds to try.
        targets: list of (x, y) targets to try (navigate only).
        max_steps: episode length cap.
        out_csv: if given, write one long-format CSV here with every step of
            every run (seed, target, step, x, y, velocity, reward). Per-run
            summaries like final_distance can be derived from this later
            (e.g. the last step's x/y per (seed, target) group).

    Returns:
        list of dicts, one per run (see rollout_episode()).
    """
    targets_to_run = targets if task == 'navigate' else [None]
    results = []
    for seed in seeds:
        for target in targets_to_run:
            result = rollout_episode(policy, task, seed, target, max_steps)
            results.append(result)
            print(f"seed={seed} target={target}: "
                  f"final_distance={result['final_distance']} "
                  f"mean_reward={result['mean_reward']:.4f} "
                  f"mean_velocity={result['mean_velocity']:.4f}")

    if out_csv:
        with open(out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['seed', 'target', 'step', 'x', 'y', 'velocity', 'reward'])
            for r in results:
                # trajectory has one more point than velocities/rewards (the
                # initial reset position has no step taken from it yet).
                for step, (x, y) in enumerate(r['trajectory']):
                    velocity = r['velocities'][step - 1] if step > 0 else ''
                    reward = r['rewards'][step - 1] if step > 0 else ''
                    writer.writerow([r['seed'], r['target'], step, x, y, velocity, reward])
        print(f"Saved per-step results (location, speed, reward) to {out_csv}")

    return results


def run_checkpoints(run_dir, itrs, seed=1, max_steps=300, out_csv=None):
    """Roll out several itr_N.pkl checkpoints from the same run, on the same
    seed and (each checkpoint's own default) target, to see how the
    trajectory changes over training.

    Args:
        run_dir: training run folder containing itr_<N>.pkl checkpoints.
        itrs: list of checkpoint epoch numbers to roll out, e.g. [10, 50, 100, 200, 300].
        seed: random seed, shared across all checkpoints so differences in
            the trajectory reflect the policy, not a different initial heading.
        max_steps: episode length cap for each rollout.
        out_csv: if given, write one long-format CSV here with every step of
            every checkpoint's rollout (itr, target, step, x, y, velocity, reward).

    Returns:
        list of dicts, one per itr (see rollout_episode()).
    """
    results = []
    for itr in itrs:
        policy, task = load_checkpoint(os.path.join(run_dir, f'itr_{itr}.pkl'))
        result = rollout_episode(policy, task, seed=seed, target=None, max_steps=max_steps)
        results.append(result)
        print(f"itr={itr}: final_distance={result['final_distance']} "
              f"mean_reward={result['mean_reward']:.4f} mean_velocity={result['mean_velocity']:.4f}")

    if out_csv:
        with open(out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['itr', 'target', 'step', 'x', 'y', 'velocity', 'reward'])
            for itr, r in zip(itrs, results):
                for step, (x, y) in enumerate(r['trajectory']):
                    velocity = r['velocities'][step - 1] if step > 0 else ''
                    reward = r['rewards'][step - 1] if step > 0 else ''
                    writer.writerow([itr, r['target'], step, x, y, velocity, reward])
        print(f"Saved per-step checkpoint rollouts to {out_csv}")

    return results


def make_video(policy, task, seed, target=None, max_steps=300, out_path=None, fps=30):
    """Roll out one episode (via rollout_episode, with render=True) and save it as an mp4."""
    result = rollout_episode(policy, task, seed, target, max_steps, render=True)
    print(f"Writing video file to {out_path}...")
    imageio.mimsave(out_path, result['frames'], fps=fps)
    print("Finished successfully!")
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'model',
        nargs='?',
        default=os.path.join(REPO_ROOT, 'outputs', '2026-10-14_16-49-14_2', 'params.pkl'),
        help="Path to a params.pkl snapshot (--mode video/sweep), or a training "
             "run's folder containing itr_N.pkl checkpoints (--mode checkpoints).",
    )
    parser.add_argument(
        '--mode', choices=['video', 'sweep', 'checkpoints'], default='video',
        help="'video' renders one episode to an mp4. 'sweep' runs every "
             "combination of SEEDS/TARGETS (edit them at the top of this "
             "file) and reports the results, without rendering anything. "
             "'checkpoints' rolls out several itr_N.pkl checkpoints from one "
             "run (--itrs) on the same seed/target, to see how the "
             "trajectory changes over training -- writes a CSV meant to be "
             "read (not re-run) from track_model.ipynb's "
             "plot_trajectory_over_checkpoints, since that notebook may not "
             "have dm_control available.",
    )
    parser.add_argument('--n_steps', type=int, default=300, help='Max steps per episode.')
    parser.add_argument('--itrs', default='10,50,100,200,300',
                         help='Comma-separated itr_N checkpoint numbers to roll out (--mode checkpoints only).')
    args = parser.parse_args()

    if args.mode == 'checkpoints':
        itrs = [int(x) for x in args.itrs.split(',')]
        out_csv = os.path.join(args.model, 'checkpoint_trajectories.csv')
        run_checkpoints(args.model, itrs, max_steps=args.n_steps, out_csv=out_csv)
    else:
        print("Loading policy and weights...")
        policy, task = load_checkpoint(args.model)
        print(f"Policy loaded. Task inferred from variant.json: {task!r}")

        if args.mode == 'video':
            target = TARGETS[0] if task == 'navigate' else None
            video_path = os.path.join(
                os.path.dirname(args.model),
                f"swimmer_result_{os.path.basename(args.model).split('.')[0]}.mp4")
            make_video(policy, task, seed=SEEDS[0], target=target,
                       max_steps=args.n_steps, out_path=video_path)
        else:
            out_csv = os.path.join(os.path.dirname(args.model), 'sweep_results.csv')
            run_sweep(policy, task, SEEDS, TARGETS, max_steps=args.n_steps, out_csv=out_csv)
