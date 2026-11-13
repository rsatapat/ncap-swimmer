"""
Train swimmer body using ES algorithm (from ESTorch).
"""
import sys
import os

# Add repo directories to Python's path, resolved relative to this file (repo/scripts/experiments/).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'garage', 'src'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))

import click
import os
import sys
import time
import shutil
import json
import torch
import numpy as np
from mpi4py import MPI
from dowel import logger, tabular
from dm_control import suite
from acme import wrappers
import datetime

from garage import wrap_experiment, rollout
from garage.envs.dm_control import DMControlEnv #An adapter that wraps a dm_control environment so Garage can use it.
from garage.experiment import deterministic
from garage.experiment.snapshotter import Snapshotter

from garage.torch.policies import GaussianMLPPolicy
from neurobo.ncap.swimmer import SwimmerPolicy
import neurobo.ncap.envs # this basically runs the code in envs.py to register the tasks with the dm_control suite
import estorch


@click.command(context_settings={'show_default': True})
@click.option('--job', default='0', help='Job id on cluster.')
@click.option('--task', default='swim', type=click.Choice(['swim', 'navigate']), help='Task name.')
@click.option('--n_epochs', default=2, help='Num epochs total.')
@click.option('--n_proc', default=1, help='Num of processes to use for distributed training.')
@click.option('--population_size', default=8, help='Num perturbations per epoch. Must be even and multiple of number of processes.')
@click.option('--sigma', default=0.02, help='Standard deviation of Gaussian noise perturbation.')
@click.option('--optim', default='adam', type=click.Choice(['adam', 'sgd']), help='Name of optimizer.')
@click.option('--lr', default=0.01, help='Learning rate of optimizer.')
@click.option('--l2_coeff', default=0.005, help='L2 regularization coefficient.')
@click.option('--n_evals', default=1, help='Num evaluation episodes per epoch.')
@click.option('--policy', default='mlp', type=click.Choice(['mlp', 'ncap']), help='Policy network architecture.')
@click.option('--policy_hidden_sizes', default='64,64', help='Policy network hidden layers sizes.')
@click.option('--ncap_opts', default='111', help='Bit string for NCAP options [ws, sign, mag].')
@click.option('--include_speed_control', is_flag=True, default=False,
  help="Let task_module's speed_control output actually reach the CPG. Only matters when --policy=ncap and --task=navigate with use_task='learned'.")
@click.option('--include_turn_control', is_flag=True, default=False,
  help="Let task_module's right/left turn_control output actually reach the CPG. Only matters when --policy=ncap and --task=navigate with use_task='learned'.")
@click.option('--osc_period', default=60, help='Swimmer oscilator period.')
@click.option('--curriculum', default='epoch',
  type=click.Choice(['epoch', 'value_gated', 'value_free', 'dist_only', 'dir_only', 'dist_and_dir']),
  help="Navigate.get_reward's w_speed/w_dir/w_dist schedule: 'epoch' (time-based), "
       "'value_gated' (performance-based, speed->dir->dist order enforced), "
       "'value_free' (performance-based, no ordering), or fixed-weight baselines with "
       "no curriculum at all -- 'dist_only', 'dir_only', 'dist_and_dir' (w_dir=w_dist=0.5). "
       "Only matters for --task=navigate.")
@click.option('--warm_start', is_flag=True, default=False,
  help='Resume from an old checkpoint instead of training from scratch. Requires --warm_start_path.')
@click.option('--warm_start_path', default=None,
  help="Path to the itr_N.pkl (or params.pkl) checkpoint to warm-start from. Only used if --warm_start is set.")
@click.option('--seed', default=1, help='Random seed.')
@wrap_experiment(
  prefix='experiments/swimmer/swimmer_es',
  name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
  name_parameters=None,
  snapshot_mode='gap_and_last',
  snapshot_gap=5,
  # Every MPI process runs this decorator, so concurrent repo archiving races with logs being written to data/.
  archive_launch_repo=False,
)
def swimmer_es(
  ctxt,
  job,
  task,
  n_epochs,
  n_proc,
  population_size,
  sigma,
  optim,
  lr,
  l2_coeff,
  n_evals,
  policy,
  policy_hidden_sizes,
  ncap_opts,
  include_speed_control,
  include_turn_control,
  osc_period,
  curriculum,
  warm_start,
  warm_start_path,
  seed,
):
  assert population_size % n_proc == 0 and population_size % 2 == 0

  # Extract MPI multi-process info.
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  seed = 1000 * seed + rank  # Worker-specific seed based on task id.
  deterministic.set_seed(seed)

  # swim() has no 'curriculum' parameter, so only pass it for navigate.
  task_kwargs = dict(random=seed)
  if task == 'navigate':
    task_kwargs['curriculum'] = curriculum

  env = DMControlEnv(suite.load('swimmer', task, task_kwargs=task_kwargs)) #this loads the task from envs.py

  assert policy in ('mlp', 'ncap')
  if policy == 'mlp':
    Policy = GaussianMLPPolicy
    policy_kwargs = dict(
      env_spec=env.spec,
      hidden_sizes=tuple(int(h) for h in policy_hidden_sizes.split(',')),
      hidden_nonlinearity=torch.tanh,
      output_nonlinearity=None,
    )
  elif policy == 'ncap':
    assert len(ncap_opts) == 3
    sharing, constraints, constant_init = tuple(bool(int(x)) for x in ncap_opts)
    Policy = SwimmerPolicy
    policy_kwargs = dict(
      env_spec=env.spec,
      n_joints=env.spec.action_space.shape[0],
      oscillator_period=osc_period,
      # swim has no target in its observation, so task_module must not run for it.
      use_task='learned' if task == 'navigate' else None,  # None, 'hardcoded', 'learned'
      use_weight_sharing=sharing,
      use_weight_constraints=constraints,
      use_weight_constant_init=constant_init,
      include_speed_control=include_speed_control,
      include_turn_control=include_turn_control,
    )

  assert optim in ('adam', 'sgd')
  if optim == 'adam':
    optimizer = torch.optim.Adam
  elif optim == 'sgd':
    optimizer = torch.optim.SGD

  class Agent():
    def __init__(self, device=torch.device('cpu')):
      self.device = device
      self.env = DMControlEnv(suite.load('swimmer', task, task_kwargs=task_kwargs))

    def set_current_epoch(self, epoch: int):
      """Helper method to update current epoch from training loop."""
      self.env._env.task.set_epoch(epoch)

    def get_curriculum_ema(self):
      """(ema_speed, ema_dir, ema_dist), or None for tasks without a
      curriculum (e.g. swim). For broadcasting master's state to workers."""
      task = self.env._env.task
      return task.get_curriculum_ema() if hasattr(task, 'get_curriculum_ema') else None

    def set_curriculum_ema(self, ema_state):
      """Overwrite this process's running averages, e.g. on a worker
      receiving master's broadcast. No-op if ema_state is None."""
      if ema_state is not None:
        self.env._env.task.set_curriculum_ema(ema_state)

    def update_curriculum_ema(self):
      """Feed the most recent rollout()'s sub-reward averages into the
      task's curriculum EMA. Call this exactly once per epoch, right after
      rollout()'ing the just weight-updated policy -- not after any
      exploratory ES candidate -- so the EMA reflects a clean measurement of
      the actual trained policy, not noisy perturbations around it."""
      task = self.env._env.task
      if hasattr(task, 'update_curriculum_ema'):
        task.update_curriculum_ema(
          self.last_speed_reward, self.last_dir_reward, self.last_dist_reward)

    def rollout(self, policy):
      total_reward = 0.
      all_target_distances = []
      all_velocities = []
      all_speed_rewards = []
      all_dir_rewards = []
      all_dist_rewards = []
      for _ in range(n_evals):
        data = rollout(self.env, policy, deterministic=True)
        total_reward += sum(data['rewards'])

        # Only the navigate task observes 'to_target'; swim has no target.
        # Use only the final timestep's distance (how close it ended up), not every
        # timestep of the episode (which would measure typical distance while
        # swimming, not navigation accuracy).
        if 'to_target' in data['env_infos']:
          final_to_target = data['env_infos']['to_target'][-1]
          all_target_distances.append(np.linalg.norm(final_to_target))

        # Mean speed within this episode, one value per episode (like
        # all_target_distances above), so a longer episode doesn't get more
        # weight than a shorter one when aggregated across n_evals below.
        all_velocities.append(np.mean(-data['env_infos']['body_velocities'][:, 1]))

        # This episode's sub-reward history, for update_curriculum_ema() --
        # only present for tasks with a curriculum (navigate), not swim.
        task = self.env._env.task
        if hasattr(task, '_episode_speed_rewards'):
          all_speed_rewards.append(np.mean(task._episode_speed_rewards))
          all_dir_rewards.append(np.mean(task._episode_dir_rewards))
          all_dist_rewards.append(np.mean(task._episode_dist_rewards))

      mean_distance = np.median(all_target_distances) if all_target_distances else np.nan
      mean_velocity = np.median(all_velocities)

      # Stashed on self (not returned) so update_curriculum_ema() can use
      # them without changing this method's return signature.
      if all_speed_rewards:
        self.last_speed_reward = np.median(all_speed_rewards)
        self.last_dir_reward = np.median(all_dir_rewards)
        self.last_dist_reward = np.median(all_dist_rewards)

      return total_reward / n_evals, mean_distance, mean_velocity

  start_time = time.time()
  epoch_time = time.time()

  snapshotter = Snapshotter(
    snapshot_dir=ctxt.snapshot_dir,
    snapshot_mode=ctxt.snapshot_mode,
    snapshot_gap=ctxt.snapshot_gap,
  )

  class ES(estorch.ES):
    def log(self):
      nonlocal epoch_time, start_time
      now = time.time()
      logger.log(f'Time {now - start_time:.2f} s')
      logger.log(f'EpochTime {now - epoch_time:.2f} s')
      epoch_time = now
      epoch = self.step + 1
      print(self.current_epoch, epoch, self.step)

      tabular.record('Epoch', epoch)
      tabular.record('TotalEnvSteps', epoch * population_size * n_evals * env.spec.max_episode_length)
      tabular.record('CurrentReturn', self.episode_reward)
      tabular.record('PopulationAvgReturn', np.mean(self.population_returns))
      tabular.record('PopulationMinReturn', np.min(self.population_returns))
      tabular.record('PopulationMaxReturn', np.max(self.population_returns))
      tabular.record('DistanceFromTarget', self.other_rewards['mean_distance'])
      tabular.record('MeanVelocity', self.other_rewards['mean_velocity'])
      logger.log(tabular)

      logger.dump_all(epoch)
      tabular.clear()

      # Raw per-candidate returns for this epoch (population_size values), so the
      # full distribution can be inspected later -- progress.csv only keeps the
      # min/avg/max summary, which hides e.g. bimodal "some candidates totally
      # broken, some fine" population structure.
      pop_returns_path = os.path.join(ctxt.snapshot_dir, 'population_returns.csv')
      write_header = not os.path.isfile(pop_returns_path)
      with open(pop_returns_path, 'a') as f:
        if write_header:
          f.write('Epoch,Return\n')
        for r in self.population_returns.flatten():
          f.write(f'{epoch},{r}\n')

      snapshotter.save_itr_params(
        epoch,
        dict(
          policy_cls=Policy,
          policy_kwargs=policy_kwargs,
          policy_dict=es.policy.state_dict(), 
          best_policy_dict=es.best_policy_dict, 
        ),
      )      

  es = ES(
    policy=Policy,
    agent=Agent,
    optimizer=torch.optim.Adam,
    population_size=population_size,
    sigma=sigma,
    device=torch.device('cpu'),
    policy_kwargs=policy_kwargs,
    agent_kwargs=dict(),
    optimizer_kwargs=dict(lr=lr, weight_decay=l2_coeff),
  )

  # --- one-off warm start from an old checkpoint ---
  if rank == 0 and warm_start:
    import io, pickle, re, glob

    assert warm_start_path, '--warm_start requires --warm_start_path to be set.'

    class CPU_Unpickler(pickle.Unpickler):
      def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
          return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

    WARM_START_PATH = warm_start_path
    with open(WARM_START_PATH, 'rb') as f:
      ckpt = CPU_Unpickler(f).load()
    es.policy.load_state_dict(ckpt['best_policy_dict'])  # use ckpt['policy_dict'] for the last (not best) weights instead
    es.best_policy_dict = ckpt['best_policy_dict']
    es.best_reward = -float('inf')  # so logging still improves monotonically from here

    # Resume the reward decay schedule (self.current_epoch) from where the old run left off.
    warm_start_dir = os.path.dirname(WARM_START_PATH)
    match = re.search(r'itr_(\d+)\.pkl$', os.path.basename(WARM_START_PATH))
    if match:
      epoch_offset = int(match.group(1))
    else:
      itr_files = glob.glob(os.path.join(warm_start_dir, 'itr_*.pkl'))
      itr_nums = [int(re.search(r'itr_(\d+)\.pkl$', f).group(1)) for f in itr_files]
      if itr_nums:
        epoch_offset = max(itr_nums)
      else:
        import pandas as pd
        epoch_offset = int(pd.read_csv(os.path.join(warm_start_dir, 'progress.csv'))['Epoch'].max())
    es.current_epoch = epoch_offset

    print(f'Warm-started policy from {WARM_START_PATH}, current_epoch set to {epoch_offset}')
  # --- end warm start ---

  if rank == 0 and os.getenv('MPI_PARENT') is not None:
    # Manager process is only one that logs.
    logger.log(f'Training started for {n_epochs} epochs on worker {rank} of {size} ...')

    # Record the curriculum's actual tuning constants for this run (not
    # exposed as CLI flags, so not already in variant.json) -- lets later
    # runs be compared against exactly what each one used.
    task_obj = env._env.task
    if hasattr(task_obj, 'get_curriculum_params'):
      with open(os.path.join(ctxt.snapshot_dir, 'curriculum_params.json'), 'w') as f:
        json.dump(task_obj.get_curriculum_params(), f, indent=2)
  elif n_proc > 1:
    # Worker processes shouldn't have a log dir.
    shutil.rmtree(ctxt.snapshot_dir)

  es.train(n_steps=n_epochs, n_proc=n_proc, hwthread=True)
  logger.log(f'Training done.')


if __name__ == '__main__':
  swimmer_es()
