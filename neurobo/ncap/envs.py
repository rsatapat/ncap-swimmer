import numpy as np
import collections

import dm_control.suite.swimmer as swimmer
from dm_control.rl import control
from dm_control.utils import rewards

# Horizontal speed above which the reward is 1.
_SWIM_SPEED = 0.1


@swimmer.SUITE.add() # adds swim task to the suite of tasks in the swimmer domain
def swim(
  n_links=6,
  desired_speed=_SWIM_SPEED,
  time_limit=swimmer._DEFAULT_TIME_LIMIT,
  random=None,
  environment_kwargs={},
):
  """Returns the Swim task for a n-link swimmer."""
  model_string, assets = swimmer.get_model_and_assets(n_links)

  # you can use the physics object to access the underlying MuJoCo simulation, including the model and data structures.
  physics = swimmer.Physics.from_xml_string(model_string, assets=assets)
  
  task = Swim(desired_speed=desired_speed, random=random)
  return control.Environment(
    physics,
    task,
    time_limit=time_limit,
    control_timestep=swimmer._CONTROL_TIMESTEP,
    **environment_kwargs,
  )


@swimmer.SUITE.add() # adds navigate task to the suite of tasks in the swimmer domain
def navigate(
  n_links=6,
  target_pos=(-2.0, 1.0),
  desired_speed=_SWIM_SPEED,
  time_limit=swimmer._DEFAULT_TIME_LIMIT,
  random=None,
  environment_kwargs={},
  curriculum='epoch',  # 'epoch', 'value_gated', or 'value_free' -- see Navigate.get_reward
):
  """Returns the Navigate task for a n-link swimmer."""
  model_string, assets = swimmer.get_model_and_assets(n_links)
  physics = swimmer.Physics.from_xml_string(model_string, assets=assets)
  task = Navigate(target_pos=target_pos, desired_speed=desired_speed, random=random,
                   curriculum=curriculum)
  # print(time_limit, swimmer._CONTROL_TIMESTEP)
  return control.Environment(
    physics,
    task,
    time_limit=time_limit,
    control_timestep=swimmer._CONTROL_TIMESTEP,
    **environment_kwargs,
  )


class Swim(swimmer.Swimmer):
  """Task to swim forwards at the desired speed."""
  def __init__(self, desired_speed=_SWIM_SPEED, **kwargs):
    super().__init__(**kwargs)
    self._desired_speed = desired_speed

  def initialize_episode(self, physics):
    super().initialize_episode(physics)
    # Hide target by setting alpha to 0.
    physics.named.model.mat_rgba['target', 'a'] = 0
    physics.named.model.mat_rgba['target_default', 'a'] = 0
    physics.named.model.mat_rgba['target_highlight', 'a'] = 0

  def set_epoch(self, epoch: int):
    """Helper method to update current epoch from training loop."""
    self.current_epoch = epoch

  def get_observation(self, physics):
    """Returns an observation of joint angles and body velocities."""
    obs = collections.OrderedDict()
    obs['joints'] = physics.joints()
    obs['body_velocities'] = physics.body_velocities()
    return obs

  def get_reward(self, physics):
    """Returns a smooth reward that is 0 when stopped or moving backwards, and rises linearly to 1
    when moving forwards at the desired speed."""
    forward_velocity = -physics.named.data.sensordata['head_vel'][1]
    return rewards.tolerance(
      forward_velocity,
      bounds=(self._desired_speed, float('inf')),
      margin=self._desired_speed,
      value_at_margin=0.,
      sigmoid='linear',
    )


# class Navigate(swimmer.Swimmer):
#   """Task to navigate to the target location."""
#   pass

class Navigate(swimmer.Swimmer):
    """A task to reward the swimmer for navigating to a specific coordinate."""

    def __init__(self, target_pos=(2.0, 2.0), desired_speed=_SWIM_SPEED, random=None,
                 curriculum='epoch', ema_alpha=0.1):
        """Initialize the task.
        Args:
            target_pos: A tuple/list of (X, Y) coordinates for the target.
            desired_speed: The speed at which the swimmer should move.
            random: Optional random number generator.
            curriculum: how get_reward schedules w_speed/w_dir/w_dist --
                'epoch' (time-based, see get_reward), 'value_gated' or
                'value_free' (performance-based, see get_reward).
            ema_alpha: smoothing factor for the running averages of
                speed_reward/dir_reward/dist_reward used by the 'value_*'
                curricula. These are updated once per epoch (see
                update_curriculum_ema), not once per step, so this is a
                per-epoch time constant: 0.1 means ~10 epochs to adapt,
                not ~10 steps. Smaller = slower/smoother.
        """
        super().__init__(random=random)
        self._target_pos = np.array(target_pos)
        self._desired_speed = desired_speed
        self.distance_to_target = np.linalg.norm(target_pos)  # Initialize distance to target
        # Distance from the swimmer's start position (0, 0) to the target. dist_reward's
        # bounds/margin scale off this so the reward shapes sensibly regardless of how
        # close or far the target is.
        self._initial_distance = np.linalg.norm(self._target_pos)
        self.current_epoch = 0

        self.curriculum = curriculum
        self._ema_alpha = ema_alpha
        # Running averages of each sub-reward, used by the 'value_*' curricula
        # as a slow-moving measure of "how mastered is this objective right
        # now" -- start at 0 (nothing mastered yet) so value_gated correctly
        # starts fully prioritizing speed, the first stage. Updated only via
        # update_curriculum_ema(), once per epoch, from a clean aggregate --
        # not automatically in get_reward, which would mix in every noisy
        # exploratory ES candidate along with the actual trained policy.
        self._ema_speed = 0.0
        self._ema_dir = 0.0
        self._ema_dist = 0.0

        # Curriculum shape constants -- attributes (not CLI flags) so
        # get_curriculum_params() can expose them for train_swimmer_es.py to
        # record into curriculum_params.json per run, without bloating the
        # CLI. Edit these directly to tune; see get_reward for how they're used.
        self.dir_peak_epoch = 200      # 'epoch': epoch where w_dir peaks
        self.dir_width = 150           # 'epoch': how fast w_dir rises/falls around its peak
        self.dist_start_epoch = 250    # 'epoch': epoch around which w_dist takes over
        self.dist_tau = 100            # 'epoch': how fast w_dist rises
        self.speed_tau = 150           # 'epoch': how fast w_speed's priority decays
        self.epoch_temperature = 0.5   # 'epoch': softmax temperature
        self.value_temperature = 0.1   # 'value_gated'/'value_free': softmax temperature

        # dir_reward = ((alignment + 1) / 2) ** dir_reward_p. p=1 is plain
        # cosine similarity (zero gradient exactly at full misalignment,
        # where a stuck policy tends to sit). p<1 stretches out
        # differences near full misalignment, giving ES's rank-based
        # gradient more to work with there. See track_model.py's
        # plot_dir_reward_shapes to compare values before changing this.
        self.dir_reward_p = 0.25

    def get_curriculum_params(self):
        """Return this task's curriculum settings as a plain dict, for
        train_swimmer_es.py to record into curriculum_params.json -- so a
        later comparison across runs knows exactly which tuning each one
        used, even though these aren't exposed as CLI flags.
        """
        return {
            'curriculum': self.curriculum,
            'ema_alpha': self._ema_alpha,
            'dir_peak_epoch': self.dir_peak_epoch,
            'dir_width': self.dir_width,
            'dist_start_epoch': self.dist_start_epoch,
            'dist_tau': self.dist_tau,
            'speed_tau': self.speed_tau,
            'epoch_temperature': self.epoch_temperature,
            'value_temperature': self.value_temperature,
            'dir_reward_p': self.dir_reward_p,
        }

    def set_epoch(self, epoch: int):
        """Helper method to update current epoch from training loop."""
        self.current_epoch = epoch

    def update_curriculum_ema(self, speed_reward, dir_reward, dist_reward):
        """Update the 'value_*' curricula's running averages from one clean,
        externally-supplied measurement, instead of from every per-step,
        per-candidate reward seen during ES's exploratory rollouts.

        Intended to be called exactly once per epoch, with the current
        (just weight-updated) policy's own evaluation -- e.g. the
        mean_speed_reward/mean_dir_reward/mean_dist_reward that
        train_swimmer_es.py's Agent.rollout computes for its deterministic,
        post-update rollout -- not with values from any perturbed candidate.

        Args:
            speed_reward: this epoch's aggregate speed_reward (e.g. mean
                across steps and across n_evals episodes).
            dir_reward: same, for dir_reward.
            dist_reward: same, for dist_reward.
        """
        self._ema_speed += self._ema_alpha * (speed_reward - self._ema_speed)
        self._ema_dir += self._ema_alpha * (dir_reward - self._ema_dir)
        self._ema_dist += self._ema_alpha * (dist_reward - self._ema_dist)

    def get_curriculum_ema(self):
        """Return (ema_speed, ema_dir, ema_dist), e.g. to broadcast from the
        master process to workers so every process's curriculum weighting
        agrees (mirrors why current_epoch needs the same treatment)."""
        return (self._ema_speed, self._ema_dir, self._ema_dist)

    def set_curriculum_ema(self, ema_state):
        """Overwrite the running averages directly, e.g. on a worker process
        receiving the master's broadcast (see get_curriculum_ema)."""
        self._ema_speed, self._ema_dir, self._ema_dist = ema_state

    def initialize_episode(self, physics):
        """Resets the environment state at the beginning of an episode."""
        super().initialize_episode(physics)
        # Per-step sub-reward history for this episode, so Agent.rollout can
        # read back this episode's average after it finishes, for
        # update_curriculum_ema(). Cleared at the start of the next episode,
        # so right after an episode ends these still hold its full history.
        self._episode_speed_rewards = []
        self._episode_dir_rewards = []
        self._episode_dist_rewards = []
        # Optional: Randomize the target position slightly each episode if desired:
        # self._target_pos = self.random.uniform(low=-3.0, high=3.0, size=2)

        # dist_reward's scaling assumes the swimmer starts at (0, 0); pin the root
        # position explicitly rather than relying on the base class leaving it there.
        physics.named.data.qpos['rootx'] = 0.
        physics.named.data.qpos['rooty'] = 0.

        # super().initialize_episode() randomizes rootz (heading) uniformly over the
        # full circle. That can start the swimmer facing ~180 degrees away from the
        # target, needing a hard U-turn before it can make any progress at all. Bias
        # it instead to within +-45 degrees of facing the target.
        target_bearing = np.arctan2(self._target_pos[0], -self._target_pos[1])
        heading_spread = np.pi / 4  # +-45 degrees
        physics.named.data.qpos['rootz'] = (
            target_bearing + self.random.uniform(-heading_spread, heading_spread))
        physics.forward()  # Recompute kinematics so xmat below reflects the new heading.

        # The sign of the rootz-to-world-heading mapping above is derived, not
        # verified against a live sim. Read back the body's actual world-frame
        # forward direction (-head_yaxis -- a different sensor than
        # speed_reward's head_vel, but the same forward convention: head_vel[1]
        # is velocity projected onto head_yaxis, so speed_reward's -head_vel[1]
        # and this -head_yaxis both mean "forward is the -head_yaxis direction")
        # and flip 180 degrees if it's facing away from the target, so the bias
        # is correct regardless of that assumption.
        forward_world = -physics.named.data.sensordata['head_yaxis'][:2]
        if np.dot(forward_world, self._target_pos) < 0:
            physics.named.data.qpos['rootz'] += np.pi
            physics.forward()

    def get_observation(self, physics):
        """Returns joint state, body velocities, and relative vector to the target."""
        observations = collections.OrderedDict()
          
        # 1. Keep original body sensing (joint angles & angular velocities)
        observations['joints'] = physics.joints()
        observations['body_velocities'] = physics.body_velocities()
        observations['head_orientation'] = np.concatenate([
            physics.named.data.sensordata['head_xaxis'][:2],
            -physics.named.data.sensordata['head_yaxis'][:2],  # negated: forward is -head_yaxis,
                                                                # same convention as
                                                                # speed_reward/initialize_episode
            ])

        # 2. Spatial navigation info, in the worm's own frame (forward/right) via
        # head_orientation, then as polar (bearing, distance) instead of raw x/y --
        # bearing in x's old "turn" slot, distance in y's old "speed" slot.
        head_pos = physics.named.data.geom_xpos['head'][:2] # [x, y] coordinates
        to_target_world = self._target_pos - head_pos
        head_lateral, head_forward = observations['head_orientation'][:2], observations['head_orientation'][2:]
        x = np.dot(to_target_world, head_lateral)  # lateral offset, body frame
        y = np.dot(to_target_world, head_forward)  # forward offset, body frame

        bearing = np.arctan2(x, y)  # 0 = dead ahead, +-pi/2 = directly to the side
        distance = np.linalg.norm(to_target_world)
        observations['to_target'] = np.array([bearing, distance])

        self.distance_to_target = distance
        
        return observations

    def get_reward(self, physics):
        forward_velocity = -physics.named.data.sensordata['head_vel'][1]
        # print(forward_velocity)
        speed_reward = rewards.tolerance(forward_velocity,bounds=(self._desired_speed, float('inf')),\
                                         margin=self._desired_speed,value_at_margin=0.,sigmoid='linear',)
        # Disabled: redundant with (and fights) the softmax w_speed schedule
        # below, which already decays speed's contribution over epochs via
        # speed_tau -- stacking this on top double-decays speed_reward and
        # adds a discontinuity at the 80% threshold.
        # if forward_velocity >= 0.8 * self._desired_speed:
        #     speed_reward = speed_reward * np.exp(-self.current_epoch / 30)

        """Returns a reward bound between [0, 1] based on proximity."""
        head_pos = physics.named.data.geom_xpos['head'][:2]
        # # Sum up the capsule lengths or geom sizes defined in physics.model
        # # For capsule geoms, size[1] (or size[0] depending on orientation) defines segment half-height
        # geom_ids = [i for i in range(physics.model.ngeom) 
        #             if 'target' not in physics.model.id2name(i, 'geom').lower()]
        # # MuJoCo capsule size layout: [radius, half_length, ...]
        # total_static_length = sum(physics.model.geom_size[gid][1] * 2 for gid in geom_ids)
        # print(f"Static Swimmer Model Length: {total_static_length:.4f}")
        
        # Calculate the Euclidean distance to the target destination
        distance = np.linalg.norm(self._target_pos - head_pos)
        # Scale "close enough" and the falloff margin to the target's distance from the
        # swimmer's start, so this works whether the target is close (e.g. (-1, 1)) or
        # far: reward is ~0 at the starting distance and rises to 1 within 10% of it.
        close_enough = 0.1 * self._initial_distance
        margin = self._initial_distance - close_enough
        dist_reward = rewards.tolerance(distance, bounds=(0, close_enough), margin=margin, sigmoid='quadratic')

        # 3. Direction Vector to Target (Normalized)
        target_vec = self._target_pos - head_pos
        if distance > 1e-6:
            target_dir = target_vec / distance  # Normalize to get direction
        else:
            target_dir = np.array([1.0, 0.0]) # Default fallback if right on top of target

        # 4. Head Facing Vector (Normalized) -- forward is -head_yaxis, not head_xaxis
        # (head_xaxis is the lateral/sideways axis), same convention as
        # speed_reward/initialize_episode/get_observation.
        head_facing = -physics.named.data.sensordata['head_yaxis'][:2]

        # 5. Cosine Alignment [-1.0 to 1.0]
        alignment = np.dot(head_facing, target_dir)
        # Rescale to [0, 1], then reshape with dir_reward_p (<1 stretches out
        # differences near full misalignment -- see self.dir_reward_p).
        dir_reward = (0.5 * (alignment + 1.0)) ** self.dir_reward_p
        
        # Record this step's sub-rewards for Agent.rollout to read back after
        # the episode ends (see initialize_episode / update_curriculum_ema).
        # Recorded regardless of self.curriculum -- cheap, and keeps
        # inspection/switching curricula consistent.
        self._episode_speed_rewards.append(speed_reward)
        self._episode_dir_rewards.append(dir_reward)
        self._episode_dist_rewards.append(dist_reward)

        # Using the same dist_reward and dir_reward calculated above: combine
        # them with speed_reward via one of a few curricula.
        #
        # Fixed-weight baselines, for comparison against an actual curriculum
        # -- no schedule, no softmax, just a constant weighting every step.
        FIXED_WEIGHTS = {
            'dist_only': (0.0, 0.0, 1.0),      # (w_speed, w_dir, w_dist)
            'dir_only': (0.0, 1.0, 0.0),
            'dist_and_dir': (0.0, 0.5, 0.5),
        }
        if self.curriculum in FIXED_WEIGHTS:
            w_speed, w_dir, w_dist = FIXED_WEIGHTS[self.curriculum]
            return (w_dist * dist_reward) + (w_dir * dir_reward) + (w_speed * speed_reward)

        # Actual curricula, each a softmax over three "priority" logits so
        # w_speed + w_dir + w_dist == 1 always -- no hand-off trough where all
        # three are simultaneously small, and no term is ever hard-zeroed,
        # just made negligible.
        if self.curriculum == 'epoch':
            # Time-based (tuned for n_epochs=300, via track_model.py's
            # plot_reward_weight_schedule widget): w_speed dominates early so
            # the swimmer first learns to move at all, w_dir peaks
            # mid-training (rise then fall) once propulsion exists, and
            # w_dist rises late and takes over for precise target-reaching.
            temperature = self.epoch_temperature  # lower = sharper hand-offs, higher = more overlap

            logit_speed = -self.current_epoch / self.speed_tau
            logit_dir = -0.5 * ((self.current_epoch - self.dir_peak_epoch) / self.dir_width) ** 2
            logit_dist = (self.current_epoch - self.dist_start_epoch) / self.dist_tau

        elif self.curriculum == 'value_gated':
            # Performance-based, with the same speed->dir->dist ordering as
            # 'epoch' enforced explicitly: dir's priority only rises once
            # speed is mastered (gated by self._ema_speed), and dist's only
            # rises once both speed and dir are mastered.
            temperature = self.value_temperature
            logit_speed = 1 - self._ema_speed
            logit_dir = (1 - self._ema_dir) * self._ema_speed
            logit_dist = (1 - self._ema_dist) * self._ema_dir * self._ema_speed

        elif self.curriculum == 'value_free':
            # Performance-based, no ordering: each term's priority is just
            # how far it is from being mastered, independent of the others --
            # whichever objective currently has the most room to improve
            # gets the most weight, even if that's dist before speed/dir.
            temperature = self.value_temperature
            logit_speed = 1 - self._ema_speed
            logit_dir = 1 - self._ema_dir
            logit_dist = 1 - self._ema_dist

        else:
            raise ValueError(f'Unknown curriculum {self.curriculum!r}')

        logits = np.array([logit_speed, logit_dir, logit_dist]) / temperature
        logits -= logits.max()
        w_speed, w_dir, w_dist = np.exp(logits) / np.exp(logits).sum()

        total_reward = (w_dist * dist_reward) + (w_dir * dir_reward) + (w_speed * speed_reward)
        # print(total_reward, dist_reward, dir_reward, speed_reward, w_dist, w_dir, w_speed)
        return total_reward