"""Helper functions for track_model.ipynb: loading ES training runs and
plotting their progress.
"""
import ast
import os
import io
import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _infer_task(df):
    has_target_data = 'DistanceFromTarget' in df.columns and df['DistanceFromTarget'].notna().any()
    return 'navigate' if has_target_data else 'swim'


def _load_progress(csv_path):
    """Load progress.csv, or concatenate several in order (e.g. a run plus its
    warm-started continuation) into one continuous DataFrame -- each run after
    the first has its Epoch column offset to continue from the previous one's
    last epoch, instead of overlapping back at epoch 1.

    Raises:
        ValueError: if the given files don't all belong to the same task
            (swim vs. navigate) -- a sign they aren't actually one continued run.
    """
    if isinstance(csv_path, (list, tuple)):
        dfs = []
        epoch_offset = 0
        tasks = []
        for path in csv_path:
            df = pd.read_csv(path)
            tasks.append(_infer_task(df))
            df['Epoch'] += epoch_offset
            epoch_offset = df['Epoch'].max()
            dfs.append(df)
        if len(set(tasks)) > 1:
            raise ValueError(
                f"Won't combine files from different tasks: {list(zip(csv_path, tasks))}")
        return pd.concat(dfs, ignore_index=True)
    return pd.read_csv(csv_path)


def plot_training_progress(csv_path, run_info=None, save=True, save_dir='.'):
    """Plot reward (avg/min/max/current), velocity and (for navigate runs)
    distance from target vs epoch.

    The task is inferred straight from progress.csv: 'DistanceFromTarget' is
    logged for every run, but it's all NaN when there's no target (swim), so a
    run is 'navigate' only if that column actually has values. This is more
    reliable than trusting variant.json, which can disagree with what was
    actually logged.

    Args:
        csv_path: path to a progress.csv, or a list of paths (in chronological
            order) to stitch into one continuous run -- e.g. an original run
            plus a warm-started continuation in a separate folder. All files
            must belong to the same task (swim vs. navigate).
        run_info: optional dict from extract_run_info(). When given, a short summary
            of the run's key settings is printed in a box above the plots so each
            figure is self-describing.
        save: if True, save the figure as an SVG named after the run folder.
        save_dir: directory to save the figure in.

    Returns:
        (fig, ax, df)
    """
    df = _load_progress(csv_path)
    task = _infer_task(df)

    # (column, y-label) for each subplot, top to bottom
    panels = [('PopulationAvgReturn', 'Reward')]
    if task == 'navigate':
        panels.append(('DistanceFromTarget', 'Distance from target'))
    panels.append(('MeanVelocity', 'Velocity'))

    summary = format_run_summary(run_info) if run_info is not None else None
    # Reserve extra figure height for the summary box, roughly one line per row of text.
    n_summary_lines = summary.count('\n') + 1 if summary else 0
    extra_h = 0.35 * n_summary_lines + 0.15 if summary else 0

    fig, ax = plt.subplots(len(panels), 1, figsize=(8, 3 * len(panels) + extra_h), sharex=True)
    for a, (col, label) in zip(ax, panels):
        if col == 'PopulationAvgReturn' and {'PopulationMinReturn', 'PopulationMaxReturn'}.issubset(df.columns):
            # Shade the population's min-max range behind the average return, and
            # overlay CurrentReturn (the actual trained policy's own performance,
            # vs. the population's disposable perturbed candidates).
            a.fill_between(df['Epoch'], df['PopulationMinReturn'], df['PopulationMaxReturn'],
                            alpha=0.2, label='min–max range')
            a.plot(df['Epoch'], df[col], label='avg')
            if 'CurrentReturn' in df.columns:
                a.plot(df['Epoch'], df['CurrentReturn'], label='current', alpha=0.8, linewidth=1)
            a.legend(fontsize=8, loc='upper left')
        else:
            a.plot(df['Epoch'], df[col])
        a.set_ylabel(label)
    ax[-1].set_xlabel('Epoch')

    if isinstance(csv_path, (list, tuple)):
        run_name = '+'.join(os.path.basename(os.path.dirname(os.path.abspath(p))) for p in csv_path)
    else:
        run_name = os.path.basename(os.path.dirname(os.path.abspath(csv_path)))

    title = f'{task} — {run_name}'
    if task == 'navigate' and run_info is not None:
        curriculum = run_info['variant'].get('curriculum')
        if curriculum:
            title = f'{task} ({curriculum}) — {run_name}'
    ax[0].set_title(title)
    fig.tight_layout()

    if summary:
        # Shrink the plot area to free a strip at the top, then print the summary there.
        top = 1 - extra_h / fig.get_size_inches()[1]
        fig.subplots_adjust(top=top)
        fig.text(0.5, 0.995, summary, ha='center', va='top', fontsize=8, family='monospace',
                  bbox=dict(boxstyle='round', facecolor='whitesmoke', edgecolor='gray'))

    if save:
        fig.savefig(os.path.join(save_dir, f'training_progress_{task}{run_name}.svg'), dpi=300)

    return fig, ax, df


def plot_population_returns(run_dir, kind='violin', max_epochs_shown=40, save=True, save_dir='.'):
    """Plot the distribution of per-candidate population returns across epochs.

    Reads population_returns.csv (one row per perturbed candidate per epoch),
    written by train_swimmer_es.py's ES.log(). This shows the full spread the
    aggregated PopulationAvgReturn/Min/Max in progress.csv hides -- e.g. whether
    the population is a tight cluster or bimodal (some candidates broken, some
    fine), which a huge min-max range with a low average usually means.

    Only available for runs trained after this logging was added -- older runs
    have no population_returns.csv.

    Args:
        run_dir: training run folder containing population_returns.csv.
        kind: 'violin' or 'box'.
        max_epochs_shown: downsample to at most this many epochs so the plot
            stays readable for long runs.
        save: if True, save the figure as an SVG named after the run folder.
        save_dir: directory to save the figure in.

    Returns:
        (fig, ax, df)
    """
    path = os.path.join(run_dir, 'population_returns.csv')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'{path} not found -- this run predates population-return logging.')
    df = pd.read_csv(path)

    epochs = sorted(df['Epoch'].unique())
    if len(epochs) > max_epochs_shown:
        step = int(np.ceil(len(epochs) / max_epochs_shown))
        epochs = epochs[::step]
        df = df[df['Epoch'].isin(epochs)]

    data = [df.loc[df['Epoch'] == e, 'Return'].values for e in epochs]
    width = max(1, (epochs[1] - epochs[0]) * 0.8) if len(epochs) > 1 else 1

    fig, ax = plt.subplots(figsize=(max(8, len(epochs) * 0.3), 5))
    if kind == 'violin':
        ax.violinplot(data, positions=epochs, widths=width, showmedians=True)
    elif kind == 'box':
        ax.boxplot(data, positions=epochs, widths=width)
    else:
        raise ValueError(f"kind must be 'violin' or 'box', got {kind!r}")

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Population return')
    run_name = os.path.basename(os.path.abspath(run_dir))
    ax.set_title(f'Population return distribution — {run_name}')
    fig.tight_layout()

    if save:
        fig.savefig(os.path.join(save_dir, f'population_returns_{run_name}.svg'), dpi=300)

    return fig, ax, df


class _StubUnpickler(pickle.Unpickler):
    """Unpickler for params.pkl that works without garage/akro installed.

    Loads torch tensors on CPU, and replaces any class that can't be
    imported in this environment (e.g. from garage or akro) with a stub
    class so unpickling still succeeds and the resulting objects can be
    inspected via their __dict__.
    """

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            import torch
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        try:
            return super().find_class(module, name)
        except ImportError:
            return type(name, (), {
                '_stub_qualname': f'{module}.{name}',
                '__setstate__': lambda self, s: self.__dict__.update(
                    s if isinstance(s, dict) else {'state': s}),
                '__repr__': lambda self: f'<{type(self).__name__} {self.__dict__}>',
            })


def _round(x, ndigits=4):
    """Recursively round floats inside (possibly nested) lists."""
    if isinstance(x, list):
        return [_round(v, ndigits) for v in x]
    if isinstance(x, float):
        return round(x, ndigits)
    return x


def _state_dict_to_json(state_dict):
    return {k: _round(v.tolist() if hasattr(v, 'tolist') else v) for k, v in state_dict.items()}


def _env_spec_to_json(env_spec):
    if env_spec is None:
        return None
    d = env_spec.__dict__
    result = {'max_episode_length': d.get('max_episode_length')}
    out_space = getattr(d.get('output_space'), '__dict__', None)
    in_space = getattr(d.get('input_space'), '__dict__', None)
    if out_space is not None:
        result['obs_dim'] = out_space.get('_shape', (None,))[-1]
    if in_space is not None:
        result['action_dim'] = in_space.get('_shape', (None,))[-1]
        low, high = in_space.get('low'), in_space.get('high')
        result['action_low'] = float(low.flat[0]) if hasattr(low, 'flat') else low
        result['action_high'] = float(high.flat[0]) if hasattr(high, 'flat') else high
    return result


def _class_name(cls):
    """Best-effort dotted name for a (possibly stub) class."""
    if hasattr(cls, '_stub_qualname'):
        return cls._stub_qualname
    module = getattr(cls, '__module__', None)
    qualname = getattr(cls, '__qualname__', None) or getattr(cls, '__name__', None) or str(cls)
    return f'{module}.{qualname}' if module else qualname


def load_curriculum_params(run_dir):
    """Load a run's curriculum_params.json, written once at training startup
    by train_swimmer_es.py from Navigate.get_curriculum_params() -- the
    curriculum constants (dir_width, dist_tau, etc.) that aren't exposed as
    CLI flags, so aren't in variant.json.

    Args:
        run_dir: folder containing curriculum_params.json.

    Returns:
        dict of curriculum settings, or None if this run predates this
        logging (e.g. a swim run, or an older navigate run).
    """
    path = os.path.join(run_dir, 'curriculum_params.json')
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_final_weights(run_dir):
    """Load the end-of-training policy weights from a run's params.pkl.

    Unlike extract_run_info()'s 'last' weights (rounded for JSON display),
    this returns the raw torch state_dict, suitable for loading straight into
    a policy via load_state_dict(). Use ckpt['best_policy_dict'] instead if
    you want the best-seen weights rather than the final ones.

    Args:
        run_dir: folder containing params.pkl.

    Returns:
        dict: torch state_dict of the policy at the end of training.
    """
    with open(os.path.join(run_dir, 'params.pkl'), 'rb') as f:
        data = _StubUnpickler(f).load()
    return data['policy_dict']


def extract_run_info(run_dir):
    """Read variant.json and params.pkl from a training run folder and
    collect the settings and weights that matter into one JSON-serializable dict.

    Args:
        run_dir: folder containing variant.json and params.pkl.

    Returns:
        dict with keys: run_dir, variant, policy, env_spec, weights, warnings.
    """
    run_dir = str(run_dir)

    with open(os.path.join(run_dir, 'variant.json')) as f:
        variant = json.load(f)

    with open(os.path.join(run_dir, 'params.pkl'), 'rb') as f:
        data = _StubUnpickler(f).load()

    policy_kwargs = dict(data['policy_kwargs'])
    env_spec = policy_kwargs.pop('env_spec', None)

    info = {
        'run_dir': run_dir,
        'variant': variant,
        'policy': {'class': _class_name(data['policy_cls']), **policy_kwargs},
        'env_spec': _env_spec_to_json(env_spec),
        'weights': {
            'n_params': len(data['best_policy_dict']),
            'best': _state_dict_to_json(data['best_policy_dict']),
            'last': _state_dict_to_json(data['policy_dict']),
        },
        'curriculum_params': load_curriculum_params(run_dir),
        'warnings': [],
    }

    use_task = policy_kwargs.get('use_task')
    if variant.get('task') == 'navigate' and not use_task:
        info['warnings'].append(
            f"variant task is 'navigate' but policy use_task is {use_task!r}: "
            "policy never received the target")

    return info


def format_run_summary(info):
    """Build a short text summary of a run's key settings, for annotating plots.

    Keeps it to a couple of lines: the task (plus use_task for navigate), then
    the training settings that most affect the result, plus a one-line warning
    when the task and policy disagree (e.g. navigate without a target input).

    Args:
        info: dict returned by extract_run_info().

    Returns:
        A short multi-line string.
    """
    variant = info['variant']
    policy = info['policy']
    env_spec = info.get('env_spec') or {}
    task = variant.get('task')
    use_task = policy.get('use_task')
    n_evals = variant.get('n_evals', 1)
    steps_per_eval = env_spec.get('max_episode_length')

    curriculum = variant.get('curriculum')
    header = task
    if task == 'navigate':
        header = f'{task} (use_task={use_task}, curriculum={curriculum})'
    settings = (f"epochs={variant.get('n_epochs')}  pop={variant.get('population_size')}  "
                f"sigma={variant.get('sigma')}  lr={variant.get('lr')}  "
                f"n_evals={n_evals}  steps/eval={steps_per_eval}")

    lines = [header, settings]

    params = info.get('curriculum_params')
    if params:
        if curriculum == 'epoch':
            lines.append(
                f"speed_tau={params['speed_tau']}  dir_peak={params['dir_peak_epoch']}  "
                f"dir_width={params['dir_width']}  dist_start={params['dist_start_epoch']}  "
                f"dist_tau={params['dist_tau']}  T={params['epoch_temperature']}")
        elif curriculum in ('value_gated', 'value_free'):
            lines.append(f"ema_alpha={params['ema_alpha']}  T={params['value_temperature']}")

    if task == 'navigate' and not use_task:
        lines.append('⚠ no target info (use_task=None)')
    return '\n'.join(lines)


def _parse_target(value):
    """Parse play.py's CSV-written target column back into a tuple or None."""
    if pd.isna(value) or value == 'None':
        return None
    return ast.literal_eval(value)


def load_sweep_results(csv_path):
    """Load play.py's run_sweep() output: a long-format CSV with one row per
    step of every (seed, target) run (columns: seed, target, step, x, y,
    velocity, reward).

    Args:
        csv_path: path to the sweep_results.csv written by run_sweep().

    Returns:
        DataFrame with 'target' parsed back into (x, y) tuples (or None for
        swim runs).
    """
    df = pd.read_csv(csv_path)
    df['target'] = df['target'].apply(_parse_target)
    return df


def _as_df(csv_path_or_df):
    return load_sweep_results(csv_path_or_df) if isinstance(csv_path_or_df, str) else csv_path_or_df


def _target_colors(targets):
    cmap = plt.get_cmap('tab10')
    return {t: cmap(i % 10) for i, t in enumerate(targets)}


def plot_sweep_trajectories(csv_path_or_df, save=True, save_dir='.'):
    """Plot the head (x, y) trajectory of every run in a sweep, colored by
    target (navigate) or by seed (swim, which has no target).

    Args:
        csv_path_or_df: path to run_sweep()'s CSV, or an already-loaded
            DataFrame from load_sweep_results().
        save: if True, save the figure as an SVG.
        save_dir: directory to save the figure in.

    Returns:
        (fig, ax, df)
    """
    df = _as_df(csv_path_or_df)
    is_navigate = df['target'].notna().any()

    fig, ax = plt.subplots(figsize=(6, 6))
    if is_navigate:
        targets = sorted(df['target'].dropna().unique(), key=str)
        colors = _target_colors(targets)
        for target in targets:
            color = colors[target]
            group = df[df['target'] == target]
            for seed, run in group.groupby('seed'):
                ax.plot(run['x'], run['y'], color=color, alpha=0.6, linewidth=1)
            ax.scatter(*target, color=color, marker='*', s=150,
                       edgecolor='black', zorder=5, label=f'target {target}')
        ax.legend(fontsize=8)
    else:
        for seed, run in df.groupby('seed'):
            ax.plot(run['x'], run['y'], alpha=0.7, linewidth=1, label=f'seed {seed}')
        ax.legend(fontsize=8)

    ax.scatter(0, 0, color='black', marker='o', s=40, zorder=5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # adjustable='datalim' (not the default 'box') keeps the subplot box itself
    # a normal shape and pads the data limits instead -- otherwise a lopsided
    # trajectory (e.g. far in x, barely moving in y) shrinks the box into a
    # thin strip, crowding the title/legend into almost no space.
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_title('Sweep trajectories')
    fig.tight_layout()

    if save:
        fig.savefig(os.path.join(save_dir, 'sweep_trajectories.svg'), dpi=300)

    return fig, ax, df


def plot_sweep_timeseries(csv_path_or_df, save=True, save_dir='.'):
    """Plot velocity and reward vs step for every run in a sweep, one line
    per (seed, target) combo, colored by target (or all one color for swim).

    Args:
        csv_path_or_df: path to run_sweep()'s CSV, or an already-loaded
            DataFrame from load_sweep_results().
        save: if True, save the figure as an SVG.
        save_dir: directory to save the figure in.

    Returns:
        (fig, ax, df)
    """
    df = _as_df(csv_path_or_df)
    is_navigate = df['target'].notna().any()
    colors = _target_colors(sorted(df['target'].dropna().unique(), key=str)) if is_navigate else {}

    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for (seed, target), run in df.groupby(['seed', 'target']):
        color = colors.get(target, 'tab:blue')
        ax[0].plot(run['step'], run['velocity'], color=color, alpha=0.5, linewidth=1)
        ax[1].plot(run['step'], run['reward'], color=color, alpha=0.5, linewidth=1)
    ax[0].set_ylabel('Velocity')
    ax[1].set_ylabel('Reward')
    ax[1].set_xlabel('Step')

    if is_navigate:
        handles = [plt.Line2D([0], [0], color=c, label=f'target {t}') for t, c in colors.items()]
        ax[0].legend(handles=handles, fontsize=8)

    fig.suptitle('Velocity and reward across the sweep')
    fig.tight_layout()

    if save:
        fig.savefig(os.path.join(save_dir, 'sweep_timeseries.svg'), dpi=300)

    return fig, ax, df


def plot_sweep_final_distance(csv_path_or_df, save=True, save_dir='.'):
    """Box plot of final distance-to-target across seeds, one box per target.

    Navigate-only: raises if the sweep has no targets (e.g. a swim run).

    Args:
        csv_path_or_df: path to run_sweep()'s CSV, or an already-loaded
            DataFrame from load_sweep_results().
        save: if True, save the figure as an SVG.
        save_dir: directory to save the figure in.

    Returns:
        (fig, ax, summary) where summary has one row per (seed, target) with
        its final_distance.
    """
    df = _as_df(csv_path_or_df)
    if df['target'].isna().all():
        raise ValueError('No targets in this sweep (looks like a swim run, not navigate).')

    rows = []
    for (seed, target), run in df.groupby(['seed', 'target']):
        last = run.iloc[-1]
        final_distance = float(np.hypot(last['x'] - target[0], last['y'] - target[1]))
        rows.append({'seed': seed, 'target': target, 'final_distance': final_distance})
    summary = pd.DataFrame(rows)

    targets = sorted(summary['target'].unique(), key=str)
    data = [summary.loc[summary['target'] == t, 'final_distance'].values for t in targets]

    fig, ax = plt.subplots(figsize=(max(6, len(targets) * 1.5), 5))
    ax.boxplot(data, labels=[str(t) for t in targets])
    ax.set_xlabel('Target')
    ax.set_ylabel('Final distance to target')
    ax.set_title('Final distance to target across seeds')
    fig.tight_layout()

    if save:
        fig.savefig(os.path.join(save_dir, 'sweep_final_distance.svg'), dpi=300)

    return fig, ax, summary


def softmax_reward_weights(epochs, n_epochs, speed_tau, dir_peak_epoch, dir_width,
                            dist_start_epoch, dist_tau, temperature=0.5):
    """Compute (w_speed, w_dir, w_dist) for the softmax-normalized reward
    schedule proposed for Navigate.get_reward: three epoch-dependent
    "priority" logits (speed decaying, direction peaking midway, distance
    rising) passed through a softmax, so the three weights always sum to 1.

    Matches Navigate.get_reward's 'epoch' curriculum exactly -- dir_peak_epoch
    is an independent dial here too (not derived from n_epochs), same as the
    actual self.dir_peak_epoch attribute it's tuned against.

    Args:
        epochs: array of epoch values to evaluate at.
        n_epochs: total epochs training will run for.
        speed_tau: time constant for speed's decay.
        dir_peak_epoch: epoch where direction's priority peaks.
        dir_width: how fast direction's bump rises/falls around its peak.
        dist_start_epoch: epoch around which distance's priority takes over.
        dist_tau: time constant for distance's rise.
        temperature: softmax temperature -- lower means sharper, more
            exclusive hand-offs between speed, direction and distance;
            higher means more overlap between them.

    Returns:
        (w_speed, w_dir, w_dist), each an array the same shape as epochs.
    """
    epochs = np.asarray(epochs, dtype=float)

    logit_speed = -epochs / speed_tau
    logit_dir = -0.5 * ((epochs - dir_peak_epoch) / dir_width) ** 2
    logit_dist = (epochs - dist_start_epoch) / dist_tau

    logits = np.stack([logit_speed, logit_dir, logit_dist]) / temperature
    logits -= logits.max(axis=0, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=0, keepdims=True)
    return weights[0], weights[1], weights[2]


def plot_reward_weight_schedule(n_epochs=300, speed_tau=40, dir_peak_epoch=90, dir_width=40,
                                 dist_start_epoch=160, dist_tau=60, temperature=0.5):
    """Plot w_speed/w_dir/w_dist vs epoch for the softmax reward schedule.

    Meant for interactive exploration (e.g. via ipywidgets.interact) of how
    these knobs shape the curriculum before committing values to
    Navigate.get_reward.

    Args:
        n_epochs: total epochs training will run for.
        speed_tau: time constant for speed's decay.
        dir_peak_epoch: epoch where direction's priority peaks.
        dir_width: how fast direction's bump rises/falls around its peak.
        dist_start_epoch: epoch around which distance's priority takes over.
        dist_tau: time constant for distance's rise.
        temperature: softmax temperature -- lower means sharper, more
            exclusive hand-offs between speed, direction and distance;
            higher means more overlap between them.

    Returns:
        (fig, ax)
    """
    epochs = np.arange(0, n_epochs + 1)
    w_speed, w_dir, w_dist = softmax_reward_weights(
        epochs, n_epochs, speed_tau, dir_peak_epoch, dir_width, dist_start_epoch, dist_tau, temperature)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, w_speed, label='w_speed')
    ax.plot(epochs, w_dir, label='w_dir')
    ax.plot(epochs, w_dist, label='w_dist')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weight')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f'Reward weight schedule (n_epochs={n_epochs}, speed_tau={speed_tau}, '
                 f'dir_peak_epoch={dir_peak_epoch}, dir_width={dir_width}, '
                 f'dist_start_epoch={dist_start_epoch}, dist_tau={dist_tau}, temperature={temperature})')
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()

    return fig, ax


def load_checkpoint_trajectories(csv_path):
    """Load play.py's run_checkpoints() output: a long-format CSV with one
    row per step of every checkpoint's rollout (columns: itr, target, step,
    x, y, velocity, reward).

    Args:
        csv_path: path to the checkpoint_trajectories_<itrs>.csv written by
            `python play.py <run_dir> --mode checkpoints`.

    Returns:
        DataFrame with 'target' parsed back into an (x, y) tuple.
    """
    df = pd.read_csv(csv_path)
    df['target'] = df['target'].apply(_parse_target)
    return df


def plot_trajectory_over_checkpoints(csv_path_or_df, save=True, save_dir='.'):
    """Plot how a navigate policy's trajectory changes over training, from
    play.py's --mode checkpoints CSV (one row per step of each (itr, target)
    rollout). One subplot per distinct target, each showing every
    checkpoint's trajectory colored by epoch -- so with more than one
    target, this also shows whether later checkpoints generalize to targets
    they weren't trained on.

    This only reads a CSV -- generate it first by running, in an
    environment with dm_control installed (e.g. WSL or the training server,
    not necessarily wherever this notebook's kernel runs):
        python play.py <run_dir> --mode checkpoints --itrs 10,50,100,200,300 \
            --targets='-3,0;0,2'

    Args:
        csv_path_or_df: path to that CSV, or an already-loaded DataFrame
            from load_checkpoint_trajectories().
        save: if True, save the figure as an SVG.
        save_dir: directory to save the figure in.

    Returns:
        (fig, axes, df)
    """
    df = (load_checkpoint_trajectories(csv_path_or_df)
          if isinstance(csv_path_or_df, str) else csv_path_or_df)

    itrs = sorted(df['itr'].unique())
    targets = sorted(df['target'].dropna().unique(), key=str) if df['target'].notna().any() else [None]
    cmap = plt.get_cmap('viridis')

    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 6), squeeze=False)
    axes = axes[0]
    for ax, target in zip(axes, targets):
        group = df[df['target'] == target] if target is not None else df
        for i, itr in enumerate(itrs):
            color = cmap(i / max(1, len(itrs) - 1))
            run = group[group['itr'] == itr]
            ax.plot(run['x'], run['y'], color=color, label=f'itr {itr}')

        if target is not None:
            ax.scatter(*target, color='red', marker='*', s=150, edgecolor='black', zorder=5, label='target')
        ax.scatter(0, 0, color='black', marker='o', s=40, zorder=5, label='start')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        # adjustable='datalim' (not the default 'box') keeps each subplot box a
        # normal shape and pads the data limits instead -- important here since
        # different targets can have very different x/y spans, which would
        # otherwise squash some subplots into thin strips and crowd their
        # title/legend into almost no space.
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(fontsize=8)
        ax.set_title(f'target {target}' if target is not None else 'trajectory')

    if isinstance(csv_path_or_df, str):
        run_name = os.path.basename(os.path.dirname(os.path.abspath(csv_path_or_df)))
    else:
        run_name = ''
    fig.suptitle(f'Trajectory over training — {run_name}')
    fig.tight_layout()

    if save:
        fig.savefig(os.path.join(save_dir, f'trajectory_over_checkpoints_{run_name}.svg'), dpi=300)

    return fig, axes, df


# --- Example usage -----------------------------------------------------
#
# play.py --mode sweep/checkpoints need dm_control, so run them in WSL/on
# the training server, not necessarily wherever this notebook's kernel runs.
# Each writes a CSV once; the plotting functions below just read it, so they
# can be re-run (in a plain notebook kernel, no dm_control needed) as often
# as you like without re-simulating anything.
#
# --- seed/target sweep (varies seed & target, fixed checkpoint) ---
#   python play.py <run_dir>/params.pkl --mode sweep
#
#   sweep_csv = os.path.join(run_dir, 'sweep_results.csv')
#   plot_sweep_trajectories(sweep_csv, save_dir=r'D:\ncap-swimmer')
#   plot_sweep_timeseries(sweep_csv, save_dir=r'D:\ncap-swimmer')
#   plot_sweep_final_distance(sweep_csv, save_dir=r'D:\ncap-swimmer')  # navigate only
#
# --- checkpoint comparison (varies checkpoint & target, fixed seed) ---
#   python play.py <run_dir> --mode checkpoints --itrs 10,50,150,200,300 \
#       --targets='-3,0;0,2'
#   # --targets is optional; omit it to just use each checkpoint's own
#   # trained-on target (no faceting, one plot).
#
#   checkpoint_csv = os.path.join(run_dir, 'checkpoint_trajectories_10-50-150-200-300.csv')
#   plot_trajectory_over_checkpoints(checkpoint_csv, save_dir=r'D:\ncap-swimmer')
#   # One subplot per distinct target in the CSV, each showing every
#   # checkpoint's trajectory colored by epoch -- so with more than one
#   # target this also shows whether later checkpoints generalize beyond
#   # whatever they were actually trained on.
#
# --- training progress (from progress.csv, no play.py needed) ---
#   run_info = extract_run_info(run_dir)
#   plot_training_progress(os.path.join(run_dir, 'progress.csv'), run_info=run_info,
#                           save_dir=r'D:\ncap-swimmer')
#   plot_population_returns(run_dir, save_dir=r'D:\ncap-swimmer')
#
# --- reward weight schedule explorer (no run data needed) ---
#   widgets.interact(
#       plot_reward_weight_schedule,
#       n_epochs=widgets.IntSlider(min=50, max=600, step=10, value=300),
#       speed_tau=widgets.IntSlider(min=5, max=300, step=5, value=40),
#       dir_peak_epoch=widgets.IntSlider(min=0, max=600, step=10, value=90),
#       dir_width=widgets.IntSlider(min=5, max=300, step=5, value=40),
#       dist_start_epoch=widgets.IntSlider(min=0, max=600, step=10, value=160),
#       dist_tau=widgets.IntSlider(min=5, max=300, step=5, value=60),
#       temperature=widgets.FloatSlider(min=0.1, max=2.0, step=0.1, value=0.5),
#   )
