from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.utils import set_random_seed

from configs import load_config
from env.multi_agent_caching_env import (
    MultiAgentCachingEnv,
    agent_id,
    local_obs_size,
    needs_decision_from_obs,
)
from env.wrappers import RandomTrafficSeedWrapper
from run_paths import RunPaths, resolve_model_path


def create_multi_agent_env(
    config: dict | None = None,
    seed: int = 42,
    *,
    randomize_traffic: bool = False,
) -> gym.Env:
    """Build MultiAgentCachingEnv, optionally randomizing traffic seed per episode."""
    if config is None:
        config = load_config()
    env: gym.Env = MultiAgentCachingEnv(config, seed=seed)
    if randomize_traffic:
        seed_range = int(config.get("episode_seed_range", 10_000))
        env = RandomTrafficSeedWrapper(env, base_seed=seed, seed_range=seed_range)
    return env


class _NodeSpaceEnv(gym.Env):
    """Dummy env matching one node's spaces so SB3 can construct a DQN."""

    metadata = {"render_modes": []}

    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        obs = self.observation_space.sample()
        return obs, 0.0, False, True, {}


def _exploration_fraction(config: dict, total_timesteps: int) -> float:
    if "exploration_fraction" in config:
        return float(config["exploration_fraction"])
    decay_steps = config["epsilon_decay_steps"]
    return min(1.0, decay_steps / total_timesteps)


def _gradient_steps(config: dict) -> int:
    """Gradient steps per training call.

    The shared-policy loop writes one transition per node per env step but trains
    every 4th env step, so a single gradient step would see 1 update per
    ``4 * num_nodes`` transitions. Defaulting to ``num_nodes`` restores the
    single-agent update-to-data ratio, which from-scratch training needs.
    """
    if "gradient_steps" in config:
        return int(config["gradient_steps"])
    return max(1, int(config["num_nodes"]))


def _parse_eval_seeds(config: dict) -> list[int]:
    seeds = config.get("eval_seeds", [int(config.get("train_seed", 42))])
    return [int(s) for s in seeds]


def _snapshot_config(config: dict, run_paths: RunPaths) -> None:
    run_paths.ensure_dirs()
    with run_paths.config_snapshot.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def create_shared_dqn(
    config: dict,
    *,
    seed: int,
    total_timesteps: int,
    tensorboard_log: str | None,
    verbose: int,
) -> DQN:
    """SB3 DQN with per-node spaces (shared across all nodes; obs width follows comm_level)."""
    probe = MultiAgentCachingEnv(config, seed=seed)
    dummy = _NodeSpaceEnv(probe.observation_space, probe.action_space)
    set_random_seed(seed)

    return DQN(
        policy="MlpPolicy",
        env=dummy,
        learning_rate=config["learning_rate"],
        buffer_size=config["buffer_size"],
        learning_starts=int(config.get("learning_starts", 5000)),
        batch_size=int(config.get("batch_size", 128)),
        gamma=float(config.get("gamma", 0.99)),
        train_freq=int(config.get("train_freq", 4)),
        gradient_steps=_gradient_steps(config),
        target_update_interval=config["target_update_interval"],
        exploration_initial_eps=float(config["epsilon_start"]),
        exploration_final_eps=config["epsilon_end"],
        exploration_fraction=_exploration_fraction(config, total_timesteps),
        policy_kwargs={"net_arch": list(config["hidden_layers"])},
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        seed=seed,
    )


def _select_action(model: DQN, obs: np.ndarray) -> int:
    """Epsilon-greedy action using SB3 DQN's exploration schedule."""
    if np.random.rand() < model.exploration_rate:
        return int(model.action_space.sample())
    action, _ = model.predict(obs, deterministic=True)
    return int(action)


def _q_values(model: DQN, obs: np.ndarray):
    import torch

    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    with torch.no_grad():
        q = model.policy.q_net(obs_tensor)[0]
    return q


def _q_margin(model: DQN, obs: np.ndarray) -> float:
    import torch

    q = _q_values(model, obs)
    if q.numel() < 2:
        return float("inf")
    top2 = torch.topk(q, k=2).values
    return float(top2[0] - top2[1])


def _local_dim(config: dict) -> int:
    return local_obs_size(
        int(config["num_container_types"]), int(config["cache_capacity"])
    )


def select_action_for_obs(
    model: DQN,
    obs: np.ndarray,
    *,
    config: dict,
    deterministic: bool = False,
) -> tuple[int, bool]:
    """Choose an eviction slot or reject; for Level 3, gate neighbor features.

    Hits and non-full misses need no decision: return reject (C) without
    communicating. Returns ``(action, communicated)``.
    """
    reject = int(config["cache_capacity"])
    comm_level = int(config.get("comm_level", 0))
    local_dim = _local_dim(config)
    if obs.shape[-1] >= local_dim and not needs_decision_from_obs(obs[:local_dim]):
        return reject, False

    communicated = False
    used = obs

    if comm_level == 3 and obs.shape[-1] > local_dim:
        local = np.array(obs, copy=True)
        local[local_dim:] = 0.0
        margin = _q_margin(model, local)
        threshold = float(config.get("selective_comm_threshold", 0.1))
        communicated = margin < threshold
        used = obs if communicated else local

    if deterministic:
        action, _ = model.predict(used, deterministic=True)
        return int(action), communicated
    if np.random.rand() < model.exploration_rate:
        return int(model.action_space.sample()), communicated
    action, _ = model.predict(used, deterministic=True)
    return int(action), communicated


def evaluate_shared_dqn(
    model: DQN,
    config: dict,
    *,
    num_episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Deterministic shared-policy rollout on MultiAgentCachingEnv."""
    env = MultiAgentCachingEnv(config, seed=seed)
    episode_returns: list[float] = []
    episode_task: list[float] = []
    episode_pen: list[float] = []
    episode_comm: list[float] = []
    episode_comm_events: list[int] = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed if ep == 0 else None)
        total = 0.0
        task = 0.0
        pen = 0.0
        comm_pen = 0.0
        while True:
            actions: dict[str, int] = {}
            for aid, agent_obs in obs.items():
                action, communicated = select_action_for_obs(
                    model, agent_obs, config=config, deterministic=True
                )
                actions[aid] = action
                if communicated:
                    cpen = env.record_communication(int(aid), True)
                    comm_pen += cpen

            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            for aid, r in rewards.items():
                total += float(r)
                task += float(infos[aid].get("task_reward", r))
                pen += float(infos[aid].get("overlap_penalty", 0.0))
            if terminateds.get("__all__") or truncateds.get("__all__"):
                break
        total += comm_pen
        episode_returns.append(total)
        episode_task.append(task)
        episode_pen.append(pen)
        episode_comm.append(comm_pen)
        episode_comm_events.append(int(env._episode_comm_events))

    stats = env.network_stats()
    return {
        "policy": "DQN",
        "ep_rew_mean": float(np.mean(episode_returns)),
        "ep_rew_std": float(np.std(episode_returns)),
        "task_return_mean": float(np.mean(episode_task)),
        "overlap_penalty_mean": float(np.mean(episode_pen)),
        "comm_penalty_mean": float(np.mean(episode_comm)),
        "comm_events_mean": float(np.mean(episode_comm_events)),
        "num_episodes": len(episode_returns),
        **stats,
    }


def _multi_seed_eval(
    model: DQN,
    config: dict,
    eval_seeds: list[int],
    n_eval_episodes: int,
) -> tuple[float, dict[int, float]]:
    per_seed: dict[int, float] = {}
    for seed in eval_seeds:
        result = evaluate_shared_dqn(
            model, config, num_episodes=n_eval_episodes, seed=seed
        )
        per_seed[seed] = result["ep_rew_mean"]
    return float(np.mean(list(per_seed.values()))), per_seed


def train_multi_agent_dqn(
    total_timesteps: int = 100_000,
    config: dict | None = None,
    seed: int = 42,
    *,
    run_name: str | None = None,
    randomize_traffic: bool | None = None,
    tensorboard_log: str | None = None,
    eval_freq: int | None = None,
    n_eval_episodes: int | None = None,
    early_stopping: bool | None = None,
    verbose: int = 1,
) -> tuple[DQN, RunPaths]:
    """Train shared-policy SB3 DQN on MultiAgentCachingEnv (comm levels 0-3).

    Each env timestep: every node acts with the same DQN (parameter sharing),
    all transitions are written to one replay buffer. Deterministic multi-seed
    eval + early stopping mirror single-agent training.

    Level 3: Q-margin selective communication -- neighbor features are used only
    when the local-only Q-margin is below ``selective_comm_threshold``; each
    communication event costs ``comm_penalty_lambda``.
    """
    if config is None:
        config = load_config()
    config = dict(config)

    if run_name is None:
        run_name = f"dqn_multi_level{int(config.get('comm_level', 0))}"

    train_seed = int(config.get("train_seed", seed))
    if randomize_traffic is None:
        randomize_traffic = bool(config.get("randomize_episode_seeds", True))
    if eval_freq is None:
        eval_freq = int(config.get("eval_freq", 20_000))
    if n_eval_episodes is None:
        n_eval_episodes = int(config.get("n_eval_episodes", 5))
    if early_stopping is None:
        early_stopping = bool(config.get("early_stopping", True))

    run_paths = RunPaths.create(run_name)
    run_paths.ensure_dirs()
    _snapshot_config(config, run_paths)
    if tensorboard_log is None:
        tensorboard_log = str(run_paths.tensorboard)

    ma_env = create_multi_agent_env(
        config, seed=train_seed, randomize_traffic=randomize_traffic
    )
    caching_env = ma_env.env if isinstance(ma_env, RandomTrafficSeedWrapper) else ma_env
    model = create_shared_dqn(
        config,
        seed=train_seed,
        total_timesteps=total_timesteps,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )

    eval_seeds = _parse_eval_seeds(config)
    best_mean = -float("inf")
    comm_level = int(config.get("comm_level", 0))

    model._setup_learn(total_timesteps, callback=None, reset_num_timesteps=True, tb_log_name="DQN")

    obs_dict, _ = ma_env.reset(seed=train_seed)
    learning_starts = int(config.get("learning_starts", 5000))
    train_freq = int(config.get("train_freq", 4))
    patience = int(config.get("early_stopping_patience", 10)) if early_stopping else None
    min_evals = int(config.get("early_stopping_min_evals", 5))
    no_improve = 0
    n_evals = 0
    evaluations_timesteps: list[int] = []
    evaluations_means: list[list[float]] = []
    local_dim = _local_dim(config)

    if verbose:
        print(
            f"Training shared-policy SB3 DQN: nodes={config['num_nodes']} "
            f"comm_level={comm_level} traffic={config.get('traffic_pattern')} "
            f"timesteps={total_timesteps} eps_start={config['epsilon_start']} "
            f"obs_dim={model.observation_space.shape[0]} "
            f"n_actions={model.action_space.n} "
            f"gradient_steps={model.gradient_steps}",
            flush=True,
        )

    while model.num_timesteps < total_timesteps:
        model._update_current_progress_remaining(model.num_timesteps, total_timesteps)
        model._on_step()

        actions: dict[str, int] = {}
        used_obs: dict[str, np.ndarray] = {}
        comm_pens: dict[str, float] = {}
        reject = int(config["cache_capacity"])
        for aid, obs in obs_dict.items():
            local = obs[:local_dim] if obs.shape[-1] >= local_dim else obs
            if not needs_decision_from_obs(local):
                used_obs[aid] = obs
                actions[aid] = reject
                comm_pens[aid] = 0.0
                continue
            if comm_level == 3:
                if np.random.rand() < model.exploration_rate:
                    used = obs
                    action = int(model.action_space.sample())
                    communicated = True
                else:
                    action, communicated = select_action_for_obs(
                        model, obs, config=config, deterministic=True
                    )
                    if not communicated:
                        used = np.array(obs, copy=True)
                        used[local_dim:] = 0.0
                    else:
                        used = obs
                cpen = (
                    caching_env.record_communication(int(aid), True)
                    if communicated
                    else 0.0
                )
                used_obs[aid] = used
                actions[aid] = action
                comm_pens[aid] = cpen
            else:
                used_obs[aid] = obs
                actions[aid] = _select_action(model, obs)
                comm_pens[aid] = 0.0

        next_obs_dict, rewards, terminateds, truncateds, _ = ma_env.step(actions)
        terminated = bool(terminateds.get("__all__", False))
        truncated = bool(truncateds.get("__all__", False))
        episode_done = terminated or truncated
        transition_info = {"TimeLimit.truncated": truncated} if truncated else {}

        for aid, obs in used_obs.items():
            next_obs = next_obs_dict[aid]
            if comm_level == 3 and next_obs.shape[-1] > local_dim:
                next_local = np.array(next_obs, copy=True)
                next_local[local_dim:] = 0.0
                next_margin = _q_margin(model, next_local)
                threshold = float(config.get("selective_comm_threshold", 0.1))
                if next_margin >= threshold:
                    next_obs = next_local
            r = float(rewards[aid]) + float(comm_pens[aid])
            model.replay_buffer.add(
                obs,
                next_obs,
                np.array([actions[aid]]),
                np.array([r], dtype=np.float32),
                np.array([episode_done]),
                [transition_info],
            )

        model.num_timesteps += 1
        obs_dict = next_obs_dict

        if episode_done:
            obs_dict, _ = ma_env.reset()

        if (
            model.num_timesteps > learning_starts
            and model.num_timesteps % train_freq == 0
        ):
            model.train(batch_size=model.batch_size, gradient_steps=model.gradient_steps)

        if eval_freq > 0 and model.num_timesteps % eval_freq == 0:
            mean_reward, per_seed = _multi_seed_eval(
                model, config, eval_seeds, n_eval_episodes
            )
            n_evals += 1
            evaluations_timesteps.append(model.num_timesteps)
            evaluations_means.append([per_seed[s] for s in eval_seeds])
            np.savez(
                run_paths.root / "evaluations.npz",
                timesteps=evaluations_timesteps,
                per_seed_means=evaluations_means,
                eval_seeds=np.array(eval_seeds),
            )
            if verbose:
                seed_summary = "  ".join(f"seed={s}:{per_seed[s]:.1f}" for s in eval_seeds)
                print(
                    f"Eval num_timesteps={model.num_timesteps}, "
                    f"mean_reward={mean_reward:.2f}  ({seed_summary})",
                    flush=True,
                )

            if mean_reward > best_mean:
                best_mean = mean_reward
                no_improve = 0
                model.save(str(run_paths.root / "best_model"))
                if verbose:
                    print(f"  New best mean reward! ({best_mean:.2f})", flush=True)
            else:
                no_improve += 1

            if (
                patience is not None
                and n_evals > min_evals
                and no_improve > patience
            ):
                if verbose:
                    print(
                        f"Early stopping: no improvement for {no_improve} evals",
                        flush=True,
                    )
                break

        if verbose and model.num_timesteps % 10_000 == 0:
            print(
                f"  timesteps={model.num_timesteps} "
                f"exploration={model.exploration_rate:.3f} "
                f"buffer={model.replay_buffer.size()}",
                flush=True,
            )

    model.save(str(run_paths.root / "model"))
    if model.logger is not None:
        model.logger.dump(model.num_timesteps)
    return model, run_paths


def create_independent_dqns(
    config: dict,
    *,
    seed: int,
    n_agents: int,
    total_timesteps: int,
    tensorboard_log: str | None,
    verbose: int,
) -> dict[str, DQN]:
    """Create one independent SB3 DQN per agent id.

    Each DQN is trained on transitions for exactly one node (IDQN baseline).
    """
    probe = MultiAgentCachingEnv(config, seed=seed)
    dummy = _NodeSpaceEnv(probe.observation_space, probe.action_space)

    models: dict[str, DQN] = {}
    for node_id in range(n_agents):
        agent_seed = int(seed) + int(node_id)
        set_random_seed(agent_seed)
        models[agent_id(node_id)] = DQN(
            policy="MlpPolicy",
            env=dummy,
            learning_rate=config["learning_rate"],
            buffer_size=config["buffer_size"],
            learning_starts=int(config.get("learning_starts", 5000)),
            batch_size=int(config.get("batch_size", 128)),
            gamma=float(config.get("gamma", 0.99)),
            train_freq=int(config.get("train_freq", 4)),
            # Match shared-policy update-to-data: each model sees one transition per
            # env step, so use num_nodes gradient steps (same default as shared DQN).
            gradient_steps=int(
                config.get(
                    "idqn_gradient_steps", max(1, int(config["num_nodes"]))
                )
            ),
            target_update_interval=config["target_update_interval"],
            exploration_initial_eps=float(config["epsilon_start"]),
            exploration_final_eps=float(config["epsilon_end"]),
            exploration_fraction=_exploration_fraction(config, total_timesteps),
            policy_kwargs={"net_arch": list(config["hidden_layers"])},
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=agent_seed,
        )
    return models


def evaluate_independent_dqn(
    models: dict[str, DQN],
    config: dict,
    *,
    num_episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Deterministic IDQN rollout on MultiAgentCachingEnv."""
    env = MultiAgentCachingEnv(config, seed=seed)
    episode_returns: list[float] = []
    episode_task: list[float] = []
    episode_pen: list[float] = []
    episode_comm_events: list[int] = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed if ep == 0 else None)
        total = 0.0
        task = 0.0
        pen = 0.0
        comm_pen = 0.0
        while True:
            actions: dict[str, int] = {}
            for aid, agent_obs in obs.items():
                model = models[aid]
                action, communicated = select_action_for_obs(
                    model, agent_obs, config=config, deterministic=True
                )
                actions[aid] = action
                if communicated:
                    cpen = env.record_communication(int(aid), True)
                    comm_pen += cpen

            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            for aid, r in rewards.items():
                total += float(r)
                task += float(infos[aid].get("task_reward", r))
                pen += float(infos[aid].get("overlap_penalty", 0.0))
            if terminateds.get("__all__") or truncateds.get("__all__"):
                break

        total += comm_pen
        episode_returns.append(total)
        episode_task.append(task)
        episode_pen.append(pen)
        episode_comm_events.append(int(env._episode_comm_events))

    stats = env.network_stats()
    return {
        "policy": "IDQN",
        "ep_rew_mean": float(np.mean(episode_returns)),
        "ep_rew_std": float(np.std(episode_returns)),
        "task_return_mean": float(np.mean(episode_task)),
        "overlap_penalty_mean": float(np.mean(episode_pen)),
        "comm_events_mean": float(np.mean(episode_comm_events)),
        "num_episodes": len(episode_returns),
        **stats,
    }


def train_idqn_agent_dqn(
    total_timesteps: int = 100_000,
    config: dict | None = None,
    seed: int = 42,
    *,
    run_name: str | None = None,
    randomize_traffic: bool | None = None,
    tensorboard_log: str | None = None,
    eval_freq: int | None = None,
    n_eval_episodes: int | None = None,
    early_stopping: bool | None = None,
    verbose: int = 1,
) -> tuple[dict[str, DQN], RunPaths]:
    """Train independent SB3 DQN per node (IDQN baseline).

    Compared to train_multi_agent_dqn(), there is no parameter sharing. Each node
    trains its own DQN on its own transitions only.
    """
    if config is None:
        config = load_config()
    config = dict(config)

    if run_name is None:
        level = int(config.get("comm_level", 0))
        run_name = f"dqn_idqn_level{level}"

    train_seed = int(config.get("train_seed", seed))
    if randomize_traffic is None:
        randomize_traffic = bool(config.get("randomize_episode_seeds", True))
    if eval_freq is None:
        eval_freq = int(config.get("eval_freq", 20_000))
    if n_eval_episodes is None:
        n_eval_episodes = int(config.get("n_eval_episodes", 5))
    if early_stopping is None:
        early_stopping = bool(config.get("early_stopping", True))

    run_paths = RunPaths.create(run_name)
    run_paths.ensure_dirs()
    _snapshot_config(config, run_paths)
    if tensorboard_log is None:
        tensorboard_log = str(run_paths.tensorboard)

    ma_env = create_multi_agent_env(
        config, seed=train_seed, randomize_traffic=randomize_traffic
    )
    caching_env = ma_env.env if isinstance(ma_env, RandomTrafficSeedWrapper) else ma_env

    comm_level = int(config.get("comm_level", 0))
    models = create_independent_dqns(
        config,
        seed=train_seed,
        n_agents=int(config["num_nodes"]),
        total_timesteps=total_timesteps,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )

    eval_seeds = _parse_eval_seeds(config)
    best_mean = -float("inf")
    local_dim = _local_dim(config)

    # SB3 internal setup for manual training (so each model owns its replay buffer).
    for m in models.values():
        m._setup_learn(
            total_timesteps,
            callback=None,
            reset_num_timesteps=True,
            tb_log_name="DQN_IDQN",
        )

    obs_dict, _ = ma_env.reset(seed=train_seed)
    learning_starts = int(config.get("learning_starts", 5000))
    train_freq = int(config.get("train_freq", 4))
    patience = int(config.get("early_stopping_patience", 10)) if early_stopping else None
    min_evals = int(config.get("early_stopping_min_evals", 5))
    no_improve = 0
    n_evals = 0

    if verbose:
        print(
            f"Training IDQN: nodes={config['num_nodes']} comm_level={comm_level} "
            f"traffic={config.get('traffic_pattern')} timesteps={total_timesteps} "
            f"eps_start={config['epsilon_start']} obs_dim={next(iter(models.values())).observation_space.shape[0]}",
            flush=True,
        )

    while next(iter(models.values())).num_timesteps < total_timesteps:
        # Use the first model's timestep as the shared "env clock".
        env_t = next(iter(models.values())).num_timesteps
        for m in models.values():
            m._update_current_progress_remaining(m.num_timesteps, total_timesteps)
            m._on_step()

        actions: dict[str, int] = {}
        used_obs: dict[str, np.ndarray] = {}
        comm_pens: dict[str, float] = {}
        reject = int(config["cache_capacity"])

        for aid, obs in obs_dict.items():
            model = models[aid]
            local = obs[:local_dim] if obs.shape[-1] >= local_dim else obs
            if not needs_decision_from_obs(local):
                used_obs[aid] = obs
                actions[aid] = reject
                comm_pens[aid] = 0.0
                continue
            if comm_level == 3:
                action, communicated = select_action_for_obs(
                    model, obs, config=config, deterministic=False
                )
                used = obs
                if not communicated and obs.shape[-1] > local_dim:
                    used = np.array(obs, copy=True)
                    used[local_dim:] = 0.0
                cpen = (
                    caching_env.record_communication(int(aid), True)
                    if communicated
                    else 0.0
                )
                used_obs[aid] = used
                actions[aid] = action
                comm_pens[aid] = cpen
            else:
                # Non-selective levels: independent DQN, no comm gating.
                action = _select_action(model, obs)
                used_obs[aid] = obs
                actions[aid] = action
                comm_pens[aid] = 0.0

        next_obs_dict, rewards, terminateds, truncateds, _ = ma_env.step(actions)
        terminated = bool(terminateds.get("__all__", False))
        truncated = bool(truncateds.get("__all__", False))
        episode_done = terminated or truncated
        transition_info = {"TimeLimit.truncated": truncated} if truncated else {}

        # Write only this node's transition into its own replay buffer.
        for aid, obs in used_obs.items():
            model = models[aid]
            next_obs = next_obs_dict[aid]

            # Level-3 next-obs gating to match shared-policy logic.
            if comm_level == 3 and next_obs.shape[-1] > local_dim:
                next_local = np.array(next_obs, copy=True)
                next_local[local_dim:] = 0.0
                next_margin = _q_margin(model, next_local)
                threshold = float(config.get("selective_comm_threshold", 0.1))
                if next_margin >= threshold:
                    next_obs = next_local

            r = float(rewards[aid]) + float(comm_pens[aid])
            model.replay_buffer.add(
                obs,
                next_obs,
                np.array([actions[aid]]),
                np.array([r], dtype=np.float32),
                np.array([episode_done]),
                [transition_info],
            )

        # Advance all models' clocks equally.
        for m in models.values():
            m.num_timesteps += 1

        obs_dict = next_obs_dict
        if episode_done:
            obs_dict, _ = ma_env.reset()

        # Train each model from its own replay.
        if (
            next(iter(models.values())).num_timesteps > learning_starts
            and next(iter(models.values())).num_timesteps % train_freq == 0
        ):
            for m in models.values():
                m.train(
                    batch_size=m.batch_size, gradient_steps=m.gradient_steps
                )

        # Periodic eval + early stopping using the set of models.
        if eval_freq > 0 and next(iter(models.values())).num_timesteps % eval_freq == 0:
            # Evaluate using all nodes' deterministic policies.
            per_seed: dict[int, float] = {}
            for seed_i in eval_seeds:
                result = evaluate_independent_dqn(
                    models,
                    config,
                    num_episodes=n_eval_episodes,
                    seed=seed_i,
                )
                per_seed[seed_i] = float(result["ep_rew_mean"])

            mean_reward = float(np.mean(list(per_seed.values())))
            n_evals += 1
            if verbose:
                seed_summary = "  ".join(f"seed={s}:{per_seed[s]:.1f}" for s in eval_seeds)
                print(
                    f"Eval IDQN timesteps={next(iter(models.values())).num_timesteps}, "
                    f"mean_reward={mean_reward:.2f} ({seed_summary})",
                    flush=True,
                )

            if mean_reward > best_mean:
                best_mean = mean_reward
                no_improve = 0
                # Save each node's best model.
                for nid, m in models.items():
                    m.save(str(run_paths.root / f"best_model_node{nid}"))
                if verbose:
                    print(f"  New best mean reward! ({best_mean:.2f})", flush=True)
            else:
                no_improve += 1

            if (
                patience is not None
                and n_evals > min_evals
                and no_improve > patience
            ):
                if verbose:
                    print(
                        f"Early stopping (IDQN): no improvement for {no_improve} evals",
                        flush=True,
                    )
                break

    # Save final models too.
    for nid, m in models.items():
        m.save(str(run_paths.root / f"model_node{nid}"))

    # For compatibility with other scripts, also write a marker file.
    (run_paths.root / "idqn_trained.txt").write_text(
        "IDQN trained. Use best_model_node{node} checkpoints.\n"
    )
    return models, run_paths


def evaluate_random_policy(
    config: dict | None = None,
    *,
    num_episodes: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    """Run uniform-random actions on MultiAgentCachingEnv and return aggregate stats."""
    if config is None:
        config = load_config()

    env = MultiAgentCachingEnv(config, seed=seed)
    rng = np.random.default_rng(seed)
    episode_returns: list[float] = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        total = 0.0
        while True:
            actions = {
                aid: int(rng.integers(0, env.action_space.n)) for aid in obs
            }
            obs, rewards, terminateds, truncateds, _ = env.step(actions)
            total += float(sum(rewards.values()))
            if terminateds.get("__all__") or truncateds.get("__all__"):
                break
        episode_returns.append(total)

    stats = env.network_stats()
    stats["ep_rew_mean"] = float(np.mean(episode_returns))
    stats["ep_rew_std"] = float(np.std(episode_returns))
    return stats


def evaluate_sb3_dqn(
    model_path: str | Path,
    config: dict,
    *,
    num_episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Load an SB3 zip and evaluate shared-policy DQN."""
    model = DQN.load(str(model_path))
    result = evaluate_shared_dqn(
        model, config, num_episodes=num_episodes, seed=seed
    )
    result["checkpoint"] = str(model_path)
    return result


def resolve_multi_model_path(path_arg: str, *, prefer_best: bool = True) -> Path | None:
    """Resolve SB3 best_model.zip / model.zip for a multi-agent run."""
    return resolve_model_path(path_arg, prefer_best=prefer_best)


def heuristic_multi_baseline_step(
    env: MultiAgentCachingEnv,
    policies: dict[int, Any],
    obs: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, float], bool]:
    """Request-first LRU/LFU step matching ``MultiAgentCachingEnv.step``.

    Policies see the pending request (in obs / ``env._pending_requests``) and
    choose an eviction slot or reject. Hits and non-full misses are handled by
    the env; the returned action is ignored in those cases.
    """
    actions: dict[str, int] = {}
    for node_id in range(env.num_nodes):
        aid = agent_id(node_id)
        node = env.network.nodes[node_id]
        actions[aid] = policies[node_id].act(
            obs[aid],
            env._pending_requests.get(node_id),
            cache=node.cache,
            cache_capacity=env.cache_capacity,
            num_container_types=env.num_container_types,
        )
    obs, rewards, _terminateds, truncateds, _infos = env.step(actions)
    return obs, rewards, bool(truncateds.get("__all__", False))


def reactive_multi_baseline_step(
    env: MultiAgentCachingEnv,
    policies: dict[int, Any],
    obs: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, float], bool]:
    """Alias for ``heuristic_multi_baseline_step`` (request-first eviction MDP)."""
    return heuristic_multi_baseline_step(env, policies, obs)


def blind_multi_baseline_step(
    env: MultiAgentCachingEnv,
    policies: dict[int, Any],
    obs: dict[str, np.ndarray],
    last_requests: dict[int, int | None],
) -> tuple[dict[str, np.ndarray], dict[str, float], bool, dict[int, int | None]]:
    """Compatibility wrapper; eviction-only env is already request-first."""
    obs, rewards, truncated = heuristic_multi_baseline_step(env, policies, obs)
    for node_id in range(env.num_nodes):
        last_requests[node_id] = env._pending_requests.get(node_id)
    return obs, rewards, truncated, last_requests
