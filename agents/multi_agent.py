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
from env.multi_agent_caching_env import MultiAgentCachingEnv, agent_id
from env.wrappers import RandomTrafficSeedWrapper
from run_paths import RunPaths, resolve_model_path, resolve_run_name


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
    exploration_initial_eps: float | None = None,
) -> DQN:
    """SB3 DQN with single-node spaces (shared across all nodes at Level 0)."""
    probe = MultiAgentCachingEnv(config, seed=seed)
    dummy = _NodeSpaceEnv(probe.observation_space, probe.action_space)
    set_random_seed(seed)
    if exploration_initial_eps is None:
        exploration_initial_eps = float(config["epsilon_start"])

    return DQN(
        policy="MlpPolicy",
        env=dummy,
        learning_rate=config["learning_rate"],
        buffer_size=config["buffer_size"],
        learning_starts=int(config.get("learning_starts", 5000)),
        batch_size=int(config.get("batch_size", 128)),
        gamma=float(config.get("gamma", 0.99)),
        train_freq=4,
        gradient_steps=1,
        target_update_interval=config["target_update_interval"],
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=config["epsilon_end"],
        exploration_fraction=_exploration_fraction(config, total_timesteps),
        policy_kwargs={"net_arch": list(config["hidden_layers"])},
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        seed=seed,
    )


def warm_start_from_pretrained(model: DQN, pretrained_path: str | Path) -> Path:
    """Copy Q-network weights from a single-node (or prior shared) SB3 DQN zip."""
    path = resolve_model_path(pretrained_path, prefer_best=True)
    if path is None:
        raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")

    pretrained = DQN.load(str(path))
    if model.observation_space.shape != pretrained.observation_space.shape:
        raise ValueError(
            f"Obs space mismatch: model {model.observation_space.shape} vs "
            f"pretrained {pretrained.observation_space.shape}"
        )
    if model.action_space.n != pretrained.action_space.n:
        raise ValueError(
            f"Action space mismatch: model {model.action_space.n} vs "
            f"pretrained {pretrained.action_space.n}"
        )

    model.policy.load_state_dict(pretrained.policy.state_dict())
    return path


def _select_action(model: DQN, obs: np.ndarray) -> int:
    """Epsilon-greedy action using SB3 DQN's exploration schedule."""
    if np.random.rand() < model.exploration_rate:
        return int(model.action_space.sample())
    action, _ = model.predict(obs, deterministic=True)
    return int(action)


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

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed if ep == 0 else None)
        total = 0.0
        while True:
            actions = {
                aid: int(model.predict(agent_obs, deterministic=True)[0])
                for aid, agent_obs in obs.items()
            }
            obs, rewards, terminateds, truncateds, _ = env.step(actions)
            total += float(sum(rewards.values()))
            if terminateds.get("__all__") or truncateds.get("__all__"):
                break
        episode_returns.append(total)

    stats = env.network_stats()
    return {
        "policy": "DQN",
        "ep_rew_mean": float(np.mean(episode_returns)),
        "ep_rew_std": float(np.std(episode_returns)),
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
    save_path: str | None = None,
    pretrained_path: str | Path | None = None,
    randomize_traffic: bool | None = None,
    tensorboard_log: str | None = None,
    eval_freq: int | None = None,
    n_eval_episodes: int | None = None,
    early_stopping: bool | None = None,
    verbose: int = 1,
) -> tuple[DQN, RunPaths]:
    """Train Level-0 shared-policy SB3 DQN on MultiAgentCachingEnv.

    Each env timestep: every node acts with the same DQN (parameter sharing),
    all transitions are written to one replay buffer. Deterministic multi-seed
    eval + early stopping mirror single-agent training.

    If ``pretrained_path`` is set (e.g. ``dqn_shifting``), copy that model's
    Q-network weights before fine-tuning. Exploration starts at 0.5 unless
    config overrides ``epsilon_start`` explicitly for the warm-start run.
    """
    if config is None:
        config = load_config()
    config = dict(config)

    if save_path is not None and run_name is None:
        run_name = resolve_run_name(save_path)
    if run_name is None:
        run_name = "dqn_multi_level0"

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
    if pretrained_path is not None:
        config["pretrained_path"] = str(pretrained_path)
    _snapshot_config(config, run_paths)
    if tensorboard_log is None:
        tensorboard_log = str(run_paths.tensorboard)

    # Softer exploration when fine-tuning a working single-node policy.
    exploration_initial_eps = float(config["epsilon_start"])
    if pretrained_path is not None and "warm_start_epsilon_start" in config:
        exploration_initial_eps = float(config["warm_start_epsilon_start"])
    elif pretrained_path is not None:
        exploration_initial_eps = min(exploration_initial_eps, 0.5)

    ma_env = create_multi_agent_env(
        config, seed=train_seed, randomize_traffic=randomize_traffic
    )
    model = create_shared_dqn(
        config,
        seed=train_seed,
        total_timesteps=total_timesteps,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        exploration_initial_eps=exploration_initial_eps,
    )

    eval_seeds = _parse_eval_seeds(config)
    best_mean = -float("inf")
    if pretrained_path is not None:
        loaded = warm_start_from_pretrained(model, pretrained_path)
        if verbose:
            print(f"Warm-started from {loaded}", flush=True)
            zero_shot, per_seed = _multi_seed_eval(
                model, config, eval_seeds, n_eval_episodes
            )
            seed_summary = "  ".join(f"seed={s}:{per_seed[s]:.1f}" for s in eval_seeds)
            print(
                f"Zero-shot eval mean_reward={zero_shot:.2f}  ({seed_summary})",
                flush=True,
            )
            best_mean = zero_shot
            model.save(str(run_paths.root / "best_model"))

    # SB3 progress / exploration bookkeeping
    model._setup_learn(total_timesteps, callback=None, reset_num_timesteps=True, tb_log_name="DQN")

    obs_dict, _ = ma_env.reset(seed=train_seed)
    learning_starts = int(config.get("learning_starts", 5000))
    train_freq = 4
    patience = int(config.get("early_stopping_patience", 10)) if early_stopping else None
    min_evals = int(config.get("early_stopping_min_evals", 5))
    no_improve = 0
    n_evals = 0
    evaluations_timesteps: list[int] = []
    evaluations_means: list[list[float]] = []

    if verbose:
        print(
            f"Training shared-policy SB3 DQN: nodes={config['num_nodes']} "
            f"traffic={config.get('traffic_pattern')} timesteps={total_timesteps} "
            f"eps_start={exploration_initial_eps}",
            flush=True,
        )

    while model.num_timesteps < total_timesteps:
        model._update_current_progress_remaining(model.num_timesteps, total_timesteps)
        model._on_step()

        actions: dict[str, int] = {}
        for aid, obs in obs_dict.items():
            actions[aid] = _select_action(model, obs)

        next_obs_dict, rewards, terminateds, truncateds, _ = ma_env.step(actions)
        terminated = bool(terminateds.get("__all__", False))
        truncated = bool(truncateds.get("__all__", False))
        episode_done = terminated or truncated
        # Mark timeouts so SB3 still bootstraps on truncated episode ends.
        transition_info = {"TimeLimit.truncated": truncated} if truncated else {}

        for aid, obs in obs_dict.items():
            model.replay_buffer.add(
                obs,
                next_obs_dict[aid],
                np.array([actions[aid]]),
                np.array([rewards[aid]], dtype=np.float32),
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


def evaluate_random_policy(
    config: dict | None = None,
    *,
    num_episodes: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
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


def reactive_multi_baseline_step(
    env: MultiAgentCachingEnv,
    policies: dict[int, Any],
    obs: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, float], bool]:
    """Score requests first, then update each node's cache (oracle baselines)."""
    requests = env.request_generator.generate()
    rewards: dict[str, float] = {}
    actions: dict[str, int] = {}

    for node_id in range(env.num_nodes):
        requested = requests[node_id]
        reward = env._process_request(node_id, requested)
        if requested is not None:
            env.network.nodes[node_id].record_request(
                requested, env.observation_window
            )
        aid = agent_id(node_id)
        rewards[aid] = reward
        node = env.network.nodes[node_id]
        actions[aid] = policies[node_id].act(
            obs[aid], requested, cache=node.cache
        )

    for aid, action in actions.items():
        env._apply_action(int(aid), action)

    env.timestep += 1
    truncated = env.timestep >= env.episode_length
    return env._get_observations(), rewards, truncated
