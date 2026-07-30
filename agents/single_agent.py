from __future__ import annotations

import gymnasium as gym
import yaml

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from agents.callbacks import MultiSeedEvalCallback
from configs import load_config
from env.caching_env import CachingEnv
from env.wrappers import RandomTrafficSeedWrapper
from run_paths import RunPaths, resolve_run_name


def create_single_node_env(
    config: dict | None = None,
    seed: int = 42,
    *,
    randomize_traffic: bool = False,
) -> gym.Env:
    """Build the single-node caching environment, optionally with random traffic per episode."""
    if config is None:
        config = load_config()

    env = CachingEnv(config, seed=seed)
    if randomize_traffic:
        seed_range = int(config.get("episode_seed_range", 10_000))
        env = RandomTrafficSeedWrapper(env, base_seed=seed, seed_range=seed_range)
    return env


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


def create_single_node_dqn(
    config: dict | None = None,
    seed: int = 42,
    total_timesteps: int = 100_000,
    *,
    randomize_traffic: bool = True,
    tensorboard_log: str | None = None,
    verbose: int = 1,
) -> DQN:
    """Configure DQN from CachingEnv spaces and training hyperparameters in config."""
    if config is None:
        config = load_config()

    train_seed = int(config.get("train_seed", seed))
    set_random_seed(train_seed)

    env = Monitor(create_single_node_env(config, seed=train_seed, randomize_traffic=randomize_traffic))

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=config["learning_rate"],
        buffer_size=config["buffer_size"],
        learning_starts=int(config.get("learning_starts", 5000)),
        batch_size=int(config.get("batch_size", 128)),
        gamma=float(config.get("gamma", 0.99)),
        train_freq=4,
        gradient_steps=1,
        target_update_interval=config["target_update_interval"],
        exploration_initial_eps=config["epsilon_start"],
        exploration_final_eps=config["epsilon_end"],
        exploration_fraction=_exploration_fraction(config, total_timesteps),
        policy_kwargs={"net_arch": list(config["hidden_layers"])},
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        seed=train_seed,
    )
    return model


def train_single_node_dqn(
    total_timesteps: int = 100_000,
    config: dict | None = None,
    seed: int = 42,
    *,
    run_name: str | None = None,
    save_path: str | None = None,
    randomize_traffic: bool | None = None,
    tensorboard_log: str | None = None,
    eval_freq: int | None = None,
    n_eval_episodes: int | None = None,
    early_stopping: bool | None = None,
    verbose: int = 1,
) -> tuple[DQN, RunPaths]:
    """Create and train a single-node DQN agent into results/runs/<run_name>/."""
    if config is None:
        config = load_config()

    if save_path is not None and run_name is None:
        run_name = resolve_run_name(save_path)

    train_seed = int(config.get("train_seed", seed))
    if randomize_traffic is None:
        randomize_traffic = bool(config.get("randomize_episode_seeds", True))

    run_paths = RunPaths.create(run_name)
    run_paths.ensure_dirs()
    _snapshot_config(config, run_paths)

    if eval_freq is None:
        eval_freq = int(config.get("eval_freq", 20_000))
    if n_eval_episodes is None:
        n_eval_episodes = int(config.get("n_eval_episodes", 5))
    if early_stopping is None:
        early_stopping = bool(config.get("early_stopping", True))
    if tensorboard_log is None:
        tensorboard_log = str(run_paths.tensorboard)

    model = create_single_node_dqn(
        config=config,
        seed=train_seed,
        total_timesteps=total_timesteps,
        randomize_traffic=randomize_traffic,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )

    callbacks = []
    if eval_freq > 0:
        patience = (
            int(config.get("early_stopping_patience", 10))
            if early_stopping
            else None
        )
        min_evals = int(config.get("early_stopping_min_evals", 5))
        callbacks.append(
            MultiSeedEvalCallback(
                config,
                _parse_eval_seeds(config),
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                run_dir=str(run_paths.root),
                early_stopping_patience=patience,
                early_stopping_min_evals=min_evals,
                verbose=verbose,
            )
        )

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
        callback=callbacks or None,
    )

    model.save(str(run_paths.root / "model"))
    return model, run_paths
