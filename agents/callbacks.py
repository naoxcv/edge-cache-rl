from __future__ import annotations

import os

import numpy as np
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from env.caching_env import CachingEnv


def _eval_env(config: dict, seed: int):
    return Monitor(CachingEnv(config, seed=seed))


class MultiSeedEvalCallback(EvalCallback):
    """EvalCallback that averages deterministic returns across multiple traffic seeds."""

    def __init__(
        self,
        config: dict,
        eval_seeds: list[int],
        *,
        eval_freq: int,
        n_eval_episodes: int,
        run_dir: str,
        early_stopping_patience: int | None = None,
        early_stopping_min_evals: int = 0,
        verbose: int = 1,
    ) -> None:
        primary_env = _eval_env(config, eval_seeds[0])
        stop_callback = None
        if early_stopping_patience is not None:
            stop_callback = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=early_stopping_patience,
                min_evals=early_stopping_min_evals,
                verbose=verbose,
            )

        super().__init__(
            primary_env,
            best_model_save_path=run_dir,
            log_path=run_dir,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False,
            verbose=verbose,
            callback_after_eval=stop_callback,
        )
        self.config = config
        self.eval_seeds = list(eval_seeds)

    def _evaluate(self) -> tuple[list[float], list[int], dict[int, float]]:
        per_seed_means: dict[int, float] = {}
        all_rewards: list[float] = []
        all_lengths: list[int] = []

        for seed in self.eval_seeds:
            eval_env = _eval_env(self.config, seed)
            vec_env = DummyVecEnv([lambda env=eval_env: env])
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                vec_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                render=self.render,
                return_episode_rewards=True,
                warn=self.warn,
            )
            per_seed_means[seed] = float(np.mean(episode_rewards))
            all_rewards.extend(episode_rewards)
            all_lengths.extend(episode_lengths)
            vec_env.close()

        return all_rewards, all_lengths, per_seed_means

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._is_success_buffer = []
            all_rewards, all_lengths, per_seed_means = self._evaluate()
            mean_reward = float(np.mean(list(per_seed_means.values())))
            std_reward = float(np.std(list(per_seed_means.values())))
            mean_ep_length = float(np.mean(all_lengths))
            std_ep_length = float(np.std(all_lengths))
            self.last_mean_reward = mean_reward

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(all_rewards)
                self.evaluations_length.append(all_lengths)
                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    eval_seeds=np.array(self.eval_seeds),
                    per_seed_means=np.array(
                        [per_seed_means[s] for s in self.eval_seeds]
                    ),
                )

            if self.verbose >= 1:
                seed_summary = "  ".join(
                    f"seed={s}:{per_seed_means[s]:.1f}" for s in self.eval_seeds
                )
                print(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}  ({seed_summary})"
                )
                print(
                    f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}"
                )

            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            for seed, seed_mean in per_seed_means.items():
                self.logger.record(f"eval/seed_{seed}_mean_reward", seed_mean)

            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training
