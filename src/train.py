# train.py
# This script trains a PPO model on the CartPole-v1 environment and logs the training reward.
from __future__ import annotations

import argparse
import os
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from common import (
    TrainConfig,
    create_run_dir,
    ensure_dirs,
    load_config_from_file,
    merge_config_with_args,
    save_config,
    save_metrics,
    set_global_seed,
)


class RewardLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self._current = 0.0

    def _on_step(self) -> bool:
        # Vectorized env returns arrays
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        if rewards is None or dones is None:
            return True

        r = float(rewards[0])
        d = bool(dones[0])
        self._current += r
        if d:
            self.episode_rewards.append(self._current)
            self._current = 0.0
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file (JSON or YAML)")
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    args = parser.parse_args()

    ensure_dirs()
    
    if args.config:
        config = load_config_from_file(args.config, TrainConfig)
        override_dict = {
            "env": args.env if args.env != "CartPole-v1" else None,
            "timesteps": args.timesteps if args.timesteps != 200_000 else None,
            "seed": args.seed if args.seed != 42 else None,
            "n_steps": args.n_steps if args.n_steps != 2048 else None,
            "batch_size": args.batch_size if args.batch_size != 64 else None,
            "learning_rate": args.learning_rate if args.learning_rate != 3e-4 else None,
        }
        override_dict = {k: v for k, v in override_dict.items() if v is not None}
        if override_dict:
            config = merge_config_with_args(config, override_dict)
            print(f"Loaded config from {args.config} with overrides: {override_dict}")
        else:
            print(f"Loaded config from {args.config}")
    else:
        config = TrainConfig(
            env=args.env,
            timesteps=args.timesteps,
            seed=args.seed,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
    
    set_global_seed(config.seed)

    run_dir = create_run_dir(config)
    save_config(config, run_dir)
    print(f"Run directory: {run_dir}")

    env = gym.make(config.env)
    env.reset(seed=config.seed)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        seed=config.seed,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        gae_lambda=config.gae_lambda,
        gamma=config.gamma,
        n_epochs=config.n_epochs,
        ent_coef=config.ent_coef,
        learning_rate=config.learning_rate,
        clip_range=config.clip_range,
        tensorboard_log=str(run_dir / "tensorboard"),
    )

    cb = RewardLogger()
    model.learn(total_timesteps=config.timesteps, callback=cb)

    model_path = run_dir / "model"
    model.save(str(model_path))

    if len(cb.episode_rewards) > 0:
        plt.figure()
        plt.plot(cb.episode_rewards)
        plt.title("Training Episode Return")
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plot_path = run_dir / "training_returns.png"
        plt.savefig(plot_path, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"Saved plot: {plot_path}")

        metrics = {
            "total_episodes": len(cb.episode_rewards),
            "final_mean_return": float(sum(cb.episode_rewards[-100:]) / min(100, len(cb.episode_rewards))),
            "max_return": float(max(cb.episode_rewards)),
            "min_return": float(min(cb.episode_rewards)),
        }
        save_metrics(metrics, run_dir, "training_metrics.json")
        print(f"Training metrics: {metrics}")

    print(f"Saved model: {model_path}.zip")
    print(f"All artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
