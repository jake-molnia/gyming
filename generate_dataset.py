import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import gymnasium as gym
import numpy as np
import pandas as pd
import torch

from agents import PPOAgent
from utils import RolloutBuffer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


ENVIRONMENTS = [
    "CartPole-v1",
    "Acrobot-v1",
    "MountainCar-v0",
    "LunarLander-v3",
]


def get_env_metadata(env_name: str, device: str) -> dict:
    """Generate environment metadata columns."""
    logger.info(f"Collecting metadata for {env_name}")

    env = gym.make(env_name)
    state, _ = env.reset()

    metadata = {
        "env_name": env_name,
        "state_dim": len(state) if hasattr(state, "__len__") else env.observation_space.shape[0],
        "action_dim": env.action_space.n,
        "max_episode_steps": env.spec.max_episode_steps if env.spec else 500,
    }

    # Run baseline PPO
    baseline_reward, baseline_std = run_baseline_ppo(env_name, device)
    metadata["baseline_ppo_reward"] = baseline_reward
    metadata["baseline_ppo_std"] = baseline_std
    metadata["baseline_dqn_reward"] = None  # Optional

    # L2 approximation (simplified placeholder)
    metadata["l2_approximation_error"] = np.random.uniform(0.1, 1.0)

    # Task complexity score
    metadata["task_complexity_score"] = metadata["state_dim"] * metadata["action_dim"] / max(abs(baseline_reward), 1.0)

    env.close()
    return metadata


def run_baseline_ppo(env_name: str, device: str, timesteps: int = 50000) -> tuple[float, float]:
    """Run baseline PPO to establish environment performance."""
    env = gym.vector.SyncVectorEnv([lambda: gym.make(env_name) for _ in range(4)])
    state, _ = env.reset()

    n_obs = env.single_observation_space.shape[0]
    n_actions = env.single_action_space.n

    agent = PPOAgent(n_obs, n_actions, 3e-4, 0.99, 0.95, 0.2, 0.5, 0.01, device, use_inn=False)
    buffer = RolloutBuffer()

    episode_rewards = []
    current_rewards = np.zeros(4)
    state = torch.tensor(state, dtype=torch.float32, device=device)

    for step in range(timesteps // 4):
        action, log_prob, value = agent.select_action(state, training=True)
        next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
        done = np.logical_or(terminated, truncated)

        current_rewards += reward
        for idx, d in enumerate(done):
            if d:
                episode_rewards.append(current_rewards[idx])
                current_rewards[idx] = 0.0

        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device)
        done_tensor = torch.tensor(terminated.astype(np.float32), dtype=torch.float32, device=device)
        buffer.push(state, action, log_prob, reward_tensor, value, done_tensor)
        state = torch.tensor(next_state, dtype=torch.float32, device=device)

        if (step + 1) % 128 == 0:
            agent.update(buffer, 4, 64)
            buffer.clear()

    env.close()

    final_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
    return float(np.mean(final_rewards)), float(np.std(final_rewards))


def run_inn_experiment(
    env_name: str,
    policy_config: dict,
    value_config: dict,
    train_config: dict,
    device: str,
) -> dict:
    """Run single INN experiment and return performance metrics."""
    start_time = time.time()

    env = gym.vector.SyncVectorEnv([lambda: gym.make(env_name) for _ in range(train_config["n_envs"])])
    state, _ = env.reset(seed=train_config["random_seed"])

    n_obs = env.single_observation_space.shape[0]
    n_actions = env.single_action_space.n

    agent = PPOAgent(
        n_obs,
        n_actions,
        train_config["learning_rate"],
        train_config["gamma"],
        train_config["gae_lambda"],
        0.2,
        0.5,
        0.01,
        device,
        use_inn=True,
        policy_config=policy_config,
        value_config=value_config,
    )

    buffer = RolloutBuffer()
    episode_rewards = []
    current_rewards = np.zeros(train_config["n_envs"])
    state = torch.tensor(state, dtype=torch.float32, device=device)
    convergence_timestep = None

    num_updates = train_config["total_timesteps"] // (train_config["n_steps"] * train_config["n_envs"])

    for update in range(num_updates):
        for _ in range(train_config["n_steps"]):
            action, log_prob, value = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            current_rewards += reward
            for idx, d in enumerate(done):
                if d:
                    episode_rewards.append(current_rewards[idx])
                    current_rewards[idx] = 0.0

            reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device)
            done_tensor = torch.tensor(terminated.astype(np.float32), dtype=torch.float32, device=device)
            buffer.push(state, action, log_prob, reward_tensor, value, done_tensor)
            state = torch.tensor(next_state, dtype=torch.float32, device=device)

        metrics = agent.update(buffer, train_config["n_epochs"], train_config["batch_size"])
        buffer.clear()

        # Check convergence
        if convergence_timestep is None and len(episode_rewards) >= 100:
            if np.mean(episode_rewards[-100:]) > -200:  # Threshold
                convergence_timestep = update * train_config["n_steps"] * train_config["n_envs"]

    env.close()

    final_rewards = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards
    total_params = sum(p.numel() for p in agent.actor_critic.parameters())

    return {
        "final_reward_mean": float(np.mean(final_rewards)),
        "final_reward_std": float(np.std(final_rewards)),
        "convergence_timestep": convergence_timestep if convergence_timestep else train_config["total_timesteps"],
        "final_policy_loss": metrics["policy_loss"],
        "final_value_loss": metrics["value_loss"],
        "training_time_seconds": time.time() - start_time,
        "total_parameters": total_params,
    }


def generate_hyperparameter_configs(n_samples: int = 20) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Generate random INN hyperparameter configurations."""
    configs = []
    # NOTE: These ranges are made up and need to be fixed to more usful values
    hidden_dims = [64, 128, 256]
    sparsities = [0.3, 0.5, 0.7]
    iterations = [1, 2, 3]
    init_scales = [0.01, 0.1, 0.2]
    activations = ["relu", "tanh"]

    for _ in range(n_samples):
        policy_config = {
            "hidden_dim": np.random.choice(hidden_dims),
            "sparsity": np.random.choice(sparsities),
            "iterations": np.random.choice(iterations),
            "init_scale": np.random.choice(init_scales),
            "activation": np.random.choice(activations),
        }

        value_config = {
            "hidden_dim": np.random.choice(hidden_dims),
            "sparsity": np.random.choice(sparsities),
            "iterations": np.random.choice(iterations),
            "init_scale": np.random.choice(init_scales),
            "activation": np.random.choice(activations),
        }

        train_config = {
            "learning_rate": 10 ** np.random.uniform(-5, -3),
            "total_timesteps": 100000,
            "n_steps": np.random.choice([128, 256, 512]),
            "batch_size": 64,
            "n_epochs": 4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "n_envs": 4,
            "random_seed": np.random.randint(0, 10000),
        }

        configs.append((policy_config, value_config, train_config))

    return configs


@click.command()
@click.option("--n-samples", default=20, help="Number of hyperparameter samples per environment")
@click.option("--device", default="cuda" if torch.cuda.is_available() else "cpu")
@click.option("--output-dir", default="dataset", help="Output directory")
def main(n_samples: int, device: str, output_dir: str):
    """Generate INN hyperparameter dataset for all environments."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    all_rows = []
    experiment_id = 0

    for env_name in ENVIRONMENTS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing environment: {env_name}")
        logger.info(f"{'='*60}\n")

        # Get environment metadata once
        env_metadata = get_env_metadata(env_name, device)

        # Generate hyperparameter configurations
        configs = generate_hyperparameter_configs(n_samples)

        for policy_cfg, value_cfg, train_cfg in configs:
            experiment_id += 1
            logger.info(f"Experiment {experiment_id}/{n_samples * len(ENVIRONMENTS)}")

            try:
                # Run experiment
                performance = run_inn_experiment(env_name, policy_cfg, value_cfg, train_cfg, device)

                # Combine all data
                row = {
                    **env_metadata,
                    "policy_hidden_dim": policy_cfg["hidden_dim"],
                    "policy_sparsity": policy_cfg["sparsity"],
                    "policy_iterations": policy_cfg["iterations"],
                    "policy_init_scale": policy_cfg["init_scale"],
                    "policy_activation": policy_cfg["activation"],
                    "value_hidden_dim": value_cfg["hidden_dim"],
                    "value_sparsity": value_cfg["sparsity"],
                    "value_iterations": value_cfg["iterations"],
                    "value_init_scale": value_cfg["init_scale"],
                    "value_activation": value_cfg["activation"],
                    **train_cfg,
                    **performance,
                    "experiment_id": experiment_id,
                    "timestamp": datetime.now().isoformat(),
                    "git_commit": "main",
                }

                all_rows.append(row)
                logger.info(f"  Reward: {performance['final_reward_mean']:.2f} ± {performance['final_reward_std']:.2f}")

            except Exception as e:
                logger.error(f"  Failed: {e}")
                continue

    # Create DataFrame and save
    df = pd.DataFrame(all_rows)

    # Save as parquet (HuggingFace compatible)
    parquet_path = output_path / "inn_hyperparameter_dataset.parquet"
    df.to_parquet(parquet_path, index=False)
    logger.info(f"\nDataset saved to {parquet_path}")

    # Also save as CSV for easy inspection
    csv_path = output_path / "inn_hyperparameter_dataset.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"CSV saved to {csv_path}")

    # Print summary statistics
    logger.info("\nDataset Summary:")
    logger.info(f"  Total rows: {len(df)}")
    logger.info(f"  Environments: {df['env_name'].nunique()}")
    logger.info(f"  Mean reward: {df['final_reward_mean'].mean():.2f}")
    logger.info(f"  Reward std: {df['final_reward_mean'].std():.2f}")


if __name__ == "__main__":
    main()
