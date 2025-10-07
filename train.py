import json
import logging
import random
from pathlib import Path

import ale_py
import click
import gymnasium as gym
import numpy as np
import torch

from agents import DQNAgent, PPOAgent, PPOCustomAgent
from utils import LivePlot, ReplayMemory, RolloutBuffer, record_video

logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def train_dqn(
    env_name: str,
    episodes: int,
    batch_size: int,
    buffer_size: int,
    lr: float,
    gamma: float,
    eps_start: float,
    eps_end: float,
    eps_decay: int,
    tau: float,
    device: str,
    save_dir: str,
    log_interval: int,
    seed: int | None,
    plot: bool = False,
) -> DQNAgent:
    """Train DQN agent"""
    logger.info(f"Training DQN on {env_name}")

    reward_plot = LivePlot("DQN - Episode Rewards") if plot else None
    loss_plot = LivePlot("DQN - Training Loss") if plot else None

    env = gym.make(env_name)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    state, _ = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n

    agent = DQNAgent(n_observations, n_actions, lr, gamma, eps_start, eps_end, eps_decay, tau, device)
    memory = ReplayMemory(buffer_size)

    episode_rewards = []
    best_avg_reward = -float("inf")

    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        episode_reward = 0.0
        losses = []

        for _ in range(500):
            action = agent.select_action(state, training=True)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            episode_reward += reward

            reward_tensor = torch.tensor([reward], device=device)
            done = terminated or truncated

            next_state = None if terminated else torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward_tensor)
            state = next_state

            loss = agent.optimize_model(memory, batch_size)
            if loss is not None:
                losses.append(loss)

            agent.soft_update_target_network()

            if done:
                episode_rewards.append(episode_reward)
                break

        avg_loss = np.mean(losses) if losses else 0.0
        avg_reward_100 = np.mean(episode_rewards[-100:]) if episode_rewards else 0.0

        if plot and reward_plot and loss_plot:
            reward_plot.update("Episode Reward", episode_reward, episode)
            reward_plot.update("Avg Reward (100)", avg_reward_100, episode)
            if losses:
                loss_plot.update("Loss", avg_loss, episode)

        if len(episode_rewards) >= 100 and avg_reward_100 > best_avg_reward:
            best_avg_reward = avg_reward_100

        if (episode + 1) % log_interval == 0:
            logger.info(
                f"Episode {episode+1}/{episodes} | "
                f"Reward: {episode_reward:.1f} | "
                f"Avg(100): {avg_reward_100:.1f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Eps: {agent.get_epsilon():.3f}"
            )

    env.close()

    if plot:
        if reward_plot:
            reward_plot.final_save(f"{save_dir}/{env_name}_dqn_reward.png")
            reward_plot.close()
        if loss_plot:
            loss_plot.final_save(f"{save_dir}/{env_name}_dqn_loss.png")
            loss_plot.close()

    return agent


def train_ppo(
    env_name: str,
    total_timesteps: int,
    n_steps: int,
    n_envs: int,
    n_epochs: int,
    batch_size: int,
    lr: float,
    gamma: float,
    gae_lambda: float,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
    device: str,
    save_dir: str,
    log_interval: int,
    seed: int | None,
    plot: bool = False,
    use_inn: bool = False,
    inn_config: str | None = None,
    policy_config: str | None = None,
    value_config: str | None = None,
) -> PPOAgent:
    """Train PPO agent following the 37 implementation details"""
    logger.info(f"Training PPO on {env_name}")

    reward_plot = LivePlot("PPO - Episode Rewards") if plot else None
    loss_plot = LivePlot("PPO - Training Loss") if plot else None

    inn_cfg = json.loads(inn_config) if inn_config else {}
    policy_cfg = json.loads(policy_config) if policy_config else inn_cfg
    value_cfg = json.loads(value_config) if value_config else inn_cfg

    if use_inn:
        logger.info("Using INN networks")
        logger.info(f"Policy config: {policy_cfg}")
        logger.info(f"Value config: {value_cfg}")

    # Create vectorized environments
    def make_env():
        env = gym.make(env_name)
        if seed is not None:
            env.reset(seed=seed)
            env.action_space.seed(seed)
        return env

    envs = gym.vector.SyncVectorEnv([make_env for _ in range(n_envs)])

    state, _ = envs.reset(seed=seed)
    n_observations = envs.single_observation_space.shape[0]
    n_actions = envs.single_action_space.n

    agent = PPOAgent(
        n_observations,
        n_actions,
        lr,
        gamma,
        gae_lambda,
        clip_epsilon,
        value_coef,
        entropy_coef,
        device,
        use_inn=use_inn,
        policy_config=policy_cfg,
        value_config=value_cfg,
    )
    buffer = RolloutBuffer()

    global_step = 0
    episode_rewards = []
    episode_count = 0
    current_episode_rewards = np.zeros(n_envs)

    state = torch.tensor(state, dtype=torch.float32, device=device)

    num_updates = total_timesteps // (n_steps * n_envs)

    for update in range(1, num_updates + 1):
        # Rollout phase: collect n_steps from n_envs
        for _ in range(n_steps):
            global_step += n_envs

            action, log_prob, value = agent.select_action(state, training=True)

            next_state, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            current_episode_rewards += reward

            # Store episode rewards when episodes complete
            for idx, d in enumerate(done):
                if d:
                    episode_rewards.append(current_episode_rewards[idx])
                    current_episode_rewards[idx] = 0.0
                    episode_count += 1

            # Store transitions
            reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device)
            done_tensor = torch.tensor(terminated.astype(np.float32), dtype=torch.float32, device=device)

            buffer.push(state, action, log_prob, reward_tensor, value, done_tensor)

            state = torch.tensor(next_state, dtype=torch.float32, device=device)

        # Learning phase: update policy
        metrics = agent.update(buffer, n_epochs, batch_size)
        buffer.clear()

        avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0.0
        if plot and reward_plot and loss_plot:
            if episode_rewards:
                reward_plot.update("Avg Reward (100)", avg_reward, update)
            loss_plot.update("Policy Loss", metrics["policy_loss"], update)
            loss_plot.update("Value Loss", metrics["value_loss"], update)
            loss_plot.update("Entropy", metrics["entropy"], update)

        # Logging
        if update % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0.0
            logger.info(
                f"Update {update}/{num_updates} | "
                f"Steps: {global_step} | "
                f"Episodes: {episode_count} | "
                f"Avg(100): {avg_reward:.1f} | "
                f"PolicyLoss: {metrics['policy_loss']:.4f} | "
                f"ValueLoss: {metrics['value_loss']:.4f} | "
                f"Entropy: {metrics['entropy']:.4f}"
            )

    envs.close()

    if plot:
        if reward_plot:
            reward_plot.final_save(f"{save_dir}/{env_name}_ppo_reward.png")
            reward_plot.close()
        if loss_plot:
            loss_plot.final_save(f"{save_dir}/{env_name}_ppo_losses.png")
            loss_plot.close()

    return agent


def train_ppo_custom(
    env_name: str,
    total_timesteps: int,
    n_steps: int,
    lr: float,
    gamma: float,
    gae_lambda: float,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
    device: str,
    save_dir: str,
    log_interval: int,
    seed: int | None,
    plot: bool = False,
) -> "PPOCustomAgent":
    """Train custom PPO agent"""
    from agents import PPOCustomAgent

    logger.info(f"Training Custom PPO on {env_name}")

    reward_plot = LivePlot("Custom PPO - Episode Rewards") if plot else None
    loss_plot = LivePlot("Custom PPO - Training Loss") if plot else None

    # Create single environment
    env = gym.make(env_name)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)

    state, _ = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n

    agent = PPOCustomAgent(n_observations, n_actions, lr, gamma, gae_lambda, clip_epsilon, value_coef, entropy_coef, device)

    global_step = 0
    episode_rewards = []
    episode_count = 0
    current_episode_reward = 0.0

    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    num_updates = total_timesteps // n_steps

    for update in range(1, num_updates + 1):
        # Rollout phase
        for _ in range(n_steps):
            global_step += 1

            action_tensor, log_prob, value = agent.select_action(state, training=True)
            action = action_tensor.item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            current_episode_reward += reward

            # Store transition
            agent.store_transition(action, reward, terminated)

            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                episode_count += 1
                next_state, _ = env.reset()

            state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

        # Learning phase
        metrics = agent.update()

        avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0.0
        if plot and reward_plot and loss_plot:
            if episode_rewards:
                reward_plot.update("Avg Reward (100)", avg_reward, update)
            loss_plot.update("Loss", metrics["policy_loss"], update)

        if update % log_interval == 0:
            logger.info(
                f"Update {update}/{num_updates} | "
                f"Steps: {global_step} | "
                f"Episodes: {episode_count} | "
                f"Avg(100): {avg_reward:.1f} | "
                f"Loss: {metrics['policy_loss']:.4f}"
            )

    env.close()

    if plot:
        if reward_plot:
            reward_plot.final_save(f"{save_dir}/{env_name}_ppo_custom_reward.png")
            reward_plot.close()
        if loss_plot:
            loss_plot.final_save(f"{save_dir}/{env_name}_ppo_custom_loss.png")
            loss_plot.close()

    return agent


@click.command()
@click.option("--algorithm", type=click.Choice(["dqn", "ppo", "ppo_custom"]), default="dqn", help="Algorithm to use")
@click.option("--env", default="CartPole-v1", help="Environment name")
@click.option("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
@click.option("--save-dir", default="checkpoints", help="Directory to save models")
@click.option("--log-interval", default=10, help="Logging interval")
@click.option("--seed", default=None, type=int, help="Random seed")
@click.option("--record-video", is_flag=True, help="Record video after training")
@click.option("--video-dir", default="videos", help="Video directory")
@click.option("--plot", is_flag=True, help="Enable live plotting")
# DQN specific
@click.option("--episodes", default=600, help="Number of training episodes (DQN)")
@click.option("--batch-size", default=128, help="Batch size")
@click.option("--buffer-size", default=10000, help="Replay buffer size (DQN)")
@click.option("--lr", default=1e-4, help="Learning rate")
@click.option("--gamma", default=0.99, help="Discount factor")
@click.option("--eps-start", default=0.9, help="Initial epsilon (DQN)")
@click.option("--eps-end", default=0.05, help="Final epsilon (DQN)")
@click.option("--eps-decay", default=1000, help="Epsilon decay (DQN)")
@click.option("--tau", default=0.005, help="Target network update rate (DQN)")
# PPO specific
@click.option("--total-timesteps", default=100000, help="Total timesteps (PPO)")
@click.option("--n-steps", default=128, help="Steps per rollout (PPO)")
@click.option("--n-envs", default=4, help="Number of parallel environments (PPO)")
@click.option("--n-epochs", default=4, help="Update epochs per batch (PPO)")
@click.option("--gae-lambda", default=0.95, help="GAE lambda (PPO)")
@click.option("--clip-epsilon", default=0.2, help="PPO clip epsilon")
@click.option("--value-coef", default=0.5, help="Value loss coefficient (PPO)")
@click.option("--entropy-coef", default=0.01, help="Entropy coefficient (PPO)")
# INN specific
@click.option("--use-inn", is_flag=True, help="Use INN networks")
@click.option("--inn-config", default=None, help="Global INN config as JSON string")
@click.option("--policy-config", default=None, help="Policy head config as JSON string")
@click.option("--value-config", default=None, help="Value head config as JSON string")
def main(**kwargs) -> None:
    """Train RL agent on Gymnasium environment"""
    if kwargs["seed"] is not None:
        random.seed(kwargs["seed"])
        torch.manual_seed(kwargs["seed"])
        np.random.seed(kwargs["seed"])

    save_path = Path(kwargs["save_dir"])
    save_path.mkdir(exist_ok=True)
    gym.register_envs(ale_py)

    if kwargs["algorithm"] == "dqn":
        agent = train_dqn(  # type: ignore
            kwargs["env"],
            kwargs["episodes"],
            kwargs["batch_size"],
            kwargs["buffer_size"],
            kwargs["lr"],
            kwargs["gamma"],
            kwargs["eps_start"],
            kwargs["eps_end"],
            kwargs["eps_decay"],
            kwargs["tau"],
            kwargs["device"],
            kwargs["save_dir"],
            kwargs["log_interval"],
            kwargs["seed"],
            kwargs["plot"],
        )
    elif kwargs["algorithm"] == "ppo":
        agent = train_ppo(  # type: ignore
            kwargs["env"],
            kwargs["total_timesteps"],
            kwargs["n_steps"],
            kwargs["n_envs"],
            kwargs["n_epochs"],
            kwargs["batch_size"],
            kwargs["lr"],
            kwargs["gamma"],
            kwargs["gae_lambda"],
            kwargs["clip_epsilon"],
            kwargs["value_coef"],
            kwargs["entropy_coef"],
            kwargs["device"],
            kwargs["save_dir"],
            kwargs["log_interval"],
            kwargs["seed"],
            kwargs["plot"],
            kwargs["use_inn"],
            kwargs["inn_config"],
            kwargs["policy_config"],
            kwargs["value_config"],
        )
    elif kwargs["algorithm"] == "ppo_custom":
        agent = train_ppo_custom(  # type: ignore
            kwargs["env"],
            kwargs["total_timesteps"],
            kwargs["n_steps"],
            kwargs["lr"],
            kwargs["gamma"],
            kwargs["gae_lambda"],
            kwargs["clip_epsilon"],
            kwargs["value_coef"],
            kwargs["entropy_coef"],
            kwargs["device"],
            kwargs["save_dir"],
            kwargs["log_interval"],
            kwargs["seed"],
            kwargs["plot"],
        )
    else:
        raise ValueError(f"Unknown algorithm: {kwargs['algorithm']}")

    model_path = save_path / f"{kwargs['algorithm']}_final.pt"
    agent.save(str(model_path))
    logger.info(f"Model saved to {model_path}")

    if kwargs["record_video"]:
        logger.info("Recording video...")
        record_video(agent, kwargs["device"], kwargs["video_dir"], kwargs["env"], num_episodes=3)


if __name__ == "__main__":
    main()
