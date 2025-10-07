import logging
from pathlib import Path

import gymnasium as gym
import torch

logger = logging.getLogger(__name__)


def evaluate(agent, env: gym.Env, device: str | torch.device, num_episodes: int = 10) -> float:
    """Evaluate agent without exploration"""
    total_reward = 0.0

    for _ in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_reward = 0.0
        done = False

        while not done:
            # DQN uses select_action, PPO uses select_action with different signature
            if hasattr(agent, "policy_net"):
                action = agent.select_action(state, training=False)
                action_value = action.item()
            else:
                action, _, _ = agent.select_action(state, training=False)
                action_value = action.item()

            state, reward, terminated, truncated, _ = env.step(action_value)
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            done = terminated or truncated
            episode_reward += reward

        total_reward += episode_reward

    return total_reward / num_episodes


def record_video(agent, device: str | torch.device, video_dir: str, env_name: str, num_episodes: int = 3) -> None:
    """Record video of trained agent"""
    video_path = Path(video_dir)
    video_path.mkdir(exist_ok=True)

    env = gym.make(env_name, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, str(video_path), episode_trigger=lambda x: True, name_prefix="trained_agent")

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_reward = 0.0
        done = False

        while not done:
            if hasattr(agent, "policy_net"):
                action = agent.select_action(state, training=False)
                action_value = action.item()
            else:
                action, _, _ = agent.select_action(state, training=False)
                action_value = action.item()

            state, reward, terminated, truncated, _ = env.step(action_value)
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            done = terminated or truncated
            episode_reward += reward

        logger.info(f"Video episode {episode+1}/{num_episodes} - Reward: {episode_reward:.1f}")

    env.close()
