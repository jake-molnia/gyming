import logging
import math
import random
from collections import deque, namedtuple
from pathlib import Path
from typing import Any

import click
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logging.basicConfig(level=logging.INFO, format="%(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class DQN(nn.Module):
    """Deep Q-Network following PyTorch tutorial architecture"""

    def __init__(self, n_observations: int, n_actions: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)


class ReplayMemory:
    """Experience replay buffer using PyTorch tutorial approach"""

    def __init__(self, capacity: int) -> None:
        self.memory: deque[Transition] = deque([], maxlen=capacity)

    def push(self, *args: Any) -> None:
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class DQNAgent:
    """DQN agent with epsilon-greedy policy and target network"""

    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        lr: float,
        gamma: float,
        eps_start: float,
        eps_end: float,
        eps_decay: int,
        tau: float,
        device: str | torch.device,
    ) -> None:
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.device = device
        self.steps_done = 0

        self.policy_net = DQN(n_observations, n_actions).to(device)
        self.target_net = DQN(n_observations, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.criterion = nn.SmoothL1Loss()

        logger.info(
            f"Initialized DQN: n_obs={n_observations}, n_actions={n_actions}, lr={lr}, gamma={gamma}, tau={tau}, eps={eps_start}->{eps_end}"
        )

    def select_action(self, state: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Epsilon-greedy action selection with exponential decay"""
        if not training:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)

        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.0 * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self, memory: ReplayMemory, batch_size: int) -> float | None:
        """Single optimization step"""
        if len(memory) < batch_size:
            return None

        transitions = memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(s is not None for s in batch.next_state), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item()

    def soft_update_target_network(self) -> None:
        """Soft update: θ′ ← τ θ + (1 −τ )θ′"""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)

        self.target_net.load_state_dict(target_net_state_dict)

    def get_epsilon(self) -> float:
        """Get current epsilon value"""
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.0 * self.steps_done / self.eps_decay)


@click.command()
@click.option("--episodes", default=600, help="Number of training episodes")
@click.option("--batch-size", default=128, help="Batch size for training")
@click.option("--buffer-size", default=10000, help="Replay buffer capacity")
@click.option("--lr", default=1e-4, help="Learning rate")
@click.option("--gamma", default=0.99, help="Discount factor")
@click.option("--eps-start", default=0.9, help="Initial exploration rate")
@click.option("--eps-end", default=0.05, help="Final exploration rate")
@click.option("--eps-decay", default=1000, help="Epsilon decay rate (steps)")
@click.option("--tau", default=0.005, help="Target network soft update rate")
@click.option("--log-interval", default=10, help="Logging interval (episodes)")
@click.option("--eval-interval", default=50, help="Evaluation interval (episodes)")
@click.option("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
@click.option("--save-dir", default="checkpoints", help="Directory to save models")
@click.option("--record-video", is_flag=True, help="Record video of trained agent")
@click.option("--video-dir", default="videos", help="Directory to save videos")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility")
def train(
    episodes: int,
    batch_size: int,
    buffer_size: int,
    lr: float,
    gamma: float,
    eps_start: float,
    eps_end: float,
    eps_decay: int,
    tau: float,
    log_interval: int,
    eval_interval: int,
    device: str,
    save_dir: str,
    record_video: bool,
    video_dir: str,
    seed: int | None,
) -> None:
    """Train on CartPole-v1 using PyTorch tutorial approach"""

    logger.info("=" * 60)
    logger.info("Starting training on CartPole-v1")
    logger.info("=" * 60)

    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        logger.info(f"Using random seed: {seed}")

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    logger.info(f"Models will be saved to {save_path}")

    env = gym.make("CartPole-v1")

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    state, info = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n  # type: ignore[attr-defined]

    logger.info(f"Environment: n_observations={n_observations}, n_actions={n_actions}")

    agent = DQNAgent(n_observations, n_actions, lr, gamma, eps_start, eps_end, eps_decay, tau, device)
    memory = ReplayMemory(buffer_size)

    episode_durations: list[int] = []
    episode_rewards: list[float] = []
    best_avg_reward = -float("inf")

    for episode in range(episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        episode_reward = 0.0
        losses: list[float] = []
        t = 0

        for t in range(500):
            action = agent.select_action(state, training=True)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward_value = reward
            episode_reward += reward_value

            reward_tensor = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push(state, action, next_state, reward_tensor)
            state = next_state

            loss = agent.optimize_model(memory, batch_size)
            if loss is not None:
                losses.append(loss)

            agent.soft_update_target_network()

            if done:
                episode_durations.append(t + 1)
                episode_rewards.append(episode_reward)
                break

        avg_loss = np.mean(losses) if losses else 0.0
        avg_reward_100 = np.mean(episode_rewards[-100:]) if episode_rewards else 0.0

        if len(episode_rewards) >= 100 and avg_reward_100 > best_avg_reward:
            best_avg_reward = avg_reward_100

        # if episode % 100 == 0:
        #     model_path = save_path / f"model_episode_{episode+1}.pt"
        #     torch.save(agent.policy_net.state_dict(), model_path)

        if (episode + 1) % log_interval == 0:
            logger.info(
                f"Episode {episode+1}/{episodes} | "
                f"Duration: {t+1} | "
                f"Reward: {episode_reward:.1f} | "
                f"Avg(100): {avg_reward_100:.1f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Epsilon: {agent.get_epsilon():.3f} | "
                f"Buffer: {len(memory)}"
            )

    env.close()

    final_model_path = save_path / "final_model.pt"
    torch.save(agent.policy_net.state_dict(), final_model_path)
    logger.info(f"Training completed. Final model saved to {final_model_path}")
    logger.info(f"Best average reward: {best_avg_reward:.1f}")

    if record_video:
        logger.info("Recording video of trained agent...")
        record_agent_video(agent, device, video_dir, num_episodes=3)
        logger.info(f"Videos saved to {video_dir}")


def evaluate(agent: DQNAgent, env: gym.Env, device: str | torch.device, num_episodes: int = 10) -> float:
    """Evaluate agent without exploration"""
    total_reward = 0.0

    for _ in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action.item())
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            done = terminated or truncated
            episode_reward += reward

        total_reward += episode_reward

    return total_reward / num_episodes


def record_agent_video(agent: DQNAgent, device: str | torch.device, video_dir: str, num_episodes: int = 3) -> None:
    """Record video of trained agent"""
    video_path = Path(video_dir)
    video_path.mkdir(exist_ok=True)

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, str(video_path), episode_trigger=lambda x: True, name_prefix="trained_agent")

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action.item())
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            done = terminated or truncated
            episode_reward += reward

        logger.info(f"Video episode {episode+1}/{num_episodes} - Reward: {episode_reward:.1f}")

    env.close()


if __name__ == "__main__":
    train()
