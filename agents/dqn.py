import math
import random

import torch
import torch.nn as nn
import torch.optim as optim

from utils.buffers import ReplayMemory
from utils.networks import DQN


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
        batch = memory.Transition(*zip(*transitions))

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

    def save(self, path: str) -> None:
        """Save policy network"""
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path: str) -> None:
        """Load policy network"""
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.policy_net.state_dict())
