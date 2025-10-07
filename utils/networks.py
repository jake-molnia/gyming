import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Deep Q-Network for value-based methods"""

    def __init__(self, n_observations: int, n_actions: int) -> None:
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)


class ActorCritic(nn.Module):
    """Actor-Critic network for policy-based methods"""

    def __init__(self, n_observations: int, n_actions: int) -> None:
        super().__init__()

        # Shared layers
        self.shared1 = nn.Linear(n_observations, 128)
        self.shared2 = nn.Linear(128, 128)

        # Actor head (policy)
        self.actor = nn.Linear(128, n_actions)

        # Critic head (value)
        self.critic = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.shared1(x))
        x = F.relu(self.shared2(x))

        # Actor outputs action probabilities
        action_probs = F.softmax(self.actor(x), dim=-1)

        # Critic outputs state value
        value = self.critic(x)

        return action_probs, value
