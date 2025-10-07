import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from utils.buffers import RolloutBuffer
from utils.networks import ActorCritic


class PPOAgent:
    """PPO agent with clipped objective and GAE"""

    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        lr: float,
        gamma: float,
        gae_lambda: float,
        clip_epsilon: float,
        value_coef: float,
        entropy_coef: float,
        device: str | torch.device,
    ) -> None:
        self.n_actions = n_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = device

        self.actor_critic = ActorCritic(n_observations, n_actions).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)

    def select_action(self, state: torch.Tensor, training: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select action using policy network"""
        with torch.no_grad():
            action_probs, value = self.actor_critic(state)
            dist = Categorical(action_probs)

            if training:
                action = dist.sample()
            else:
                action = action_probs.argmax()

            log_prob = dist.log_prob(action)
            value = value.squeeze(-1)  # Remove last dimension: [batch, 1] -> [batch]

        return action, log_prob, value

    def compute_gae(self, buffer: RolloutBuffer) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation"""
        advantages = []  # type: ignore
        gae = 0.0

        # Add terminal value of 0
        values = buffer.values + [torch.tensor([0.0], dtype=torch.float32, device=self.device)]

        for t in reversed(range(len(buffer.rewards))):
            delta = buffer.rewards[t] + self.gamma * values[t + 1] * (1 - buffer.dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - buffer.dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.cat(advantages)
        returns = advantages + torch.cat(buffer.values)

        return advantages, returns

    def update(self, buffer: RolloutBuffer, n_epochs: int, batch_size: int) -> dict[str, float]:
        """Update policy using PPO clipped objective"""
        advantages, returns = self.compute_gae(buffer)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Prepare batch data
        states = torch.cat(buffer.states)
        actions = torch.cat(buffer.actions)
        old_log_probs = torch.cat(buffer.log_probs)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(n_epochs):
            # Mini-batch updates
            indices = torch.randperm(len(states))

            for start in range(0, len(states), batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Forward pass
                action_probs, values = self.actor_critic(batch_states)
                dist = Categorical(action_probs)

                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Policy loss with clipping
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss - flatten both to 1D
                value_loss = nn.MSELoss()(values.squeeze(-1), batch_returns.squeeze(-1))

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def save(self, path: str) -> None:
        """Save actor-critic network"""
        torch.save(self.actor_critic.state_dict(), path)

    def load(self, path: str) -> None:
        """Load actor-critic network"""
        self.actor_critic.load_state_dict(torch.load(path))
