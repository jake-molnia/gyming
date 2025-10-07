# agent/policy/ppo.py

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.distributions import Categorical

if TYPE_CHECKING:
    from agents.ppo_custom import PPOModel


@dataclass
class PPOConfig:
    """Configuration for PPO training."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ppo_epochs: int = 1
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    lr: float = 3e-4


# NOTES: llok into torch-rl, SB3
class PPOPolicy:
    """PPO Policy implementation that handles experience collection and policy updates."""

    def __init__(self, model: "PPOModel", config: PPOConfig, debug: bool = False):
        self.model = model
        self.config = config
        self.debug = debug
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.95)
        self.scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None
        self.update_count = 0

    def sample_action(
        self, obs: dict[str, torch.Tensor], deterministic: bool = False
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            # SPEED: CACHE THE EXPENSIVE ENCODING STEP
            encoded_obs = self.model._encode_obs(obs)
            action_logits, value = self.model.forward_encoded(encoded_obs)

            if deterministic:
                action = torch.argmax(action_logits)
            else:
                dist = Categorical(logits=action_logits)
                action = dist.sample()

            dist = Categorical(logits=action_logits)
            log_prob = dist.log_prob(action)

            return int(action.item()), log_prob, value.squeeze(), encoded_obs  # RETURN CACHED ENCODING

    def evaluate_actions_batch(
        self, encoded_obs_batch: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Single batch forward pass instead of individual processing
        action_logits, values = self.model.forward_batch(encoded_obs_batch)

        dist = Categorical(logits=action_logits)
        actions = actions.to(action_logits.device)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values.squeeze(-1), entropy

    def collect_experience_from_trajectory(self, trajectory_data: list[dict[str, Any]]) -> dict[str, list]:
        if not trajectory_data:
            return {"encoded_obs": [], "actions": [], "log_probs": [], "values": [], "rewards": [], "dones": []}

        return {
            "encoded_obs": [step["encoded_obs"] for step in trajectory_data],  # SPEED: USE CACHED
            "actions": [step["action"] for step in trajectory_data],
            "log_probs": [step["log_prob"] for step in trajectory_data],
            "values": [step["value"] for step in trajectory_data],
            "rewards": [step["reward"] for step in trajectory_data],
            "dones": [step["done"] for step in trajectory_data],
        }

    def update_policy(self, trajectory: dict[str, list]) -> float:
        if not trajectory or len(trajectory["rewards"]) == 0:
            return 0.0

        # Get final value for GAE
        final_encoded_obs = trajectory["encoded_obs"][-1] if trajectory["encoded_obs"] else None
        if final_encoded_obs is not None:
            with torch.no_grad():
                _, next_value = self.model.forward_encoded(final_encoded_obs)
                next_value = next_value.squeeze()
        else:
            next_value = torch.tensor(0.0)

        advantages, returns = self._compute_gae_vectorized(
            trajectory["rewards"], trajectory["values"], trajectory["dones"], next_value
        )

        if len(advantages) == 0:
            return 0.0

        encoded_obs_batch = torch.stack(trajectory["encoded_obs"])  # Already encoded!
        actions = torch.tensor(trajectory["actions"], dtype=torch.long)
        log_probs_old = torch.stack([lp.detach() for lp in trajectory["log_probs"]])

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0

        for _ in range(self.config.ppo_epochs):  # Only 1 epoch by default
            if self.scaler:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
                    log_probs, values, entropy = self.evaluate_actions_batch(encoded_obs_batch, actions)
                    loss = self._compute_ppo_loss(log_probs, log_probs_old, values, returns, advantages, entropy)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                # grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # CPU fallback
                log_probs, values, entropy = self.evaluate_actions_batch(encoded_obs_batch, actions)
                loss = self._compute_ppo_loss(log_probs, log_probs_old, values, returns, advantages, entropy)

                self.optimizer.zero_grad()
                loss.backward()
                # grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

            total_loss += loss.item()

        self.update_count += 1
        if self.update_count % 10 == 0:
            self.scheduler.step()

        return total_loss / self.config.ppo_epochs

    def _compute_gae_vectorized(self, rewards, values, dones, next_value):
        values_tensor = torch.stack(
            [v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32) for v in values]
        )

        # Determine the target device from a model-based tensor
        device = values_tensor.device
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device)
        next_value = next_value.to(device)

        advantages = torch.zeros_like(rewards_tensor, device=device)
        gae = 0.0

        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                next_non_terminal = 1.0 - dones_tensor[step]
                next_value_step = next_value
            else:
                next_non_terminal = 1.0 - dones_tensor[step]
                next_value_step = values_tensor[step + 1]

            delta = rewards_tensor[step] + self.config.gamma * next_value_step * next_non_terminal - values_tensor[step]
            gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
            advantages[step] = gae

        returns = advantages + values_tensor
        return advantages, returns

    def _compute_ppo_loss(self, log_probs, log_probs_old, values, returns, advantages, entropy):
        ratio = torch.exp(log_probs - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = nn.functional.mse_loss(values, returns)
        entropy_loss = -entropy.mean()

        return policy_loss + self.config.vf_coef * value_loss + self.config.ent_coef * entropy_loss

    def save_policy(self, filepath: str):
        torch.save(self.model.state_dict(), filepath)

    def load_policy(self, filepath: str):
        self.model.load_state_dict(torch.load(filepath, map_location="cpu"))
