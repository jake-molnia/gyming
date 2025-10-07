import torch
import torch.nn as nn

from agents.policy.ppo import PPOConfig, PPOPolicy


class PPOModel(nn.Module):
    """Model compatible with custom PPO policy."""

    def __init__(self, n_observations: int, n_actions: int) -> None:
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Actor head
        self.actor = nn.Linear(128, n_actions)

        # Critic head
        self.critic = nn.Linear(128, 1)

    def _encode_obs(self, obs: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Encode observation to latent representation."""
        if isinstance(obs, dict):
            obs = obs.get("obs", obs.get("observation"))

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        return self.encoder(obs)

    def forward_encoded(self, encoded_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass from encoded observation."""
        action_logits = self.actor(encoded_obs)
        value = self.critic(encoded_obs)
        return action_logits, value

    def forward_batch(self, encoded_obs_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch forward pass."""
        action_logits = self.actor(encoded_obs_batch)
        values = self.critic(encoded_obs_batch)
        return action_logits, values


class PPOCustomAgent:
    """Wrapper for custom PPO implementation to match standard agent interface."""

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
        ppo_epochs: int = 1,
        max_grad_norm: float = 0.5,
    ) -> None:
        self.n_actions = n_actions
        self.device = device

        # Create model
        self.model = PPOModel(n_observations, n_actions).to(device)

        # Create PPO config
        config = PPOConfig(
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_eps=clip_epsilon,
            ppo_epochs=ppo_epochs,
            vf_coef=value_coef,
            ent_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            lr=lr,
        )

        # Create policy
        self.policy = PPOPolicy(self.model, config, debug=False)

        # Trajectory storage
        self.trajectory: list[dict] = []

    def select_action(self, state: torch.Tensor, training: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select action using policy network."""
        obs_dict = {"obs": state}

        if training:
            action, log_prob, value, encoded_obs = self.policy.sample_action(obs_dict, deterministic=False)
            # Store for trajectory
            self._last_encoded_obs = encoded_obs
            self._last_log_prob = log_prob
            self._last_value = value
        else:
            action, log_prob, value, _ = self.policy.sample_action(obs_dict, deterministic=True)

        action_tensor = torch.tensor([action], device=self.device)
        return action_tensor, log_prob, value

    def store_transition(self, action: int, reward: float, done: bool) -> None:
        """Store transition in trajectory."""
        self.trajectory.append(
            {
                "encoded_obs": self._last_encoded_obs,
                "action": action,
                "log_prob": self._last_log_prob,
                "value": self._last_value,
                "reward": reward,
                "done": done,
            }
        )

    def update(self) -> dict[str, float]:
        """Update policy using collected trajectory."""
        if len(self.trajectory) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # Collect experience
        experience = self.policy.collect_experience_from_trajectory(self.trajectory)

        # Update policy
        loss = self.policy.update_policy(experience)

        # Clear trajectory
        self.trajectory = []

        return {
            "policy_loss": loss,
            "value_loss": 0.0,  # Combined in custom PPO
            "entropy": 0.0,  # Combined in custom PPO
        }

    def save(self, path: str) -> None:
        """Save model."""
        self.policy.save_policy(path)

    def load(self, path: str) -> None:
        """Load model."""
        self.policy.load_policy(path)
