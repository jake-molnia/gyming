from typing import Any

import torch
import torch.nn as nn
from iterativennsimple.MaskedLinear import MaskedLinear


class INN(nn.Module):
    """Simple iterative neural network with sparse connections"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        sparsity: float = 0.5,
        iterations: int = 2,
        init_scale: float = 0.1,
        activation: str = "relu",
        device: str | torch.device = "cpu",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.sparsity = sparsity
        self.iterations = iterations
        self.device = device

        # Input to hidden
        self.input_layer = MaskedLinear.from_description(
            in_features_sizes=[input_dim],
            out_features_sizes=[hidden_dim],
            block_types=[[f"R={sparsity}"]],
            initialization_types=[[f"G=0.0,{init_scale}"]],
            trainable=[["non-zero"]],
        ).to(device)

        # Hidden to hidden (for iterations)
        self.hidden_layer = MaskedLinear.from_description(
            in_features_sizes=[hidden_dim],
            out_features_sizes=[hidden_dim],
            block_types=[[f"R={sparsity}"]],
            initialization_types=[[f"G=0.0,{init_scale}"]],
            trainable=[["non-zero"]],
        ).to(device)

        # Hidden to output
        self.output_layer = MaskedLinear.from_description(
            in_features_sizes=[hidden_dim],
            out_features_sizes=[output_dim],
            block_types=[[f"R={sparsity}"]],
            initialization_types=[[f"G=0.0,{init_scale}"]],
            trainable=[["non-zero"]],
        ).to(device)

        # Activation function
        self.activation = self._get_activation(activation)

    def _get_activation(self, name: str):
        activations = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(), "identity": nn.Identity()}
        return activations.get(name, nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input to hidden
        h = self.input_layer(x)
        h = self.activation(h)

        # Iterate
        for _ in range(self.iterations):
            h = self.hidden_layer(h)
            h = self.activation(h)

        # Output
        out = self.output_layer(h)
        return out


class INNActorCritic(nn.Module):
    """Actor-Critic using two INNs"""

    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        policy_config: dict[str, Any],
        value_config: dict[str, Any],
        device: str = "cpu",
    ):
        super().__init__()

        # Policy head
        self.actor = INN(
            input_dim=n_observations,
            output_dim=n_actions,
            hidden_dim=policy_config.get("hidden_dim", 128),
            sparsity=policy_config.get("sparsity", 0.5),
            iterations=policy_config.get("iterations", 2),
            init_scale=policy_config.get("init_scale", 0.1),
            activation=policy_config.get("activation", "relu"),
            device=device,
        )

        # Value head
        self.critic = INN(
            input_dim=n_observations,
            output_dim=1,
            hidden_dim=value_config.get("hidden_dim", 64),
            sparsity=value_config.get("sparsity", 0.5),
            iterations=value_config.get("iterations", 2),
            init_scale=value_config.get("init_scale", 0.1),
            activation=value_config.get("activation", "relu"),
            device=device,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_probs, value)"""
        action_logits = self.actor(x)
        action_probs = torch.softmax(action_logits, dim=-1)
        value = self.critic(x)
        return action_probs, value
