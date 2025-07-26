"""
Shared utilities and neural network architectures for Reinforcement Learning project.

Authors: Amit Ezer, Gal Yaacov Noy
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seeds(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def create_env(env_name, **kwargs):
    """Create a MiniGrid environment with RGB observation wrappers."""
    env = gym.make(env_name, **kwargs)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    return env

def preprocess(obs, target_size=(56, 56)):
    """
    Convert RGB observation(s) to normalized torch tensor(s) of shape:
      - (C, H, W) for single image
      - (N, C, H, W) for batch of images

    Resizes input to `target_size` using bilinear interpolation.
    """
    obs_tensor = torch.tensor(obs, dtype=torch.float32)

    if obs_tensor.ndim == 3:
        # Single image: (H, W, C) → (1, C, H, W)
        obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
        obs_tensor = F.interpolate(obs_tensor, size=target_size, mode='bilinear', align_corners=False)
        return obs_tensor.squeeze(0)

    elif obs_tensor.ndim == 4:
        # Batched: (N, H, W, C) → (N, C, H, W)
        obs_tensor = obs_tensor.permute(0, 3, 1, 2) / 255.0
        obs_tensor = F.interpolate(obs_tensor, size=target_size, mode='bilinear', align_corners=False)
        return obs_tensor

    else:
        raise ValueError(f"Unexpected observation shape: {obs_tensor.shape}")
            
class MiniGridCNN(nn.Module):
    """CNN encoder for MiniGrid environments."""
    
    def __init__(self, output_dim=128, input_channels=3, input_size=56):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Compute the output size after the conv layers by passing a dummy input.
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_size, input_size)
            conv_out_size = self.conv(dummy_input).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, output_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.fc(self.conv(x))

class QNetwork(nn.Module):
    """Q-Network for DQN/Double DQN."""
    
    def __init__(self, num_actions, feature_dim=128, input_size=56):
        super().__init__()
        self.encoder = MiniGridCNN(output_dim=feature_dim, input_size=input_size)
        self.q_head = nn.Linear(feature_dim, num_actions)
        
    def forward(self, x):
        features = self.encoder(x)
        return self.q_head(features)

class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(self, num_actions, input_size=56, feature_dim=128):
        super().__init__()
        self.encoder = MiniGridCNN(output_dim=feature_dim, input_size=input_size)
        self.actor = nn.Linear(feature_dim, num_actions)
        self.critic = nn.Linear(feature_dim, 1)

    def forward(self, x):
        features = self.encoder(x)
        return self.actor(features), self.critic(features)

    def get_action(self, state):
        logits, _ = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate(self, obs, actions):
        logits, values = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(), entropy