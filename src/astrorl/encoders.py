"""Legacy projection and sparse RBF observations for current RSNN models."""

import numpy as np
import torch


class InputSpikeCoder:
    def __init__(self, config):
        self.c = config
        self.context = np.zeros(config.context_channels, dtype=np.float32)
        if config.input_encoding not in ("projection", "rbf"):
            raise ValueError("input_encoding must be projection or rbf")
        if config.input_encoding == "rbf":
            self.observation_encoder = RBFEncoder(config)
            self.n_actions = config.n_actions
            self.action_gain = config.input_action_gain
            self.probability_scale = config.input_probability_scale
            if not config.project_rbf:
                if (
                    config.input_channels != config.observation_channels + config.context_channels
                    or self.action_gain != 0
                ):
                    raise ValueError("Direct RBF requires matching channel counts and zero action gain")
                return
        rng = np.random.default_rng(config.seed + 10)
        channels = config.observation_channels if config.input_encoding == "rbf" else config.obs_dim
        std = config.input_projection_std if config.input_encoding == "rbf" else 2.5
        self.w = torch.tensor(
            rng.normal(0, std, (channels + config.n_actions, config.input_channels)),
            dtype=torch.float32,
        )
        self.bias = torch.tensor(rng.normal(0, 0.65, config.input_channels), dtype=torch.float32)
        self.n_actions = config.n_actions
        self.action_gain = config.input_action_gain
        self.probability_scale = config.input_probability_scale

    def encode(self, obs, prev_action=None):
        if self.c.input_encoding == "rbf":
            return self.encode_features(self.observation_encoder.encode(obs), prev_action)
        obs = torch.as_tensor(obs, dtype=torch.float32).reshape(-1)
        action = torch.zeros(self.n_actions)
        if prev_action is not None:
            action[prev_action] = 1.0
        data = torch.cat((obs - 0.5, self.action_gain * action))
        return ((0.05 + 0.82 * torch.sigmoid(data @ self.w + self.bias)) * self.probability_scale).clamp(0, 1)

    def encode_features(self, observation, prev_action=None):
        if self.c.input_encoding != "rbf":
            raise ValueError("encode_features requires RBF input mode")
        features = torch.as_tensor(observation, dtype=torch.float32).reshape(-1)
        if features.shape != (self.c.observation_channels,) or not torch.isfinite(features).all():
            raise ValueError("Unexpected observation shape or non-finite features")
        if not self.c.project_rbf:
            return (torch.cat((features, torch.as_tensor(self.context))) * self.probability_scale).clamp(0, 1)
        action = torch.zeros(self.n_actions)
        if prev_action is not None:
            action[prev_action] = 1.0
        data = torch.cat((features, self.action_gain * action))
        return ((0.05 + 0.82 * torch.sigmoid(data @ self.w + self.bias)) * self.probability_scale).clamp(0, 1)


class RBFEncoder:
    def __init__(self, config):
        self.channels, self.width, self.topk = config.observation_channels, config.rbf_width, config.rbf_topk
        side = int(np.sqrt(self.channels))
        if config.obs_dim != 2 or side * side != self.channels:
            raise ValueError("RBF observations require a square 2D lattice")
        if not 1 <= self.topk <= self.channels or self.width <= 0:
            raise ValueError("Invalid RBF width or topk")
        self.centers = np.array(
            [((col + 0.5) / side, (row + 0.5) / side) for row in range(side) for col in range(side)]
        )

    def encode(self, coordinates):
        position = np.asarray(coordinates, dtype=np.float64)
        if position.shape != (2,) or not np.isfinite(position).all():
            raise ValueError("Expected finite normalized coordinates in the environment encoder")
        score = np.exp(-np.square(self.centers - position).sum(-1) / (2 * self.width**2))
        active = np.argpartition(score, -self.topk)[-self.topk :]
        result = np.zeros(self.channels)
        result[active] = score[active]
        result /= max(float(score[active].max()), 1e-8)
        return result.astype(np.float32)
