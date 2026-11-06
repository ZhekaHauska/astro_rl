"""Tabular Q-learning and target-network episodic DQN/DRQN controls."""

import copy
import numpy as np
import torch
from torch import nn
from .config import ModelConfig
from .encoders import InputSpikeCoder


class QNetwork(nn.Module):
    def __init__(self, recurrent):
        super().__init__()
        self.recurrent = recurrent
        self.hidden = (
            nn.RNN(64, 100, batch_first=True, nonlinearity="tanh")
            if recurrent
            else nn.Sequential(nn.Linear(64, 100), nn.Tanh(), nn.Linear(100, 100), nn.Tanh())
        )
        self.output = nn.Linear(100, 4)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, h=None):
        if self.recurrent:
            y, h = self.hidden(x, h)
        else:
            y = self.hidden(x)
        return self.output(y), h


class QLearner:
    has_prediction = False

    def __init__(self, kind, rl, seed, hp, spec, model_config):
        self.kind, self.rl, self.hp = kind, rl, hp
        self.c = ModelConfig(seed=seed, **model_config)
        self.input_coder = InputSpikeCoder(self.c)
        self.observation_channels = 64
        self.spec = spec
        self.free = sorted(map(tuple, spec["free"]))
        self.index = {p: i for i, p in enumerate(self.free)}
        self.codes = {
            self.observation_code(np.asarray(p, dtype=np.float32) / (spec["size"] - 1)).tobytes(): i
            for i, p in enumerate(self.free)
        }
        assert len(self.codes) == len(self.free)
        self.rng = np.random.default_rng(seed + 711113)
        self.replay = []
        self.episode = 0
        self.table = np.zeros((len(self.free), 4))
        self.actor_lr = hp["q_lr"]
        self.entropy = 0.0
        self.epsilon = hp["epsilon_start"]
        if kind != "q_table":
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                self.net = QNetwork(kind == "q_rnn")
            self.target = copy.deepcopy(self.net)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=hp["q_lr"])
        self.reset()

    def observation_code(self, coordinates):
        return self.input_coder.observation_encoder.encode(coordinates)

    def reset(self):
        self.h = None

    def schedule(self, ep, episodes):
        f = min(1.0, (ep - 1) / max(1, (episodes - 1) * self.hp["epsilon_fraction"]))
        self.epsilon = self.hp["epsilon_start"] + (self.hp["epsilon_end"] - self.hp["epsilon_start"]) * f

    def representation(self, coordinates, previous, training):
        obs = self.observation_code(coordinates)
        return self.representation_features(obs, training)

    def representation_features(self, observation, training):
        """Consume an observation encoded by the environment, never coordinates."""
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (self.observation_channels,) or not np.isfinite(obs).all():
            raise ValueError("expected RBF observations")
        if self.kind == "q_table":
            q = self.table[self.codes[obs.tobytes()]].copy()
        else:
            with torch.no_grad():
                q, self.h = self.net(torch.tensor(obs).reshape(1, 1, 64), self.h)
            q = q.numpy().reshape(4)
        return dict(observation=obs, q=q, training=training)

    def weights_hash(self):
        from .mlp import fingerprint

        return fingerprint(
            [self.table] if self.kind == "q_table" else [p.detach().numpy() for p in self.net.parameters()]
        )

    def metadata(self):
        count = self.table.size if self.kind == "q_table" else sum(p.numel() for p in self.net.parameters())
        return dict(
            parameter_count=count,
            prediction_target=None,
            memory_input=None,
            prediction_learning=None,
            policy_learning="episodic Q learning" if self.rl else "frozen uniform control",
            hidden_neurons=100 if self.kind != "q_table" else None,
        )

    def policy(self, rep, gate, memory_on):
        if not self.rl:
            p = np.full(4, 0.25)
        else:
            eps = self.epsilon if rep["training"] else 0.0
            best = np.isclose(rep["q"], np.max(rep["q"]), rtol=0, atol=1e-10)
            p = np.full(4, eps / 4) + (1 - eps) * best / best.sum()
        return p, None, dict(motor_silent_fraction=0.0, mean_gate=0.0)

    def update(self, records):
        if not self.rl:
            return dict(td_loss=None, actor_gradient_norm=0.0, core_step_norm=0.0)
        self.episode += 1
        if self.kind == "q_table":
            losses = []
            for r in records:
                i = self.codes[r["observation"].tobytes()]
                j = self.codes[r["next_observation"].tobytes()]
                target = r["reward"] + (0 if r["terminal"] else self.hp["gamma"] * self.table[j].max())
                error = target - self.table[i, r["action"]]
                self.table[i, r["action"]] += self.hp["q_lr"] * error
                losses.append(error * error)
            return dict(td_loss=float(np.mean(losses)), actor_gradient_norm=0.0, core_step_norm=0.0)
        episode = {
            k: np.asarray([r[k] for r in records])
            for k in ("observation", "next_observation", "action", "reward", "terminal")
        }
        self.replay.append(episode)
        self.replay = self.replay[-100:]
        losses = []
        for _ in range(self.hp["q_epochs"]):
            selected = [self.replay[i] for i in self.rng.integers(len(self.replay), size=min(4, len(self.replay)))]
            # Full episodes reconstruct recurrent state from zero. Padding is masked;
            # no replay of a hidden state produced by outdated recurrent weights.
            length = max(len(x["action"]) for x in selected)
            obs = np.zeros((len(selected), length + 1, 64), np.float32)
            actions = np.zeros((len(selected), length), int)
            rewards = np.zeros_like(actions, dtype=np.float32)
            terminal = np.ones_like(actions, bool)
            mask = np.zeros_like(rewards)
            for b, x in enumerate(selected):
                n = len(x["action"])
                obs[b, :n] = x["observation"]
                obs[b, n] = x["next_observation"][-1]
                actions[b, :n] = x["action"]
                rewards[b, :n] = x["reward"]
                terminal[b, :n] = x["terminal"]
                mask[b, :n] = 1
            inputs = torch.tensor(obs)
            q, _ = self.net(inputs)
            with torch.no_grad():
                target, _ = self.target(inputs)
                y = (
                    torch.tensor(rewards)
                    + self.hp["gamma"] * (1 - torch.tensor(terminal).float()) * target[:, 1:].max(-1).values
                )
            chosen = q[:, :-1].gather(-1, torch.tensor(actions)[..., None]).squeeze(-1)
            loss = (((chosen - y) ** 2) * torch.tensor(mask)).sum() / mask.sum()
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
            self.optimizer.step()
            losses.append(float(loss.detach()))
        if self.episode % self.hp["target_every"] == 0:
            self.target.load_state_dict(self.net.state_dict())
        return dict(td_loss=float(np.mean(losses)), actor_gradient_norm=0.0, core_step_norm=0.0)
