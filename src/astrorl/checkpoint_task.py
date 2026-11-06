"""Reward-once task dynamics; observation/model interfaces intentionally separate."""

from dataclasses import dataclass


@dataclass(frozen=True)
class State:
    position: tuple
    remaining: int
    steps: int = 0


class CheckpointTask:
    def __init__(self, config, mode, horizon):
        if mode not in ("ordered", "unordered"):
            raise ValueError(mode)
        self.free = set(map(tuple, config["map"]["free"]))
        self.points = list(map(tuple, config["checkpoints"]))
        self.start = tuple(config["fixed_start"])
        self.mode = mode
        self.horizon = horizon
        assert self.start in self.free and self.start not in self.points and horizon > 0

    def reset(self):
        return State(self.start, (1 << len(self.points)) - 1)

    def step(self, state, action):
        if state.remaining == 0 or state.steps >= self.horizon:
            raise ValueError("Episode already ended")
        dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[action]
        q = (state.position[0] + dr, state.position[1] + dc)
        if q not in self.free:
            q = state.position
        remaining = state.remaining
        reward = 0
        active = [i for i in range(len(self.points)) if remaining & (1 << i)]
        if self.mode == "ordered":
            active = active[:1]
        for i in active:
            if q == self.points[i]:
                remaining &= ~(1 << i)
                reward = 1
                break
        nxt = State(q, remaining, state.steps + 1)
        return nxt, reward, remaining == 0, nxt.steps >= self.horizon and remaining != 0
