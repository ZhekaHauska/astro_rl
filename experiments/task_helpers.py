"""Reward-once rollout with reconstructable per-step and per-checkpoint logs."""

from functools import lru_cache
from collections import deque
from itertools import permutations
import numpy as np


def visibility(remaining, mode):
    bits = np.array([(remaining >> i) & 1 for i in range(4)], np.float32)
    if mode == "ordered" and remaining:
        first = next(i for i in range(4) if bits[i])
        bits[:] = 0
        bits[first] = 1
    return bits


class Oracle:
    def __init__(self, config, mode):
        self.free = set(map(tuple, config["map"]["free"]))
        self.points = list(map(tuple, config["checkpoints"]))
        self.mode = mode
        self.dist = {}
        for start in self.free:
            d = {start: 0}
            queue = deque([start])
            while queue:
                p = queue.popleft()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    q = p[0] + dr, p[1] + dc
                    if q in self.free and q not in d:
                        d[q] = d[p] + 1
                        queue.append(q)
            self.dist[start] = d

    @lru_cache(None)
    def remaining(self, pos, mask):
        points = [p for i, p in enumerate(self.points) if mask & (1 << i)]
        if not points:
            return 0

        def cost(order):
            return self.dist[pos][order[0]] + sum(self.dist[a][b] for a, b in zip(order, order[1:]))

        return cost(points) if self.mode == "ordered" else min(map(cost, permutations(points)))


ORACLES = {}
