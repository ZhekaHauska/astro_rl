"""Causal episodic familiarity for dense vectors. No predicted writes, no labels.

Inputs are signed, standardized RSNN rates divided by sqrt(neurons).
RFF implements a discounted kernel MEAN, not a maximum. ExactDensity is its
matched mathematical reference; WholePattern is a separate max-similarity model.
"""

from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class DenseConfig:
    neurons: int = 100
    cells: int = 5
    capacity: int = 50
    sigma: float = 0.5
    decay_steps: float = 12.0
    features: int = 512
    feature_seed: int = 73119

    def __post_init__(self):
        if min(self.neurons, self.cells, self.capacity, self.features) < 1 or self.sigma <= 0 or self.decay_steps <= 0:
            raise ValueError("positive dimensions, width and timescale required")


class WholePattern:
    def __init__(self, config=DenseConfig(), reduction="max"):
        self.c = config
        self.reduction = reduction
        self.vectors = np.zeros((config.capacity, config.neurons))
        self.weights = np.zeros(config.capacity)
        self.cursor = 0
        self.count = 0

    def write(self, activity, seed=None):
        x = np.asarray(activity, float)
        assert x.shape == (self.c.neurons,) and np.isfinite(x).all()
        self.weights *= np.exp(-1 / self.c.decay_steps)
        self.vectors[self.cursor] = x
        self.weights[self.cursor] = 1.0
        self.cursor = (self.cursor + 1) % self.c.capacity
        self.count += 1

    def score(self, query):
        q = np.atleast_2d(np.asarray(query, float))
        assert q.shape[1] == self.c.neurons and np.isfinite(q).all()
        use = self.weights > 0
        if not use.any():
            return np.zeros(len(q))
        z = self.vectors[use]
        dist = np.maximum((q * q).sum(1)[:, None] + (z * z).sum(1)[None] - 2 * q @ z.T, 0.0)
        kernel = np.exp(-dist / (2 * self.c.sigma**2))
        weighted = kernel * self.weights[use]
        return weighted.max(1) if self.reduction == "max" else weighted.sum(1) / self.weights.sum()

    def read(self, query):
        score = self.score(query)
        out = np.repeat(score[:, None], self.c.cells, axis=1)
        mass = np.full_like(out, float(self.count > 0))
        return (out[0], mass[0]) if np.asarray(query).ndim == 1 else (out, mass)

    def diagnostics(self):
        return dict(
            memory_entries=min(self.count, self.c.capacity), memory_state_scalars=self.vectors.size + self.weights.size
        )

    def metadata(self):
        return dict(kind="whole_pattern_" + self.reduction, config=asdict(self.c), shared_paths=self.c.cells)


class RandomFeatureMemory(WholePattern):
    def __init__(self, config=DenseConfig()):
        self.c = config
        rng = np.random.default_rng(config.feature_seed)
        self.omega = rng.normal(size=(config.neurons, config.features)) / config.sigma
        self.phase = rng.uniform(0, 2 * np.pi, config.features)
        self.trace = np.zeros(config.features)
        self.total = 0.0
        self.count = 0

    def phi(self, x):
        x = np.asarray(x, float)
        assert x.shape[-1] == self.c.neurons and np.isfinite(x).all()
        return np.sqrt(2 / self.c.features) * np.cos(x @ self.omega + self.phase)

    def write(self, activity, seed=None):
        x = np.asarray(activity, float)
        assert x.shape == (self.c.neurons,)
        decay = np.exp(-1 / self.c.decay_steps)
        self.trace = decay * self.trace + self.phi(x)
        self.total = decay * self.total + 1
        self.count += 1

    def raw_score(self, query):
        return self.phi(np.atleast_2d(query)) @ self.trace / max(self.total, 1e-30)

    def score(self, query):
        return np.clip(self.raw_score(query), 0, 1)

    def diagnostics(self):
        return dict(memory_entries=self.count, memory_state_scalars=self.trace.size + 1)

    def metadata(self):
        return dict(
            kind="random_fourier_kernel_mean",
            config=asdict(self.c),
            shared_paths=self.c.cells,
            fixed_projection_scalars=self.omega.size + self.phase.size,
            signed_features=True,
            calcium=False,
        )
