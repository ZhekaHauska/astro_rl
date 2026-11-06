"""Local astrocytic territories in RSNN space, with no observation buffer."""

from dataclasses import dataclass, asdict
import copy
import numpy as np
from .memory import FamiliarityMemory


@dataclass(frozen=True)
class PopulationConfig:
    cells: int = 5
    overlap: int = 1
    tau: float = 1.2
    heterogeneous_tau: bool = False
    tau_span: float = 4.0
    connectivity_seed: int = 1911
    activity_code: str = "excess"


def territories(neurons, cells, overlap, seed):
    """Balanced degree, fixed local masks; interleave LIF/ALIF populations."""
    if neurons % cells or not 1 <= overlap <= cells:
        raise ValueError("cells must divide neuron count and 1<=overlap<=cells")
    rng = np.random.default_rng(seed)
    half = neurons // 2
    first, second = rng.permutation(half), rng.permutation(np.arange(half, neurons))
    order = np.ravel(np.column_stack((first, second)))
    chunks = np.split(order, cells)
    mask = np.zeros((cells, neurons), bool)
    for i in range(cells):
        for d in range(overlap):
            mask[i, chunks[(i + d) % cells]] = True
    assert np.all(mask.sum(0) == overlap)
    return mask


class LocalAstroPopulation:
    def __init__(self, neurons, params, config):
        self.config = config
        self.mask = territories(neurons, config.cells, config.overlap, config.connectivity_seed)
        self.indices = [np.flatnonzero(m) for m in self.mask]
        if config.tau <= 0 or config.tau_span < 1:
            raise ValueError("positive tau and tau_span>=1 required")
        self.taus = (
            config.tau * np.geomspace(1 / config.tau_span, config.tau_span, config.cells)
            if config.heterogeneous_tau
            else np.full(config.cells, config.tau)
        )
        self.cells = []
        for ix, tau in zip(self.indices, self.taus):
            p = copy.deepcopy(params)
            p["astro"]["tau"] = float(tau)
            self.cells.append(FamiliarityMemory(len(ix), "astro", p))

    def write(self, activity, seed):
        activity = np.asarray(activity, float)
        if activity.shape != (self.mask.shape[1],) or not np.isfinite(activity).all() or (activity < 0).any():
            raise ValueError("expected finite nonnegative RSNN signal")
        for i, (ix, cell) in enumerate(zip(self.indices, self.cells)):
            # Distinct deterministic event streams; independent of global NumPy RNG.
            local_seed = int(np.random.SeedSequence([int(seed), i, 431]).generate_state(1)[0])
            cell.write(activity[ix], local_seed)

    def familiarity(self, future_activity):
        q = np.asarray(future_activity, float)
        if q.ndim != 2 or q.shape[1] != self.mask.shape[1] or not np.isfinite(q).all() or (q < 0).any():
            raise ValueError("expected actions x neurons nonnegative forecast")
        local = np.stack([cell.familiarity(q[:, ix]) for ix, cell in zip(self.indices, self.cells)], axis=1)
        # Equal territorial contribution, unlike one global activity-weighted readout.
        return local.mean(1), local

    def diagnostics(self):
        a = np.concatenate([cell.active() for cell in self.cells])
        return dict(
            memory_active_fraction=float((a > 0).mean()),
            memory_level_mean=float(a.mean()),
            astro_cell_level_std=float(np.std([c.active().mean() for c in self.cells])),
        )

    def metadata(self):
        return dict(
            config=asdict(self.config),
            taus=self.taus.tolist(),
            connectivity=self.mask.astype(int).tolist(),
            readout="mean of within-territory query-weighted calcium familiarity",
            states="local glutamate, calcium and calcium velocity per territory process; no episode buffer",
            trainable_parameters=0,
        )
