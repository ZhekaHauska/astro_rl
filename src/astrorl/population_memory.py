"""Local calcium processes, independent write/forget times, and query calibration."""

from dataclasses import dataclass, asdict
import numpy as np
from numba import njit
from .local_memory import territories
from .memory import ca_constant


@dataclass(frozen=True)
class MemoryConfig:
    cells: int = 5
    neurons: int = 100
    connectivity_seed: int = 1911
    tau_write: float = 0.1
    tau_forget: float = 1.2
    heterogeneous: bool = False
    duration: float = 1.0
    dt: float = 0.001
    amplitude: float = 0.73
    gthr: float = 0.1
    threshold: float = 0.03
    reliability_tol: float = 0.1
    reliability_alpha: float = 0.05
    reliability_prior_count: float = 3.0
    query_mass_floor: float = 0.2
    connectivity: str = "local"
    overlap: int = 1
    tau_span: float = 4.0

    def __post_init__(self):
        if self.cells < 1 or self.neurons < 2 or self.neurons % self.cells or self.neurons % 2:
            raise ValueError("cells must divide a positive even neuron count")
        if self.connectivity not in ("local", "full") or not 1 <= self.overlap <= self.cells:
            raise ValueError("invalid connectivity or overlap")
        if self.tau_write <= 0 or self.tau_forget <= 0 or self.tau_span < 1 or self.dt != 0.001 or self.duration <= 0:
            raise ValueError("invalid calcium time constants")
        if not 0 <= self.threshold < self.amplitude or self.gthr <= 0:
            raise ValueError("invalid calcium thresholds")
        if (
            self.reliability_tol <= 0
            or not 0 < self.reliability_alpha <= 1
            or self.reliability_prior_count < 0
            or self.query_mass_floor <= 0
        ):
            raise ValueError("invalid reliability settings")


@njit(cache=True)
def gap(g, c, v, n, dt, write_tau, forget_tau, amplitude, gthr):
    high = 0
    if g > gthr:
        high = min(n, max(0, int(np.ceil(np.log(g / gthr) / dt)) - 1))
    if high > 0:
        c, v = ca_constant(c, v, high * dt, write_tau, amplitude)
    if n > high:
        c, v = ca_constant(c, v, (n - high) * dt, forget_tau, 0.0)
    if c < 0:
        c = 0.0
        v = max(v, 0.0)
    if c > amplitude:
        c = amplitude
        v = min(v, 0.0)
    return g * np.exp(-dt * n), c, v


@njit(cache=True)
def advance(g, ca, velocity, signal, seed, duration, dt, write_tau, forget_tau, amplitude, gthr):
    np.random.seed(seed)
    nsteps = max(1, int(round(duration / dt)))
    dt = duration / nsteps
    eg = np.exp(-dt)
    for j in range(len(g)):
        count = np.random.poisson(15.0 * signal[j] * duration)
        bins = np.sort(np.random.randint(0, nsteps, count))
        last = -1
        k = 0
        while k < count:
            tick = bins[k]
            gg, cc, vv = gap(g[j], ca[j], velocity[j], tick - last - 1, dt, write_tau, forget_tau, amplitude, gthr)
            same = 1
            while k + same < count and bins[k + same] == tick:
                same += 1
            gg = gg * eg + 0.2 * same
            cc, vv = ca_constant(cc, vv, dt, write_tau if gg > gthr else forget_tau, amplitude if gg > gthr else 0.0)
            g[j] = gg
            ca[j] = min(max(cc, 0.0), amplitude)
            velocity[j] = vv if 0 < cc < amplitude else 0.0
            last = tick
            k += same
        gg, cc, vv = gap(g[j], ca[j], velocity[j], nsteps - last - 1, dt, write_tau, forget_tau, amplitude, gthr)
        g[j] = gg
        ca[j] = cc
        velocity[j] = vv


class ProcessMemory:
    def __init__(self, config):
        self.c = config
        assert config.dt == 0.001 and config.tau_write > 0 and config.tau_forget > 0
        self.mask = (
            np.ones((config.cells, config.neurons), bool)
            if config.connectivity == "full"
            else territories(config.neurons, config.cells, config.overlap, config.connectivity_seed)
        )
        self.indices = [np.flatnonzero(m) for m in self.mask]
        self.taus = (
            config.tau_forget * np.geomspace(1 / config.tau_span, config.tau_span, config.cells)
            if config.heterogeneous
            else np.full(config.cells, config.tau_forget)
        )
        shape = (config.cells, len(self.indices[0]))
        self.g = np.zeros(shape)
        self.ca = np.zeros(shape)
        self.velocity = np.zeros(shape)

    def active(self):
        return np.clip((self.ca - self.c.threshold) / max(self.c.amplitude - self.c.threshold, 1e-8), 0, 1)

    def write(self, activity, seed):
        x = np.asarray(activity, float)
        if x.shape != (self.c.neurons,) or not np.isfinite(x).all() or (x < 0).any():
            raise ValueError("expected nonnegative RSNN activity")
        for k, ix in enumerate(self.indices):
            local_seed = int(np.random.SeedSequence([int(seed), k, 431]).generate_state(1)[0])
            advance(
                self.g[k],
                self.ca[k],
                self.velocity[k],
                x[ix],
                local_seed,
                self.c.duration,
                self.c.dt,
                self.c.tau_write,
                self.taus[k],
                self.c.amplitude,
                self.c.gthr,
            )

    def read(self, query):
        q = np.asarray(query, float)
        single = q.ndim == 1
        q = q[None] if single else q
        if q.ndim != 2 or q.shape[1] != self.c.neurons or not np.isfinite(q).all() or (q < 0).any():
            raise ValueError("expected action x RSNN activity queries")
        local = np.stack(
            [q[:, ix] @ self.active()[k] / np.maximum(q[:, ix].sum(1), 1e-8) for k, ix in enumerate(self.indices)],
            axis=1,
        )
        mass = np.stack([q[:, ix].sum(1) for ix in self.indices], axis=1)
        # Same reference territory mass for full and local inputs. For five
        # full territories this is the original factorial FullMemory's 0.2.
        mass *= (self.c.neurons / self.c.cells) / len(self.indices[0])
        return (local[0], mass[0]) if single else (local, mass)

    def diagnostics(self):
        a = self.active()
        return dict(
            memory_active_fraction=float((a > 0).mean()),
            memory_level_mean=float(a.mean()),
            memory_saturated_fraction=float((a > 0.95).mean()),
        )

    def metadata(self):
        return dict(
            config=asdict(self.c),
            forgetting_taus=self.taus.tolist(),
            territories=self.mask.astype(int).tolist(),
            processes=int(self.g.size),
            state_variables_per_process=3,
        )


class TraceMemory:
    """Same pre-write/read protocol, with genuine EMA neuronal traces."""

    def __init__(self, config, gain=2.0):
        from .memory import NeuronalTraceBank

        self.c = config
        self.mask = (
            np.ones((config.cells, config.neurons), bool)
            if config.connectivity == "full"
            else territories(config.neurons, config.cells, config.overlap, config.connectivity_seed)
        )
        self.indices = [np.flatnonzero(m) for m in self.mask]
        self.banks = [NeuronalTraceBank(len(ix), config.tau_forget, gain) for ix in self.indices]

    def write(self, activity, seed=None):
        x = np.asarray(activity, float)
        if x.shape != (self.c.neurons,) or not np.isfinite(x).all() or (x < 0).any():
            raise ValueError("expected nonnegative RSNN activity")
        for bank, ix in zip(self.banks, self.indices):
            bank.advance(x[ix], self.c.duration)

    def read(self, query):
        q = np.asarray(query, float)
        single = q.ndim == 1
        q = q[None] if single else q
        if q.ndim != 2 or q.shape[1] != self.c.neurons or not np.isfinite(q).all() or (q < 0).any():
            raise ValueError("invalid query")
        local = np.stack([bank.familiarity(q[:, ix]) for bank, ix in zip(self.banks, self.indices)], axis=1)
        mass = (
            np.stack([q[:, ix].sum(1) for ix in self.indices], axis=1)
            * (self.c.neurons / self.c.cells)
            / len(self.indices[0])
        )
        return (local[0], mass[0]) if single else (local, mass)

    def diagnostics(self):
        a = np.concatenate([b.active_memory() for b in self.banks])
        return dict(
            memory_active_fraction=float((a > 0).mean()),
            memory_level_mean=float(a.mean()),
            memory_saturated_fraction=float((a > 0.95).mean()),
        )

    def metadata(self):
        return dict(
            backend="trace",
            config=asdict(self.c),
            processes=sum(len(ix) for ix in self.indices),
            territories=self.mask.astype(int).tolist(),
        )


class AstroMemory(TraceMemory):
    """Original single-timescale calcium, now driven by RSNN activity."""

    def __init__(self, config):
        from .memory import FamiliarityMemory

        self.c = config
        self.mask = (
            np.ones((config.cells, config.neurons), bool)
            if config.connectivity == "full"
            else territories(config.neurons, config.cells, config.overlap, config.connectivity_seed)
        )
        self.indices = [np.flatnonzero(m) for m in self.mask]
        params = dict(
            duration=config.duration,
            astro=dict(tau=config.tau_forget, amplitude=config.amplitude, gthr=config.gthr, threshold=config.threshold),
        )
        self.banks = [FamiliarityMemory(len(ix), "astro", params) for ix in self.indices]

    def write(self, activity, seed):
        x = np.asarray(activity, float)
        if x.shape != (self.c.neurons,) or not np.isfinite(x).all() or (x < 0).any():
            raise ValueError("expected nonnegative RSNN activity")
        for i, (bank, ix) in enumerate(zip(self.banks, self.indices)):
            local_seed = int(np.random.SeedSequence([int(seed), i, 431]).generate_state(1)[0])
            bank.write(x[ix], local_seed)

    def diagnostics(self):
        a = np.concatenate([b.active() for b in self.banks])
        return dict(
            memory_active_fraction=float((a > 0).mean()),
            memory_level_mean=float(a.mean()),
            memory_saturated_fraction=float((a > 0.95).mean()),
        )

    def metadata(self):
        result = super().metadata()
        result["backend"] = "astro"
        return result


class NullMemory:
    def diagnostics(self):
        return {}

    def metadata(self):
        return dict(backend="none", processes=0)


class Reliability:
    def __init__(self, config, error=None, count=None):
        self.c = config
        shape = (config.cells, 4)
        self.error = np.zeros(shape) if error is None else error.copy()
        self.count = np.zeros(shape) if count is None else count.copy()

    def confidence(self, local, mass):
        # Uncertain action comparisons attenuate ALL memory currents together.
        support = self.count / (self.count + self.c.reliability_prior_count)
        r = support / (1 + (self.error / self.c.reliability_tol) ** 2)
        certainty = float(np.min(r.mean(0)))
        contrast = float(np.ptp(local, axis=0).mean())
        noise = float(self.error.mean())
        query_valid = float(np.min(np.clip(mass / self.c.query_mass_floor, 0, 1).mean(1)))
        return float(np.clip(certainty * contrast / max(contrast + noise, 1e-8) * query_valid, 0, 1))

    def observe(self, action, predicted_familiarity, actual_familiarity):
        error = np.abs(np.asarray(predicted_familiarity) - np.asarray(actual_familiarity))
        alpha = self.c.reliability_alpha
        self.error[:, action] = (1 - alpha) * self.error[:, action] + alpha * error
        self.count[:, action] += 1
        return float(error.mean())
