"""Twenty independent inhibitory LIF neurons and four motor LIF neurons at1ms."""

import numpy as np
from numba import njit
from .motor import SpikingMotorHead


@njit(cache=True)
def pooled_window(
    raw,
    evidence,
    gains,
    gain_derivative,
    confidence,
    threshold,
    phase,
    bias,
    amplitude,
    inhibition_scale,
    gate_gain,
    steps,
    conductance_cap,
):
    k, n = evidence.shape
    dt = 0.001
    alpha = np.exp(-dt / 0.02)
    syn_decay = 0.78 ** (1 / 3)
    warm = int(round(0.2 * steps))
    vm = phase.copy()
    vi = np.zeros((k, n))
    cond = vi.copy()
    ic = vi.copy()
    ec = vi.copy()
    em = vi.copy()
    ir = np.zeros((k, n))
    mr = np.zeros(n)
    eraw = np.zeros(n)
    counts = np.zeros(n)
    derivative = np.zeros(n)
    local_derivative = vi.copy()
    current_sum = np.zeros(n)
    current_peak = np.zeros(n)
    excit = bias + amplitude * np.tanh(raw)
    drive = amplitude * (1 - np.tanh(raw) ** 2)
    for tick in range(steps):
        for a in range(n):
            for cell in range(k):
                gate = gate_gain * gains[cell, a] * evidence[cell, a]
                ti = 1.25 / (1 + gate)
                vi[cell, a] = alpha * vi[cell, a] + (1 - alpha) * 1.12
                active = tick >= warm and ir[cell, a] <= 0
                si = 1.0 if active and vi[cell, a] >= ti else 0.0
                psi = (1 / (1 + 6 * abs(vi[cell, a] - ti)) ** 2) if active else 0.0
                dsi = psi * 1.25 * gate_gain * evidence[cell, a] * gain_derivative[cell, a] / (1 + gate) ** 2
                cond[cell, a] = syn_decay * cond[cell, a] + si
                ec[cell, a] = syn_decay * ec[cell, a] + dsi
                ic[cell, a] += si
                vi[cell, a] *= 1 - si
                ir[cell, a] = 18 if si > 0 else max(0.0, ir[cell, a] - 1)
            total = cond[:, a].mean()
            bounded = min(total, conductance_cap)
            current = inhibition_scale * confidence * bounded
            current_sum[a] += current
            current_peak[a] = max(current_peak[a], current)
            vm[a] = alpha * vm[a] + (1 - alpha) * (excit[a] - current)
            eraw[a] = alpha * eraw[a] + (1 - alpha) * drive[a]
            for cell in range(k):
                em[cell, a] = alpha * em[cell, a] - (1 - alpha) * inhibition_scale * confidence * ec[cell, a] / k * (
                    total < conductance_cap
                )
            active = mr[a] <= 0
            sm = 1.0 if active and vm[a] >= threshold[a] else 0.0
            psi = (1 / (1 + 6 * abs(vm[a] - threshold[a])) ** 2) if active else 0.0
            if tick >= warm:
                counts[a] += sm
                derivative[a] += psi * eraw[a]
                for cell in range(k):
                    local_derivative[cell, a] += psi * em[cell, a]
            vm[a] *= 1 - sm
            eraw[a] *= 1 - sm
            for cell in range(k):
                em[cell, a] *= 1 - sm
            mr[a] = 9 if sm > 0 else max(0.0, mr[a] - 1)
    return counts, derivative, local_derivative, ic, current_sum / steps, current_peak


class PoolHead(SpikingMotorHead):
    def __init__(self, inputs, config, aggregation="pool"):
        super().__init__(inputs, config)
        if aggregation not in ("pool", "single"):
            raise ValueError("aggregation must be pool or single")
        self.aggregation = aggregation

    def forward(self, feature, evidence, gains, dgains, confidence, gate_gain=1.0):
        c = self.config
        raw = self.params["wo"] @ feature + self.params["bo"]
        evidence = np.asarray(evidence)
        if self.aggregation == "single":
            effective = (evidence * gains).mean(0)
            counts, derivative, elig, inh, mean_current, peak = pooled_window(
                raw,
                effective[None],
                np.ones((1, 4)),
                np.ones((1, 4)),
                confidence,
                self.threshold,
                self.phase,
                c.bias,
                c.amplitude,
                c.inhibition_scale,
                gate_gain,
                c.steps,
                4.0,
            )
            local = (
                np.divide(elig[0], effective, out=np.zeros(4), where=effective > 1e-12)[None]
                * evidence
                * dgains
                / len(evidence)
            )
        else:
            counts, derivative, local, inh, mean_current, peak = pooled_window(
                raw,
                evidence,
                gains,
                dgains,
                confidence,
                self.threshold,
                self.phase,
                c.bias,
                c.amplitude,
                c.inhibition_scale,
                gate_gain,
                c.steps,
                4.0,
            )
        # Original temperature converted to Hz; same144ms observation window.
        # Division retains the original literal .144 exactly at 180 steps;
        # multiplying 144 by .001 introduces an avoidable rounding difference.
        scale = ((c.steps - round(0.2 * c.steps)) / 1000.0) * (self.temperature / 0.003)
        logits = counts / scale
        p = np.exp(logits - logits.max())
        p /= p.sum()
        diag = dict(
            motor_silent_fraction=float((counts == 0).mean()),
            motor_counts=float(counts.mean()),
            inhibitory_counts=float(inh.mean()),
            mean_gate=float(evidence.mean()),
            inhibitory_current_mean=float(mean_current.mean()),
            inhibitory_current_peak=float(peak.max()),
            inhibitory_current_budget=float(4 * c.inhibition_scale),
            memory_confidence=float(confidence),
        )
        return p, (feature, derivative / scale), local / scale, diag
