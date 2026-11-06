"""Action-conditioned next-RSNN prediction and RL-trained astrocytic mixing."""

import numpy as np
import torch
from torch import nn
from .learner import Learner
from .prediction import ObservationPredictor
from .plasticity import eprop_vector, apply_core
from .local_memory import LocalAstroPopulation
from .memory import rsnn_activity_code
from .mlp import fingerprint


class HardSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, relative):
        ctx.save_for_backward(relative)
        return (relative >= 0).to(relative.dtype)

    @staticmethod
    def backward(ctx, grad):
        (relative,) = ctx.saved_tensors
        return grad / (1 + 6 * relative.abs()).square()


def motor_probabilities(head, features, gate, gain, influence=1.0):
    """Exact hard forward; detached resets/refractory, surrogate gate credit."""
    c = head.config
    dtype = gate.dtype
    raw = torch.as_tensor(features, dtype=dtype) @ torch.as_tensor(head.params["wo"].T, dtype=dtype) + torch.as_tensor(
        head.params["bo"], dtype=dtype
    )
    shape = raw.shape
    vm = torch.as_tensor(head.phase, dtype=dtype).expand(shape).clone()
    vi = torch.zeros_like(vm)
    cond = torch.zeros_like(vm)
    mr = torch.zeros_like(vm)
    ir = torch.zeros_like(vm)
    counts = torch.zeros_like(vm)
    alpha = np.exp(-0.003 / 0.02)
    warm = round(c.steps * 0.2)
    excit = c.bias + c.amplitude * torch.tanh(raw)
    threshold = torch.as_tensor(head.threshold, dtype=dtype)
    for tick in range(c.steps):
        vi = alpha * vi + (1 - alpha) * 1.12
        active = ((ir <= 0) & (tick >= warm)).to(dtype)
        si = HardSpike.apply(vi - 1.25 / (1 + gate * gain)) * active
        vi = vi * (1 - si.detach())
        ir = torch.where(si.detach() > 0, torch.full_like(ir, 6), torch.clamp(ir - 1, min=0))
        cond = 0.78 * cond + si
        vm = alpha * vm + (1 - alpha) * (excit - influence * c.inhibition_scale * cond)
        spike = HardSpike.apply(vm - threshold) * (mr <= 0).to(dtype)
        if tick >= warm:
            counts = counts + spike
        vm = vm * (1 - spike.detach())
        mr = torch.where(spike.detach() > 0, torch.full_like(mr, 3), torch.clamp(mr - 1, min=0))
    return torch.softmax(counts / ((c.steps - warm) * head.temperature), dim=-1)


def weighted_gate(local, logits, floor, contrast):
    w = floor / logits.numel() + (1 - floor) * torch.softmax(logits, dim=-1)
    familiarity = torch.sum(local * w, dim=-1)
    return torch.clamp((familiarity - familiarity.min(-1, keepdim=True).values) / contrast, 0, 1)


class AdaptiveAstroLearner(Learner):
    def __init__(self, *args, population_config, motor_options=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.c.input_encoding != "rbf" or self.is_mlp:
            raise ValueError("Modified astrocytes require RSNN with explicit RBF observations")
        self.population_config = population_config
        options = dict(motor_options or {})
        if options.pop("aggregation", "mixture") != "mixture" or options.pop("dt_ms", 3) != 3:
            raise ValueError("Adaptive mixture uses the original 3ms single-inhibitor motor")
        self.head.config.steps = options.pop("steps", 60)
        self.weight_floor = options.pop("weight_floor", 0.1)
        self.learn_mixture = options.pop("learn_weights", True)
        if options or not 0 <= self.weight_floor <= 1 or self.head.config.steps < 1:
            raise ValueError("Invalid adaptive-mixture motor options")
        self.weight_lr = self.hp.get("astro_weight_lr", 0.0003)
        # Keep total number of RSNN neurons and predictor hidden widths fixed.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed + 10119)
            self.model.predictor = ObservationPredictor(self.c.neurons, self.c.neurons, hidden=self.c.predictor_hidden)
            last = self.model.predictor.next_input
            nn.init.normal_(last.weight, std=0.005)
            mean = np.clip(self.model.center.numpy() / 0.2, 1e-4, 1 - 1e-4)
            with torch.no_grad():
                last.bias.copy_(torch.tensor(np.log(mean / (1 - mean)), dtype=last.bias.dtype))
        self.model.prediction_optimizer = torch.optim.Adam(self.model.predictor.parameters(), lr=self.hp["pred_lr"])
        self.astro_logits = nn.Parameter(torch.zeros(population_config.cells, dtype=torch.float64))
        self.weight_optimizer = torch.optim.Adam([self.astro_logits], lr=self.weight_lr)
        self.initial_policy_hash = self.policy_hash()
        self.initial_predictor_hash = self.predictor_hash()
        self.initial_weights = self.weights_hash()
        self.uniform_weights = False

    def parameters(self):
        base = super().parameters()
        if hasattr(self, "astro_logits"):
            base.append(self.astro_logits.detach().numpy())
        return base

    def weights(self, uniform=False):
        if uniform or self.uniform_weights or not self.learn_mixture:
            return np.full(self.population_config.cells, 1 / self.population_config.cells)
        return (
            (
                self.weight_floor / self.population_config.cells
                + (1 - self.weight_floor) * torch.softmax(self.astro_logits, 0)
            )
            .detach()
            .numpy()
        )

    def policy_hash(self):
        values = list(self.actor.p.values()) if self.is_mlp else list(self.head.params.values())
        if hasattr(self, "astro_logits"):
            values.append(self.astro_logits.detach().numpy())
        return fingerprint(values)

    def make_memory(self):
        return LocalAstroPopulation(self.c.neurons, self.memory_params, self.population_config)

    def activity_code(self, rates):
        mode = (
            "rate"
            if getattr(self, "episode_mode", "learned") == "background_off"
            else self.population_config.activity_code
        )
        return rsnn_activity_code(rates, self.model.center.numpy(), self.model.scale.numpy(), mode)

    def observation_code(self, observation):
        x = np.asarray(observation, dtype=np.float32)
        if x.shape != (self.c.observation_channels,) or not np.isfinite(x).all():
            raise ValueError("agent receives encoded RBF observations, not coordinates")
        return x.copy()

    def representation(self, observation, previous, training):
        obs = self.observation_code(observation)
        signal = self.input_coder.encode_features(obs, previous).numpy()
        z, ei, er = self.core.observe(signal, training, False)
        return dict(observation=obs, input_code=signal, z=z, ei=ei if training else None, er=er if training else None)

    def begin_episode(self, training, mode="learned"):
        if mode not in ("learned", "uniform", "off", "background_off"):
            raise ValueError("unsupported adaptive-mixture evaluation mode")
        self.reset()
        self.episode_mode = mode
        self.memory_on = mode != "off"
        self.uniform_weights = mode == "uniform"
        self.memory = self.make_memory()
        self.memory_version = 0

    def write_real(self, rates, seed):
        if self.memory_on:
            self.memory.write(self.activity_code(rates), seed)
            self.memory_version += 1

    def decision(self, rep, predicted):
        from .memory import gating

        local = np.zeros((self.c.n_actions, self.population_config.cells))
        if self.memory_on:
            _, local = self.memory.familiarity(self.activity_code(predicted))
            gate = gating(local @ self.weights(self.uniform_weights), self.memory_params)
        else:
            gate = np.zeros(self.c.n_actions)
        with torch.no_grad():
            feature = self.model.features(torch.tensor(rep["z"], dtype=torch.float32)).numpy().astype(float)
        if self.kind == "random":
            p = np.full(self.c.n_actions, 1 / self.c.n_actions)
            cache = None
            diag = dict(mean_gate=0.0, motor_silent_fraction=0.0)
        else:
            p, cache, diag = self.head.forward(feature, gate, self.memory_params["gate"]["gain"], float(self.memory_on))
        off = self.head.forward(feature, np.zeros(self.c.n_actions), self.memory_params["gate"]["gain"], 0.0)[0]
        diag.update(mean_gate=float(gate.mean()), **self.memory.diagnostics())
        context = dict(local=local.copy(), version=self.memory_version)
        return p, cache, np.zeros((self.population_config.cells, self.c.n_actions)), diag, context, off

    def observe_outcome(self, action, context, actual, seed):
        if context["version"] != self.memory_version:
            raise RuntimeError("outcome must be queried before the next real write")
        error = 0.0
        if self.memory_on:
            _, real = self.memory.familiarity(self.activity_code(actual)[None])
            error = float(np.abs(context["local"][action] - real[0]).mean())
            self.write_real(actual, seed)
        return error

    def finish_episode(self, training):
        pass

    def prediction(self, rep):
        with torch.no_grad():
            z = torch.tensor(rep["z"], dtype=torch.float32) * self.settings.prediction_scale
            return 0.2 * self.model.predictor(z.expand(4, -1), torch.arange(4)).numpy()

    def mixture_update(self, records):
        if not self.rl or not self.learn_mixture:
            return dict(astro_weight_gradient_norm=0.0, astro_weight_step_norm=0.0)
        features = np.array([r["head_cache"][0] for r in records])
        local = np.array([r["local_familiarity"] for r in records])
        old = self.astro_logits.detach().clone()
        gate = weighted_gate(
            torch.tensor(local, dtype=torch.float64),
            self.astro_logits,
            self.weight_floor,
            self.memory_params["gate"]["contrast"],
        )
        p = motor_probabilities(self.head, features, gate, self.memory_params["gate"]["gain"])
        # Forward parity is necessary for an on-policy score-function update.
        original = np.array([r["probability"] for r in records])
        assert np.max(np.abs(p.detach().numpy() - original)) < 1e-10
        actions = torch.tensor([r["action"] for r in records], dtype=torch.long)
        ret = []
        carry = 0.0
        for r in reversed(records):
            carry = r["reward"] + 0.99 * carry
            ret.append(carry)
        ret = np.array(ret[::-1])
        adv = torch.tensor(ret - self.baseline[: len(records)], dtype=torch.float64)
        discount = torch.tensor(0.99 ** np.arange(len(records)) / self.horizon, dtype=torch.float64)
        lp = torch.log(p.clamp_min(1e-30))
        entropy = -(p * lp).sum(-1)
        objective = (discount * (adv * lp[torch.arange(len(records)), actions] + self.entropy * entropy)).sum()
        self.weight_optimizer.zero_grad(set_to_none=True)
        (-objective).backward()
        norm = float(self.astro_logits.grad.norm())
        torch.nn.utils.clip_grad_norm_([self.astro_logits], 1.0)
        self.weight_optimizer.step()
        with torch.no_grad():
            self.astro_logits -= self.astro_logits.mean()
        return dict(
            astro_weight_gradient_norm=norm, astro_weight_step_norm=float((self.astro_logits.detach() - old).norm())
        )

    def update(self, records):
        # Compute gate-credit before the head/baseline change, with everything
        # except astro_logits detached. Original update preserves RL/core split.
        diagnostic = self.mixture_update(records)
        out = super().update(records)
        out.update(diagnostic)
        w = self.weights()
        out.update(astro_effective_count=float(1 / (w @ w)), astro_weights=w.tolist())
        return out

    def _prediction_loss(self, z, records):
        a = torch.tensor([r["action"] for r in records], dtype=torch.long)
        target = torch.tensor(np.array([r["next_z"] for r in records]), dtype=torch.float32).detach()
        predicted = 0.2 * self.model.predictor(z * self.settings.prediction_scale, a)
        # Fixed calibration prevents high-rate neurons alone setting the scale.
        scale = (1 / self.model.scale).clamp_min(0.002)
        return ((predicted - target) / scale).square().mean()

    def _update_snn_prediction(self, records, policy_gradient=None):
        assert policy_gradient is None
        m = self.model
        s = self.settings
        z = torch.tensor(np.array([r["z"] for r in records]), dtype=torch.float32, requires_grad=True)
        m.prediction_optimizer.zero_grad(set_to_none=True)
        loss = self._prediction_loss(z, records)
        loss.backward()
        gradient = eprop_vector(
            z.grad.detach().numpy(),
            np.array([r["ei"] for r in records]),
            np.array([r["er"] for r in records]),
            m.rnn.rec_mask.numpy(),
        )
        norm = float(np.linalg.norm(gradient))
        before = np.concatenate([p.detach().numpy().ravel().copy() for p in m.rnn.parameters()])
        if np.any(gradient):
            apply_core(m, m.core_optimizer, gradient, s.core_step_cap)
        after = np.concatenate([p.detach().numpy().ravel() for p in m.rnn.parameters()])
        torch.nn.utils.clip_grad_norm_(m.predictor.parameters(), 5.0)
        m.prediction_optimizer.step()
        self.replay.extend({k: r[k] for k in ("z", "action", "next_z")} for r in records)
        self.replay = self.replay[-6000:]
        for _ in range(s.prediction_epochs - 1):
            batch = [self.replay[i] for i in self.replay_rng.integers(len(self.replay), size=128)]
            zz = torch.tensor(np.array([r["z"] for r in batch]), dtype=torch.float32)
            m.prediction_optimizer.zero_grad(set_to_none=True)
            extra = self._prediction_loss(zz, batch)
            extra.backward()
            torch.nn.utils.clip_grad_norm_(m.predictor.parameters(), 5.0)
            m.prediction_optimizer.step()
        return dict(
            prediction_loss=float(loss.detach()),
            next_activity_normalized_mse=float(loss.detach()),
            core_step_norm=float(np.linalg.norm(after - before)),
            prediction_core_gradient_norm=norm,
            prediction_head_updated=True,
        )

    def metadata(self):
        result = super().metadata()
        result.update(
            prediction_target=f"next actual{self.c.neurons}-neuron RSNN rates, detached target",
            memory_input="real RSNN activity; read-only action-conditioned RSNN forecasts",
            prediction_objective="single fixed-calibration-scaled latent MSE",
            mixture="global logits, softmax floor0.1, RL-only",
            virtual_rollout=False,
        )
        if hasattr(self, "astro_logits"):
            result["trainable_parameter_count"] = result["parameter_count"] - (
                sum(x.size for x in self.head.params.values()) + self.population_config.cells if not self.rl else 0
            )
            result["policy_parameter_count"] = (
                sum(v.size for v in self.head.params.values()) + self.population_config.cells
            )
        if self.rl and not self.learn_mixture:
            result["trainable_parameter_count"] -= self.population_config.cells
        result.update(astrocytes=self.population_config.cells, inhibitory_neurons=4, motor_dt_ms=3, rsnn_dt_ms=1)
        return result
