"""Direct RSNN prediction, configurable astrocytes and 4/20 inhibitory LIFs.

Full/local input, overlap, cell count and timescales are parameters of one
implementation. Calcium and trace store per-channel activity; Whole/RFF store
whole-pattern familiarity and therefore share one read across motor paths.
"""

import numpy as np
import torch
from torch import nn
from .adaptive import AdaptiveAstroLearner
from .learner import Learner
from .local_memory import PopulationConfig
from .population_memory import MemoryConfig, ProcessMemory, TraceMemory, AstroMemory, NullMemory, Reliability
from .pool_motor import PoolHead
from .dense_memory import DenseConfig, WholePattern, RandomFeatureMemory
from .memory import rsnn_activity_code


class PopulationLearner(AdaptiveAstroLearner):
    def __init__(self, *args, memory_options=None, motor_options=None, **kwargs):
        memory_options = dict(memory_options or {})
        motor_options = dict(motor_options or {})
        self.backend = memory_options.pop("backend", "calcium")
        self.code = memory_options.pop("activity_code", "excess")
        self.kernel_options = memory_options.pop("kernel", {})
        self.learn_paths = motor_options.pop("learn_paths", True)
        self.reliability_enabled = motor_options.pop("reliability", True)
        aggregation = motor_options.pop("aggregation", "pool")
        window = motor_options.pop("steps", 180)
        if motor_options:
            raise ValueError(f"unknown motor options: {sorted(motor_options)}")
        if self.backend not in ("calcium", "astro", "trace", "whole", "rff", "none"):
            raise ValueError("unknown familiarity backend")
        if self.code not in ("rate", "excess", "signed"):
            raise ValueError("unknown activity code")
        if self.backend in ("whole", "rff") and self.code != "signed":
            raise ValueError("dense familiarity needs signed, standardized rates")
        if self.backend in ("calcium", "astro", "trace") and self.code == "signed":
            raise ValueError("calcium and trace require nonnegative rates")
        cells = memory_options.get("cells", 5)
        super().__init__(*args, population_config=PopulationConfig(cells=cells, activity_code=self.code), **kwargs)
        del self.astro_logits
        del self.weight_optimizer
        if self.kind in ("noastro", "random"):
            self.backend = "none"
            self.learn_paths = False
        if self.backend == "none":
            self.learn_paths = False
        self.path_logits = (
            nn.Parameter(torch.full((cells, self.c.n_actions), np.log(0.75), dtype=torch.float64))
            if self.learn_paths
            else None
        )
        self.path_optimizer = (
            torch.optim.Adam([self.path_logits], lr=self.hp.get("path_lr", 0.001)) if self.learn_paths else None
        )
        self.rel_error = np.zeros((cells, self.c.n_actions))
        self.rel_count = np.zeros_like(self.rel_error)
        self.head = PoolHead(self.c.neurons, self.head.config, aggregation=aggregation)
        self.head.config.steps = window
        physical = dict(
            neurons=self.c.neurons,
            tau_write=self.hp.get("memory_write_tau", 0.1),
            tau_forget=self.hp["memory_tau"],
            gthr=self.hp["memory_glutamate_threshold"],
            threshold=self.hp["memory_threshold"],
            reliability_tol=self.hp.get("reliability_tol", 0.1),
        )
        physical.update(memory_options)
        self.memory_config = MemoryConfig(**physical)
        if self.memory_config.neurons != self.c.neurons:
            raise ValueError("memory/RSNN dimensions differ")
        if self.c.n_actions != 4 or window < 1:
            raise ValueError("four-action motor with a positive window required")
        self.initial_policy_hash = self.policy_hash()
        self.initial_predictor_hash = self.predictor_hash()
        self.initial_weights = self.weights_hash()

    def parameters(self):
        result = Learner.parameters(self)
        if getattr(self, "path_logits", None) is not None:
            result.append(self.path_logits.detach().numpy())
        return result

    def policy_hash(self):
        from .mlp import fingerprint

        values = list(self.head.params.values()) if not self.is_mlp else list(self.actor.p.values())
        if getattr(self, "path_logits", None) is not None:
            values.append(self.path_logits.detach().numpy())
        return fingerprint(values)

    def weights_hash(self):
        from .mlp import fingerprint

        values = self.parameters()
        if hasattr(self, "rel_error"):
            values += [self.rel_error, self.rel_count]
        return fingerprint(values)

    def gains(self, uniform=False):
        shape = (self.memory_config.cells, self.c.n_actions)
        if uniform or self.path_logits is None:
            return np.ones(shape), np.zeros(shape)
        s = torch.sigmoid(self.path_logits).detach().numpy()
        return 0.25 + 1.75 * s, 1.75 * s * (1 - s)

    def activity_code(self, rates):
        mode = "rate" if self.episode_mode == "background_off" else self.code
        return rsnn_activity_code(rates, self.model.center.numpy(), self.model.scale.numpy(), mode)

    def make_memory(self):
        if self.backend == "none":
            return NullMemory()
        if self.backend in ("whole", "rff"):
            cfg = DenseConfig(neurons=self.c.neurons, cells=self.memory_config.cells, **self.kernel_options)
            return (WholePattern if self.backend == "whole" else RandomFeatureMemory)(cfg)
        if self.backend == "trace":
            return TraceMemory(self.memory_config, self.hp.get("memory_gain", 2.0))
        if self.backend == "astro":
            return AstroMemory(self.memory_config)
        return ProcessMemory(self.memory_config)

    def begin_episode(self, training, mode="learned"):
        if mode not in ("learned", "uniform", "off", "confidence_off", "background_off", "pooled_control"):
            raise ValueError("unknown evaluation mode")
        if mode == "background_off" and self.code == "signed":
            raise ValueError("background_off is not a signed-kernel ablation")
        self.reset()
        self.episode_mode = mode
        self.memory_on = mode != "off" and self.backend != "none"
        self.memory = self.make_memory()
        self.reliability = Reliability(self.memory_config, self.rel_error, self.rel_count)
        self.memory_version = 0

    def decision(self, rep, predicted):
        shape = (self.c.n_actions, self.memory_config.cells)
        local = np.zeros(shape)
        confidence = 0.0
        if self.memory_on:
            local, mass = self.memory.read(self.activity_code(predicted))
            evidence = np.clip((local - local.min(0)) / max(self.memory_params["gate"]["contrast"], 1e-8), 0, 1).T
            confidence = (
                1.0
                if self.episode_mode == "confidence_off" or not self.reliability_enabled
                else self.reliability.confidence(local, mass)
            )
        else:
            evidence = np.zeros(shape[::-1])
        gains, dgains = self.gains(self.episode_mode == "uniform")
        if self.episode_mode == "pooled_control":
            f = local.mean(1)
            e = np.clip((f - f.min()) / max(self.memory_params["gate"]["contrast"], 1e-8), 0, 1)
            evidence = np.repeat(e[None], self.memory_config.cells, axis=0)
            gains = np.repeat(gains.mean(0)[None], self.memory_config.cells, axis=0)
            dgains = np.zeros_like(gains)
        with torch.no_grad():
            feature = self.model.features(torch.tensor(rep["z"], dtype=torch.float32)).numpy().astype(float)
        p, cache, elig, diag = self.head.forward(
            feature, evidence, gains, dgains, confidence, self.memory_params["gate"]["gain"]
        )
        off = self.head.forward(
            feature, np.zeros_like(evidence), gains, np.zeros_like(dgains), 0.0, self.memory_params["gate"]["gain"]
        )[0]
        if self.kind == "random":
            p = np.full(self.c.n_actions, 1 / self.c.n_actions)
            off = p.copy()
        diag.update(self.memory.diagnostics())
        diag.update(
            path_gain_mean=float(gains.mean()),
            path_gain_std=float(gains.std()),
            query_foreground_mean=float(self.activity_code(predicted).mean()),
            write_foreground_mean=float(self.activity_code(rep["z"]).mean()),
        )
        return p, cache, elig, diag, dict(local=local.copy(), version=self.memory_version), off

    def observe_outcome(self, action, context, actual, seed):
        if context["version"] != self.memory_version:
            raise RuntimeError("real familiarity must be read before writing outcome")
        error = 0.0
        if self.memory_on:
            actual_f, _ = self.memory.read(self.activity_code(actual))
            error = self.reliability.observe(action, context["local"][action], actual_f)
            self.write_real(actual, seed)
        return error

    def finish_episode(self, training):
        if training and self.memory_on:
            self.rel_error[:] = self.reliability.error
            self.rel_count[:] = self.reliability.count

    def path_update(self, records):
        if not self.rl or not self.learn_paths:
            return dict(path_gradient_norm=0.0, path_step_norm=0.0)
        carry = 0.0
        returns = []
        for record in reversed(records):
            carry = record["reward"] + 0.99 * carry
            returns.append(carry)
        gradient = np.zeros_like(self.rel_error)
        for t, (r, value) in enumerate(zip(records, reversed(returns))):
            p = r["probability"]
            lp = np.log(np.maximum(p, 1e-30))
            score = -p.copy()
            score[r["action"]] += 1
            credit = (
                0.99**t / self.horizon * ((value - self.baseline[t]) * score - self.entropy * p * (lp - np.sum(p * lp)))
            )
            gradient += r["path_eligibility"] * credit[None]
        before = self.path_logits.detach().clone()
        self.path_optimizer.zero_grad(set_to_none=True)
        self.path_logits.grad = torch.tensor(-gradient, dtype=torch.float64)
        norm = float(np.linalg.norm(gradient))
        torch.nn.utils.clip_grad_norm_([self.path_logits], 1.0)
        self.path_optimizer.step()
        return dict(path_gradient_norm=norm, path_step_norm=float((self.path_logits.detach() - before).norm()))

    def update(self, records):
        diagnostic = self.path_update(records)
        result = Learner.update(self, records)
        result.update(diagnostic)
        gains, _ = self.gains()
        result.update(path_gain_mean=float(gains.mean()), path_gain_std=float(gains.std()), path_gains=gains.tolist())
        return result

    def metadata(self):
        result = Learner.metadata(self)
        result.update(
            prediction_target="next real RSNN activity, stop-gradient target",
            memory_input="actual RSNN rates only; read-only predicted RSNN queries",
            prediction_objective="single fixed-scale next-activity MSE",
            memory_backend=self.backend,
            activity_code=self.code,
            cells=self.memory_config.cells,
            connectivity=self.memory_config.connectivity,
            aggregation=self.head.aggregation,
            inhibitory_neurons=4 * (self.memory_config.cells if self.head.aggregation == "pool" else 1),
            memory_connected=self.backend != "none",
            motor_dt_ms=1,
            rsnn_dt_ms=1,
            astrocytes=self.memory_config.cells if self.backend in ("astro", "calcium") else 0,
            reliability_enabled=self.reliability_enabled,
            local_path_parameters=self.memory_config.cells * 4 if self.learn_paths else 0,
            virtual_rollout=False,
            memory=self.make_memory().metadata(),
        )
        policy = (
            0
            if self.kind == "random"
            else sum(v.size for v in self.head.params.values())
            + (self.memory_config.cells * 4 if self.learn_paths else 0)
        )
        result["policy_parameter_count"] = policy
        result["trainable_parameter_count"] = result["parameter_count"] - (
            policy if not self.rl and self.kind != "random" else 0
        )
        return result
