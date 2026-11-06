"""Scientific invariants of the integrated RSNN-memory and inhibitory models."""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np
import torch
from astrorl.dense_memory import DenseConfig, WholePattern, RandomFeatureMemory
from astrorl.memory import rsnn_activity_code
from astrorl.motor import MotorConfig
from astrorl.pool_motor import PoolHead, pooled_window
from astrorl.population_memory import MemoryConfig, ProcessMemory, TraceMemory
from experiments.composition import ConfigStore, load_experiment
from experiments.models import build_model
from experiments.modified import make_tasks, run_episode

ROOT = Path(__file__).resolve().parents[1]


def first_task(name="smoke_pool", group="rl"):
    cfg, _ = load_experiment("experiments/" + name + ".json")
    return next(t for t in make_tasks(cfg) if t["group"] == group)


class ComposedConfigurationTests(unittest.TestCase):
    def test_map_cluster_uncertainty_does_not_treat_seeds_as_maps(self):
        from experiments.modified_analysis import METRICS, clustered_summary

        records = [
            dict(map=m, seed=s, **{k: value for k in METRICS}) for m, value in [("a", 0.2), ("b", 0.8)] for s in [1, 2]
        ]
        settings = dict(seed=20261104, bootstrap_draws=500, interval=0.95)
        result = clustered_summary(records, settings)
        self.assertAlmostEqual(result["reward01"], 0.5)
        self.assertLess(result["reward01_lo"], result["reward01_hi"])
        self.assertEqual(result["maps"], 2)
        self.assertEqual(result["seeds_per_map"], 2)
        with self.assertRaises(ValueError):
            clustered_summary(records[:-1], settings)

    def test_original_many_map_budgets_and_factorial_controls(self):
        for name, count, episodes in [("mdp_factorial", 1440, 300), ("mdp_pool", 720, 250), ("mdp_adaptive", 720, 250)]:
            cfg, hashes = load_experiment("experiments/" + name + ".json")
            tasks = make_tasks(cfg)
            self.assertEqual(len(tasks), count)
            self.assertTrue(all(t["episodes"] == episodes and t["eval_episodes"] == 100 for t in tasks))
            self.assertEqual({t["seed"] for t in tasks}, {72000, 72001, 72002})
            self.assertEqual(len({t["map"]["name"] for t in tasks}), 60)
            self.assertTrue(any(k.startswith("maps/") for k in hashes))
            self.assertTrue(any(k.startswith("architectures/") for k in hashes))
            if name == "mdp_factorial":
                self.assertEqual({t["task_mode"] for t in tasks}, {"ordered", "unordered"})
                for family in cfg["protocol"]["families"]:
                    self.assertEqual(
                        len({json.dumps(t["hp"], sort_keys=True) for t in tasks if t["family"] == family}), 1
                    )

    def test_reference_cycles_and_escape_fail(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "a.json").write_text('{"extends":"b.json"}')
            (root / "b.json").write_text('{"extends":"a.json"}')
            with self.assertRaisesRegex(ValueError, "Cyclic"):
                ConfigStore(root).load("a.json")
            with self.assertRaises(ValueError):
                ConfigStore(root).load("../other.json")

    def test_full_and_local_are_parameters_of_one_implementation(self):
        cfg, _ = load_experiment("experiments/mdp_factorial.json")
        self.assertEqual({m["learner"] for m in cfg["models"].values()}, {"population"})
        self.assertEqual({m["memory"]["connectivity"] for m in cfg["models"].values()}, {"local", "full"})
        self.assertEqual({m["motor"]["aggregation"] for m in cfg["models"].values()}, {"pool", "single"})


class PopulationMemoryTests(unittest.TestCase):
    def test_territories_capacity_timescales_and_read_only_queries(self):
        x = np.linspace(0, 1, 100)
        query = np.tile(x, (4, 1))
        for connectivity, shape in [("local", (5, 20)), ("full", (5, 100))]:
            memory = ProcessMemory(MemoryConfig(connectivity=connectivity, heterogeneous=True))
            self.assertEqual(memory.g.shape, shape)
            self.assertEqual(len(set(memory.taus)), 5)
            memory.write(x, 811)
            before = [v.copy() for v in (memory.g, memory.ca, memory.velocity)]
            f, m = memory.read(query)
            self.assertEqual(f.shape, (4, 5))
            self.assertEqual(m.shape, (4, 5))
            for a, b in zip(before, (memory.g, memory.ca, memory.velocity)):
                np.testing.assert_array_equal(a, b)
            if connectivity == "full":
                np.testing.assert_allclose(m, query.sum(1)[:, None] * np.ones((1, 5)) * 0.2)
            with self.assertRaises(ValueError):
                memory.write(-x, 1)
        overlap = ProcessMemory(MemoryConfig(cells=10, overlap=2))
        self.assertEqual(overlap.g.shape, (10, 20))
        np.testing.assert_array_equal(overlap.mask.sum(0), np.full(100, 2))

    def test_trace_is_an_actual_ema(self):
        c = MemoryConfig(cells=1, connectivity="full", tau_forget=2.0)
        memory = TraceMemory(c, gain=0.5)
        x = np.ones(100)
        memory.write(x)
        expected = 0.5 * (1 - np.exp(-0.5))
        np.testing.assert_allclose(memory.read(x)[0], [expected])

    def test_fixed_normalization_and_signed_queries(self):
        z = np.array([0.1, 0.5, 1.0])
        center = np.array([0.2, 0.2, 0.2])
        scale = np.array([2.0, 3.0, 4.0])
        np.testing.assert_allclose(
            rsnn_activity_code(z, center, scale, "excess"), np.clip((z - center) * scale, 0, 3) / 3
        )
        np.testing.assert_allclose(rsnn_activity_code(z, center, scale, "signed"), (z - center) * scale / np.sqrt(3))
        np.testing.assert_array_equal(center, [0.2, 0.2, 0.2])

    def test_dense_familiarity_preserves_whole_combinations(self):
        x = np.array([[1.0, 1, 0, 0], [0.0, 0, 1, 1], [1.0, 0, 1, 0]])
        u = np.array([0.1, 0.3, 0.6, 0.7])
        u /= np.linalg.norm(u)
        rotation = np.eye(4) - 2 * np.outer(u, u)
        scores = []
        for data in (x, x @ rotation):
            memory = WholePattern(DenseConfig(neurons=4, cells=1))
            memory.write(data[0])
            memory.write(data[1])
            scores.append(memory.score(data))
            self.assertGreater(scores[-1][1], scores[-1][2])
        self.assertTrue(np.all(np.abs(x @ rotation) > 1e-4))
        np.testing.assert_allclose(*scores, atol=1e-12)
        rff = RandomFeatureMemory(DenseConfig(neurons=4, cells=1, features=4096))
        data = x @ rotation
        for z in data[:2]:
            rff.write(z)
        weights = np.array([np.exp(-1 / rff.c.decay_steps), 1.0])
        weights /= weights.sum()
        exact = rff.phi(data) @ rff.phi(data[:2]).T @ weights
        np.testing.assert_allclose(rff.raw_score(data), exact, atol=1e-12)
        self.assertGreater(rff.score(data)[1], rff.score(data)[2])


class InhibitoryPoolTests(unittest.TestCase):
    def test_pool_and_single_equal_when_territorial_signals_are_equal(self):
        config = MotorConfig(steps=180, bias=1.45, amplitude=1.0)
        pooled = PoolHead(100, config, aggregation="pool")
        single = PoolHead(100, copy.deepcopy(config), aggregation="single")
        feature = np.ones(100) / 10
        e = np.tile([0.1, 0.3, 0.5, 0.8], (5, 1))
        g = np.ones((5, 4))
        dg = g * 0.2
        p, _, ep, dp = pooled.forward(feature, e, g, dg, 0.7, 2.0)
        q, _, es, ds = single.forward(feature, e, g, dg, 0.7, 2.0)
        np.testing.assert_array_equal(p, q)
        self.assertAlmostEqual(dp["inhibitory_current_mean"], ds["inhibitory_current_mean"], 12)
        np.testing.assert_allclose(ep.sum(0), es.sum(0))

    def test_pooling_does_not_commute_with_inhibitory_nonlinearity(self):
        e = np.zeros((5, 4))
        e[0, 0] = 0.9
        e[:, 1] = 0.18

        def currents(evidence):
            return pooled_window(
                np.zeros(4),
                evidence,
                np.ones_like(evidence),
                np.ones_like(evidence),
                1.0,
                np.ones(4),
                np.zeros(4),
                1.45,
                1.0,
                1.0,
                1.0,
                180,
                4.0,
            )[4]

        pool = currents(e)
        single = currents(e.mean(0)[None])
        self.assertAlmostEqual(single[0], single[1], 12)
        self.assertNotAlmostEqual(pool[0], pool[1], 8)


class ModifiedLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_actual_state_targets_reused_and_all_queries_are_read_only(self):
        task = first_task()
        learner = build_model(task)
        row, records, _, _ = run_episode(learner, task, 1, 1, True)
        self.assertTrue(records)
        for current, nxt in zip(records, records[1:]):
            np.testing.assert_array_equal(current["next_z"], nxt["z"])
        self.assertEqual(learner.memory_version, len(records) + 1)
        before = [a.copy() for a in (learner.memory.g, learner.memory.ca, learner.memory.velocity)]
        version = learner.memory_version
        pred = learner.prediction(records[-1])
        learner.decision(records[-1], pred)
        learner.decision(records[-1], pred)
        self.assertEqual(learner.memory_version, version)
        for a, b in zip(before, (learner.memory.g, learner.memory.ca, learner.memory.velocity)):
            np.testing.assert_array_equal(a, b)
        with self.assertRaises(ValueError):
            learner.representation(np.array([0.2, 0.3]), None, False)
        learner.begin_episode(False)
        self.assertEqual(learner.memory_version, 0)
        np.testing.assert_array_equal(learner.memory.ca, 0)

    def test_predictor_uses_real_next_activity_and_detached_target(self):
        task = first_task()
        learner = build_model(task)
        _, records, _, _ = run_episode(learner, task, 1, 1, True)
        z = torch.tensor(np.array([r["z"] for r in records]), dtype=torch.float32, requires_grad=True)
        loss = learner._prediction_loss(z, records)
        action = torch.tensor([r["action"] for r in records])
        target = torch.tensor(np.array([r["next_z"] for r in records]), dtype=torch.float32)
        expected = (
            (
                (0.2 * learner.model.predictor(z * learner.settings.prediction_scale, action) - target)
                / (1 / learner.model.scale).clamp_min(0.002)
            )
            .square()
            .mean()
        )
        torch.testing.assert_close(loss, expected, atol=0, rtol=0)
        loss.backward()
        self.assertGreater(float(z.grad.norm()), 0.0)
        altered = [dict(r, next_input=np.full(64, -100.0)) for r in records]
        torch.testing.assert_close(learner._prediction_loss(z.detach(), altered), loss.detach(), atol=0, rtol=0)

    def test_rl_path_credit_does_not_update_core_or_predictor(self):
        learner = build_model(first_task())
        core = learner.backbone_hash()
        predictor = learner.predictor_hash()
        record = dict(probability=np.full(4, 0.25), action=0, reward=1.0, path_eligibility=np.ones((5, 4)))
        result = learner.path_update([record])
        self.assertGreater(result["path_step_norm"], 0.0)
        self.assertEqual(learner.backbone_hash(), core)
        self.assertEqual(learner.predictor_hash(), predictor)
        learner = build_model(first_task(group="no_rl"))
        before = learner.policy_hash()
        learner.path_update([record])
        self.assertEqual(before, learner.policy_hash())

    def test_checkpoint_metadata_and_training_invariants(self):
        task = first_task("smoke_factorial")
        learner = build_model(task)
        self.assertEqual(learner.horizon, 150)
        self.assertEqual(learner.metadata()["parameter_count"], 60176)
        learner.schedule(1, 2)
        _, records, _, _ = run_episode(learner, task, 1, 1, True)
        for current, nxt in zip(records, records[1:]):
            np.testing.assert_array_equal(current["next_z"], nxt["z"])
        update = learner.update(records)
        self.assertEqual(update["policy_core_gradient_norm"], 0.0)
        self.assertGreater(update["prediction_core_gradient_norm"], 0.0)
        frozen = learner.weights_hash()
        baseline = learner.baseline.copy()
        for mode in ("learned", "off", "uniform", "confidence_off", "background_off", "pooled_control"):
            run_episode(learner, task, 1, 8, False, mode)
            self.assertEqual(learner.weights_hash(), frozen)
            np.testing.assert_array_equal(learner.baseline, baseline)


if __name__ == "__main__":
    unittest.main()
