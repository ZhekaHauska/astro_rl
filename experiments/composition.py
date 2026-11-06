"""Compose experiment, model and environment configs without model-side IO."""

import copy
import hashlib
import json
from pathlib import Path
from .configuration import CONFIGS

REFERENCES = {
    "architecture",
    "memory",
    "memory_base",
    "motor",
    "parameters",
    "environment",
    "map_set",
    "protocol",
    "statistics",
}


def merge(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else copy.deepcopy(value)
        )
    return result


class ConfigStore:
    def __init__(self, root=CONFIGS):
        self.root = Path(root).resolve()
        self.hashes = {}

    def load(self, name, stack=()):
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root) or path.suffix != ".json":
            raise ValueError("Config references must point to JSON files inside configs/")
        if path in stack:
            raise ValueError(
                "Cyclic configuration reference: " + " -> ".join(str(p.relative_to(self.root)) for p in (*stack, path))
            )
        raw = path.read_bytes()
        self.hashes[str(path.relative_to(self.root))] = hashlib.sha256(raw).hexdigest()
        value = json.loads(raw)
        return self.resolve(value, (*stack, path))

    def resolve(self, value, stack):
        if isinstance(value, list):
            return [self.resolve(v, stack) for v in value]
        if not isinstance(value, dict):
            return value
        parent = self.load(value["extends"], stack) if "extends" in value else {}
        if not isinstance(parent, dict):
            raise ValueError("Only object configs support extends")
        child = {}
        for key, item in value.items():
            if key == "extends":
                continue
            if key == "models":
                child[key] = {
                    name: self.load(v, stack) if isinstance(v, str) else self.resolve(v, stack)
                    for name, v in item.items()
                }
            elif key in REFERENCES and isinstance(item, str):
                child[key] = self.load(item, stack)
            else:
                child[key] = self.resolve(item, stack)
        return merge(parent, child)


def load_experiment(name, root=CONFIGS):
    store = ConfigStore(root)
    config = store.load(name)
    for key in ("name", "models", "environment", "map_set", "protocol"):
        if key not in config:
            raise ValueError(f"Missing experiment component: {key}")
    p = config["protocol"]
    if not isinstance(p["workers"], int) or not 1 <= p["workers"] <= 30:
        raise ValueError("protocol.workers must be 1..30")
    stats = p["statistics"]
    if (
        stats["unit"] != "map_then_seed"
        or not isinstance(stats["bootstrap_draws"], int)
        or stats["bootstrap_draws"] < 1
        or not 0 < stats["interval"] < 1
    ):
        raise ValueError("Invalid map-cluster statistics configuration")
    if any(not isinstance(p[k], int) or p[k] < 1 for k in ("episodes", "eval_episodes", "maps_per_family")):
        raise ValueError("Positive integer training/evaluation/map counts required")
    if not p["seeds"] or not p["families"] or not p["groups"] or not config["models"]:
        raise ValueError("Empty experimental conditions")
    if set(p["groups"]) - {"rl", "no_rl"}:
        raise ValueError("Unknown learning group")
    for key in ("seeds", "families", "groups"):
        if len(p[key]) != len(set(p[key])):
            raise ValueError(f"Duplicate entries in protocol.{key}")
    if config["environment"]["task"] not in ("single_goal", "three_goal"):
        raise ValueError("Unknown environment task")
    env = config["environment"]
    if not env["modes"] or len(env["modes"]) != len(set(env["modes"])):
        raise ValueError("Empty or duplicate environment modes")
    if (
        not isinstance(env["max_steps"], int)
        or env["max_steps"] < 1
        or env["reward_goal"] <= 0
        or env["step_penalty"] < 0
    ):
        raise ValueError("Invalid environment horizon/rewards")
    if env["task"] == "three_goal" and (
        env["reward_goal"] != 1 or env["step_penalty"] != 0 or set(env["modes"]) - {"ordered", "unordered"}
    ):
        raise ValueError("The checkpoint protocol uses reward-once +1 and ordered/unordered modes")
    if env["task"] == "single_goal" and (env["modes"] != ["single"] or env["starts"] != "iid_uniform_non_goal"):
        raise ValueError("Single-goal protocol requires paired IID non-goal starts")
    for model in config["models"].values():
        c = model["architecture"]
        if c["input_encoding"] != "rbf":
            raise ValueError("Current MDP experiments require explicit RBF observations")
        required = 4 if config["environment"]["task"] == "three_goal" else 0
        if c["context_channels"] != required:
            raise ValueError("Model context width must match environment observations")
        if model["learner"] not in ("population", "adaptive", "q"):
            raise ValueError("Unknown learner implementation")
        if model["learner"] in ("population", "adaptive") and model["kind"] not in (
            "astro",
            "trace",
            "noastro",
            "random",
        ):
            raise ValueError("Current RSNN experiment profiles exclude MLP/unknown model kinds")
        if model["learner"] == "q" and required:
            raise ValueError("The historical Q baseline here is single-goal only")
        if model["learner"] == "q" and c["observation_channels"] != 64:
            raise ValueError("Historical Q baselines use 64 observation channels")
        if model["learner"] == "q" and model["kind"] not in ("q_table", "q_rnn"):
            raise ValueError("Current experiment profiles support Q-table and Q-RNN only")
        if model["learner"] == "adaptive" and "memory" in model:
            raise ValueError("Adaptive mixture config uses population, not path-memory options")
        if (
            model["learner"] == "population"
            and model["memory"]["backend"] in ("whole", "rff")
            and "background_off" in p["evaluation_modes"]
        ):
            raise ValueError("background_off is not a signed-kernel ablation")
    return config, dict(sorted(store.hashes.items()))
