"""Many-map MDP experiments using composable configurations and src models."""

import copy
import gzip
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
from astrorl.environment import geometry
from astrorl.logging import read_json, write_json
from .composition import load_experiment
from .models import build_model
from .rollouts import trajectory, schedule, action_seed
from .checkpoint_rollout import rollout


def identity(task):
    return hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()[:24]


def make_tasks(config):
    protocol = config["protocol"]
    env = config["environment"]
    dataset = config["map_set"]
    items = dataset.get("tasks", dataset.get("maps", []))
    tasks = []
    for family in protocol["families"]:
        maps = [
            v
            for v in items
            if v.get("map", v)["family"] == family and v.get("map", v)["split"] == protocol["map_split"]
        ][: protocol["maps_per_family"]]
        if len(maps) != protocol["maps_per_family"]:
            raise ValueError(
                f"Expected {protocol['maps_per_family']} maps for {family}/{protocol['map_split']}, found {len(maps)}"
            )
        for model, profile in config["models"].items():
            for group in protocol["groups"]:
                matches = dict(profile.get("parameter_match", {}), family=family, group=group)
                selected = [r for r in profile["parameters"] if all(r.get(k) == v for k, v in matches.items())]
                if len(selected) != 1:
                    raise ValueError(f"Expected one parameter set for {model}/{family}/{group}")
                hp = copy.deepcopy(selected[0]["hp"])
                hp.update(profile.get("parameter_overrides", {}))
                hp["horizon"] = env["max_steps"]
                model_config = {k: copy.deepcopy(v) for k, v in profile.items() if k != "parameters"}
                for item in maps:
                    if env["task"] == "three_goal" and len(item["checkpoints"]) != 3:
                        raise ValueError("The preserved factorial protocol requires exactly three checkpoints")
                    for mode in env["modes"]:
                        for seed in protocol["seeds"]:
                            tasks.append(
                                dict(
                                    experiment=config["name"],
                                    model=model,
                                    family=family,
                                    group=group,
                                    seed=seed,
                                    map=copy.deepcopy(item.get("map", item)),
                                    task_config=copy.deepcopy(item) if env["task"] == "three_goal" else None,
                                    task_mode=mode,
                                    environment=env,
                                    model_config=model_config,
                                    hp=hp,
                                    episodes=protocol["episodes"],
                                    eval_episodes=protocol["eval_episodes"],
                                    evaluation_modes=protocol["evaluation_modes"],
                                    seed_scheme=protocol["seed_scheme"],
                                    validation=protocol["validation"],
                                )
                            )
    return tasks


def mean(rows):
    keys = {k for row in rows for k, v in row.items() if isinstance(v, (float, int, np.number)) and k != "episode"}
    return {
        key: float(np.mean([r[key] for r in rows if isinstance(r.get(key), (float, int, np.number))]))
        for key in sorted(keys)
    }


def episode_seed(task, episode, phase):
    if task["seed_scheme"] == "three_goal_20261030":
        return int(np.random.SeedSequence([20261030, task["seed"], episode, phase]).generate_state(1)[0])
    if task["seed_scheme"] != "single_goal_20261023":
        raise ValueError("Unknown seed scheme")
    return action_seed(task["seed"], episode, phase)


def q_trajectory(learner, task, start, seed, training):
    spec = task["map"]
    env = task["environment"]
    free, index, goal, dist, tr = geometry(spec)
    state = index[tuple(start)]
    initial = dist[tuple(start)]
    rng = np.random.default_rng(seed)
    learner.reset()
    encoder = learner.input_coder.observation_encoder
    observation = lambda s: encoder.encode(np.asarray(free[s], dtype=np.float32) / (spec["size"] - 1))
    rep = learner.representation_features(observation(state), training)
    records = []
    actions = []
    visited = {state}
    occupancy = np.zeros((spec["size"],) * 2, np.int64)
    for tick in range(env["max_steps"]):
        p, _, _ = learner.policy(rep, np.zeros(4), False)
        action = min(3, int(np.searchsorted(p.cumsum(), rng.random(), side="right")))
        nxt = int(tr[state, action])
        hit = nxt == goal
        nr = learner.representation_features(observation(nxt), training)
        if training:
            records.append(
                dict(
                    observation=rep["observation"],
                    next_observation=nr["observation"],
                    action=action,
                    reward=float(hit) * env["reward_goal"] - env["step_penalty"],
                    terminal=hit or tick == env["max_steps"] - 1,
                )
            )
        occupancy[free[state]] += 1
        actions.append(action)
        visited.add(nxt)
        state = nxt
        rep = nr
        if hit:
            break
    steps = tick + 1
    success = float(state == goal)
    reward = success * env["reward_goal"] - env["step_penalty"] * steps
    row = dict(
        steps=steps,
        success=success,
        reward=reward,
        reward01=(reward + env["step_penalty"] * env["max_steps"])
        / (env["reward_goal"] + env["step_penalty"] * env["max_steps"] - env["step_penalty"] * initial),
        path_efficiency=success * initial / steps,
        coverage=len(visited) / len(free),
        optimal_steps=initial,
    )
    return row, records, np.array(actions, np.uint8), occupancy


def run_episode(learner, task, index, phase, training, mode="learned"):
    seed = episode_seed(task, index, phase)
    env = task["environment"]
    if env["task"] == "three_goal":
        row, recs, actions, occupancy = rollout(
            learner, task["task_config"], task["task_mode"], seed, training, memory_mode=mode, horizon=env["max_steps"]
        )
        row["reward01"] = row["normalized_reward"]
        return row, recs, actions, occupancy
    start = schedule(task["map"], task["seed"], max(task["episodes"], task["eval_episodes"]), phase)[index - 1]
    if task["model_config"]["learner"] == "q":
        return q_trajectory(learner, task, start, seed, training)
    row, recs, spatial, _ = trajectory(
        learner,
        task["map"],
        start,
        seed,
        training,
        mode,
        horizon=env["max_steps"],
        reward_goal=env["reward_goal"],
        step_penalty=env["step_penalty"],
    )
    return row, recs, spatial["actions"], spatial["occupancy"]


def prediction_probe(learner, spec, seed, contexts=32):
    before = learner.weights_hash()
    free, _, _, _, tr = geometry(spec)
    rng = np.random.default_rng(seed)
    errors = []
    persistence = []
    rates = []
    observation = lambda s: learner.input_coder.observation_encoder.encode(
        np.asarray(free[s], dtype=np.float32) / (spec["size"] - 1)
    )
    for _ in range(contexts):
        state = int(rng.integers(len(free)))
        learner.reset()
        rep = learner.representation(observation(state), None, False)
        for _ in range(4):
            pred = learner.prediction(rep)
            action = int(rng.integers(4))
            state = int(tr[state, action])
            nr = learner.representation(observation(state), action, False)
            errors.append(float(np.square(pred[action] - nr["z"]).mean()))
            persistence.append(float(np.square(rep["z"] - nr["z"]).mean()))
            rates.append(rep["z"].copy())
            rep = nr
    assert learner.weights_hash() == before
    return dict(
        mse=float(np.mean(errors)),
        persistence_mse=float(np.mean(persistence)),
        representation_variance=float(np.var(rates, axis=0).mean()),
    )


def gzip_json(path, value):
    with gzip.open(path, "wt") as f:
        json.dump(value, f, allow_nan=False)


def case(task, output):
    torch.set_num_threads(1)
    began = time.time()
    folder = Path(output) / "runs" / identity(task)
    if (folder / "result.json").exists():
        cached = read_json(folder / "result.json")
        if cached["task"] != task:
            raise RuntimeError("Cached task differs from requested task")
        return cached
    folder.mkdir(parents=True, exist_ok=True)
    learner = build_model(task)
    initial = learner.weights_hash()
    has_prediction = task["model_config"]["learner"] != "q"
    initial_core = learner.backbone_hash() if has_prediction else None
    initial_predictor = learner.predictor_hash() if has_prediction else None
    initial_policy = learner.policy_hash() if has_prediction else initial
    probe_before = (
        prediction_probe(learner, task["map"], task["seed"] + 19291)
        if has_prediction and task["environment"]["task"] == "single_goal"
        else None
    )
    history = []
    logs = []
    occupancy = np.zeros((task["map"]["size"],) * 2, np.int64)
    for ep in range(1, task["episodes"] + 1):
        learner.schedule(ep, task["episodes"])
        row, records, actions, counts = run_episode(learner, task, ep, 1, True)
        update = learner.update(records)
        if update.get("policy_core_gradient_norm", 0) != 0:
            raise AssertionError("RL must not update RSNN")
        row.update(episode=ep, learning=update)
        history.append(row)
        logs.append(actions)
        occupancy += counts
        if ep % 10 == 0 or ep == task["episodes"]:
            write_json(folder / "progress.json", dict(episode=ep, total=task["episodes"]))
    frozen = learner.weights_hash()
    if task["validation"]["require_prediction_updates"] and has_prediction:
        if learner.backbone_hash() == initial_core or learner.predictor_hash() == initial_predictor:
            raise AssertionError("Prediction must update both RSNN and predictor")
    if task["group"] == "no_rl" or task["model_config"]["kind"] == "random":
        if (learner.policy_hash() if has_prediction else frozen) != initial_policy:
            raise AssertionError("No-RL/random policy weights changed during training")
    arrays = (
        learner.parameters()
        if has_prediction
        else [learner.table]
        if learner.kind == "q_table"
        else [p.detach().numpy() for p in learner.net.parameters()]
    )
    if not all(np.isfinite(p).all() for p in arrays):
        raise FloatingPointError("Non-finite trained parameters")
    baseline = learner.baseline.copy() if has_prediction else None
    evaluation = {}
    evaluation_hashes = {}
    modes = task["evaluation_modes"] if has_prediction and getattr(learner, "backend", "calcium") != "none" else ["off"]
    for mode in modes:
        rows = []
        for ep in range(1, task["eval_episodes"] + 1):
            row, _, _, _ = run_episode(learner, task, ep, 8, False, mode)
            rows.append(row)
        if learner.weights_hash() != frozen or has_prediction and not np.array_equal(learner.baseline, baseline):
            raise AssertionError("Evaluation changed learned state")
        evaluation[mode] = mean(rows)
        evaluation_hashes[mode] = learner.weights_hash()
        gzip_json(folder / f"evaluation_{mode}.json.gz", rows)
    probe_after = prediction_probe(learner, task["map"], task["seed"] + 19291) if probe_before is not None else None
    if probe_after is not None and probe_after["representation_variance"] <= max(
        task["validation"]["variance_floor"],
        task["validation"]["retained_variance_fraction"] * probe_before["representation_variance"],
    ):
        raise AssertionError("Latent collapse on the fixed independent probe")
    gzip_json(folder / "training.json.gz", history)
    np.savez_compressed(
        folder / "navigation.npz",
        actions=np.concatenate(logs),
        offsets=np.r_[0, np.cumsum([len(x) for x in logs])],
        occupancy=occupancy,
    )
    auc = {}
    for key in ["reward01", "coverage", "path_efficiency"]:
        values = np.array([r[key] for r in history])
        auc[key] = float(np.mean((values[1:] + values[:-1]) * 0.5)) if len(values) > 1 else float(values[0])
    result = dict(
        task=task,
        metadata=learner.metadata(),
        evaluation=evaluation,
        evaluation_checkpoint_hashes=evaluation_hashes,
        initial_hash=initial,
        final_hash=frozen,
        probe_before=probe_before,
        probe_after=probe_after,
        training_auc=auc,
        seconds=time.time() - began,
    )
    write_json(folder / "result.json", result)
    return result


def run(config_name, output, workers=None, plan=False):
    config, hashes = load_experiment(config_name)
    tasks = make_tasks(config)
    print(
        json.dumps(
            dict(
                experiment=config["name"],
                tasks=len(tasks),
                episodes_per_run=config["protocol"]["episodes"],
                training_episodes=sum(t["episodes"] for t in tasks),
            )
        ),
        flush=True,
    )
    if plan:
        return tasks
    workers = config["protocol"]["workers"] if workers is None else workers
    if not 1 <= workers <= 30:
        raise ValueError("workers must be 1..30")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    sources = [*sorted((root / "src/astrorl").glob("*.py")), *sorted((root / "experiments").glob("*.py"))]
    if not (root / "src/astrorl").exists():
        import astrorl

        sources = [*sorted(Path(astrorl.__file__).parent.glob("*.py")), *sorted(Path(__file__).parent.glob("*.py"))]
    manifest = dict(
        configuration=config,
        config_sha256=hashes,
        source_sha256={
            str(p.relative_to(p.parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        python=sys.version,
        torch=torch.__version__,
        numpy=np.__version__,
        platform=platform.platform(),
    )
    if (output / "manifest.json").exists() and read_json(output / "manifest.json") != manifest:
        raise RuntimeError("Output belongs to another configuration/source/environment; choose a new directory")
    write_json(output / "manifest.json", manifest)
    write_json(output / "tasks.json", tasks)
    rows = []
    write_json(output / "status.json", dict(stage="running", completed=0, total=len(tasks), workers=workers))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(case, t, str(output)) for t in tasks]):
            try:
                rows.append(f.result())
            except Exception:
                write_json(output / "error.json", dict(traceback=traceback.format_exc()))
                write_json(
                    output / "status.json", dict(stage="failed", completed=len(rows), total=len(tasks), workers=workers)
                )
                raise
            write_json(output / "index.json", rows)
            write_json(
                output / "status.json", dict(stage="running", completed=len(rows), total=len(tasks), workers=workers)
            )
    write_json(output / "status.json", dict(stage="complete", completed=len(rows), total=len(tasks), workers=workers))
    return rows
