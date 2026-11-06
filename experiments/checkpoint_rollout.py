"""Matched three-goal environment, true next-RSNN targets and evaluator-only oracles."""

import numpy as np
from .task_helpers import Oracle, visibility
from astrorl.checkpoint_task import CheckpointTask

ORACLES = {}


def rollout(learner, config, mode, seed, training, memory_mode="learned", horizon=150):
    env = CheckpointTask(config, mode, horizon)
    key = (tuple(sorted(env.free)), tuple(env.points), mode)
    if key not in ORACLES:
        ORACLES[key] = Oracle(config, mode)
    oracle = ORACLES[key]
    state = env.reset()
    learner.begin_episode(training, memory_mode)
    rng = np.random.default_rng(seed)
    encoder = learner.input_coder.observation_encoder

    def observe(state, previous):
        learner.input_coder.context = visibility(state.remaining, mode)
        return learner.representation(
            encoder.encode(np.asarray(state.position, dtype=np.float32) / (config["map"]["size"] - 1)),
            previous,
            training,
        )

    rep = observe(state, None)
    learner.write_real(rep["z"], int(seed % 2**32))
    records = []
    actions = []
    points = [
        dict(
            checkpoint_id=i + 1,
            activation_step=0 if mode == "unordered" or i == 0 else None,
            collection_step=None,
            first_visit_step=None,
            visits=0,
            minimum_distance=oracle.dist[state.position][p],
        )
        for i, p in enumerate(oracle.points)
    ]
    initial = oracle.remaining(state.position, state.remaining)
    visited = {state.position}
    pairs = set()
    path = [state.position]
    last_collection = 0
    last_pos = state.position
    back = collisions = 0
    errors = []
    entropies = []
    diagnostics = []
    occupancy = np.zeros((config["map"]["size"],) * 2, np.int64)
    for tick in range(horizon):
        before = state
        predicted = learner.prediction(rep)
        p, cache, elig, diag, ctx, p_off = learner.decision(rep, predicted)
        outcomes = [env.step(before, a)[0].position for a in range(4)]
        blocked = np.array([q == before.position for q in outcomes])
        novel = np.array([q not in visited for q in outcomes])
        newedge = np.array([(before.position, a) not in pairs for a in range(4)]) & ~blocked
        diag.update(
            collision_probability_shift=float(p[blocked].sum() - p_off[blocked].sum()),
            novel_probability_shift=float(p[novel].sum() - p_off[novel].sum()),
            novel_suppression_mass=float(np.maximum(p_off[novel] - p[novel], 0).sum()),
            novel_transition_suppression_mass=float(np.maximum(p_off[newedge] - p[newedge], 0).sum()),
        )
        assert np.isfinite(p).all() and abs(p.sum() - 1) < 1e-8
        a = min(3, int(np.searchsorted(np.cumsum(p), rng.random(), side="right")))
        actions.append(a)
        state, reward, terminated, truncated = env.step(before, a)
        nxt = observe(state, a)
        diag["familiarity_prediction_error"] = learner.observe_outcome(
            a, ctx, nxt["z"], int((seed + (tick + 1) * 7919) % 2**32)
        )
        diagnostics.append(diag)
        errors.append(float(np.square(predicted[a] - nxt["z"]).mean()))
        entropies.append(float(-np.sum(p * np.log(np.maximum(p, 1e-30)))))
        collected = next((i for i in range(3) if (before.remaining ^ state.remaining) & (1 << i)), None)
        collision = before.position == state.position
        collisions += collision
        back += not collision and len(path) > 1 and state.position == path[-2]
        occupancy[before.position] += 1
        visited.add(state.position)
        path.append(state.position)
        pairs.add((before.position, a))
        for i, point in enumerate(oracle.points):
            q = points[i]
            q["minimum_distance"] = min(q["minimum_distance"], oracle.dist[state.position][point])
            if state.position == point:
                q["visits"] += 1
                if q["first_visit_step"] is None:
                    q["first_visit_step"] = tick + 1
        if collected is not None:
            points[collected].update(
                collection_step=tick + 1,
                collection_rank=3 - state.remaining.bit_count(),
                steps_from_activation=tick + 1 - points[collected]["activation_step"],
                steps_from_previous_collection=tick + 1 - last_collection,
                segment_optimal_steps=oracle.dist[last_pos][oracle.points[collected]],
            )
            last_collection = tick + 1
            last_pos = state.position
            if mode == "ordered" and state.remaining:
                points[next(i for i in range(3) if state.remaining & (1 << i))]["activation_step"] = tick + 1
        if training:
            records.append(
                dict(
                    **rep,
                    action=a,
                    next_z=nxt["z"].copy(),
                    path_eligibility=elig.copy(),
                    local_familiarity=ctx["local"].copy(),
                    reward=float(reward),
                    terminal=terminated or truncated,
                    probability=p,
                    head_cache=cache,
                )
            )
        rep = nxt
        if terminated or truncated:
            break
    learner.finish_episode(training)
    for q in points:
        q.update(
            collected=q["collection_step"] is not None,
            censored=q["collection_step"] is None,
            observation_end_step=state.steps,
        )
    row = dict(
        steps=state.steps,
        collected=3 - state.remaining.bit_count(),
        reward=3 - state.remaining.bit_count(),
        normalized_reward=(3 - state.remaining.bit_count()) / 3,
        success=float(state.remaining == 0),
        optimal_steps=initial,
        optimal_over_steps=initial / state.steps,
        path_efficiency=initial / state.steps if state.remaining == 0 else 0.0,
        coverage=len(visited) / len(env.free),
        collision_rate=collisions / state.steps,
        backtracking_rate=back / max(1, state.steps - collisions),
        prediction_mse=float(np.mean(errors)),
        entropy=float(np.mean(entropies)),
        checkpoints=points,
        start=list(env.start),
    )
    row.update({k: float(np.mean([d[k] for d in diagnostics])) for k in diagnostics[0]})
    return row, records, np.asarray(actions, np.uint8), occupancy
