"""MDP single-goal rollouts; coordinates and visited labels stay in evaluator."""

import numpy as np
from astrorl.environment import geometry


def schedule(spec, seed, count, split):
    free = np.asarray([p for p in sorted(map(tuple, spec["free"])) if p != max(map(tuple, spec["free"]))])
    rng = np.random.default_rng(np.random.SeedSequence([20261023, seed, split, 917]))
    return free[rng.integers(len(free), size=count)]


def action_seed(seed, episode, split):
    return int(np.random.SeedSequence([20261023, seed, split, episode, 913]).generate_state(1)[0])


def trajectory(
    learner, spec, start, seed, training=False, mode="learned", horizon=50, reward_goal=1.0, step_penalty=0.01
):
    free, index, goal, dist, tr = geometry(spec)
    size = spec["size"]
    state = index[tuple(start)]
    assert 0 < dist[tuple(start)] <= horizon
    learner.begin_episode(training, mode)
    rng = np.random.default_rng(seed)
    encoder = learner.input_coder.observation_encoder
    observation = lambda p: encoder.encode(np.asarray(p, dtype=np.float32) / (size - 1))
    rep = learner.representation(observation(free[state]), None, training)
    learner.write_real(rep["z"], int(seed % 2**32))
    neural_diagnostics = []
    latent_rates = []
    records = []
    action_log = []
    visited = {tuple(start)}
    path = [tuple(start)]
    pairs = set()
    cover = []
    occupancy = np.zeros((size, size), np.int64)
    arrivals = occupancy.copy()
    pred_count = occupancy.copy()
    pred_error = occupancy.copy()
    back = collisions = revisits = correct = move_correct = collision_correct = moves = 0
    errors = []
    entropies = []
    gates = []
    silent = []
    discrimination = []
    for tick in range(horizon):
        pos = free[state]
        obs = np.asarray(pos, dtype=np.float32) / (size - 1)
        latent_rates.append(rep["z"].copy())
        predicted_input = learner.prediction(rep)
        p, cache, path_eligibility, diag, context, p_off = learner.decision(rep, predicted_input)
        # Coordinates and visited labels are evaluator-only diagnostics.
        collided = tr[state] == state
        novel = np.array([free[int(x)] not in visited for x in tr[state]])
        novel_transition = np.array([(pos, a) not in pairs for a in range(4)]) & ~collided
        diag.update(
            collision_probability_shift=float(p[collided].sum() - p_off[collided].sum()),
            novel_probability_shift=float(p[novel].sum() - p_off[novel].sum()),
            novel_suppression_mass=float(np.maximum(p_off[novel] - p[novel], 0).sum()),
            suppressed_novel_actions=float(np.sum(novel & (p < p_off - 0.01))),
            novel_actions=float(novel.sum()),
            novel_transition_probability_shift=float(p[novel_transition].sum() - p_off[novel_transition].sum()),
            novel_transition_suppression_mass=float(np.maximum(p_off[novel_transition] - p[novel_transition], 0).sum()),
            suppressed_novel_transitions=float(np.sum(novel_transition & (p < p_off - 0.01))),
            novel_transitions=float(novel_transition.sum()),
        )
        neural_diagnostics.append(diag)
        assert np.isfinite(p).all() and abs(p.sum() - 1) < 1e-6
        action = min(3, int(np.searchsorted(np.cumsum(p), rng.random(), side="right")))
        action_log.append(action)
        nxt_state = int(tr[state, action])
        nxt = free[nxt_state]
        hit = nxt_state == goal
        collision = nxt_state == state
        moved = not collision
        target_obs = np.asarray(nxt, dtype=np.float32) / (size - 1)
        accurate = False
        correct += accurate
        move_correct += accurate and moved
        collision_correct += accurate and collision
        moves += moved
        pred_count[pos] += 1
        pred_error[pos] += not accurate
        next_observation = observation(nxt)
        # Observe the actual reached state once, after the action. Reuse it
        # next iteration, so there is no virtual or duplicated RSNN step.
        next_rep = learner.representation(next_observation, action, training)
        teacher = next_rep["z"].copy()
        neural_diagnostics[-1]["familiarity_prediction_error"] = learner.observe_outcome(
            action, context, teacher, int((seed + (tick + 1) * 7919) % 2**32)
        )
        neural_diagnostics[-1]["forecast_mse"] = float(np.square(predicted_input[action] - teacher).mean())
        neural_diagnostics[-1]["persistence_forecast_mse"] = float(np.square(rep["z"] - teacher).mean())
        errors.append(float(np.square(predicted_input[action] - teacher).mean()))
        discrimination.append(float(np.var(predicted_input, axis=0).mean()))
        occupancy[pos] += 1
        arrivals[nxt] += 1
        pairs.add((pos, action))
        back += moved and len(path) > 1 and nxt == path[-2]
        revisits += nxt in visited
        collisions += collision
        visited.add(nxt)
        path.append(nxt)
        cover.append(len(visited) / len(free))
        entropies.append(float(-np.sum(p * np.log(np.maximum(p, 1e-30)))))
        gates.append(float(diag["mean_gate"]))
        silent.append(float(diag["motor_silent_fraction"]))
        if training:
            records.append(
                dict(
                    **rep,
                    action=action,
                    next_observation=next_observation,
                    next_z=teacher,
                    path_eligibility=path_eligibility.copy(),
                    local_familiarity=context["local"].copy(),
                    reward=float(hit) * reward_goal - step_penalty,
                    terminal=hit or tick == horizon - 1,
                    probability=p,
                    head_cache=cache,
                )
            )
        previous = action
        rep = next_rep
        state = nxt_state
        if hit:
            break
    learner.finish_episode(training)
    T = tick + 1
    success = int(state == goal)
    distance = dist[tuple(start)]
    reward = success * reward_goal - step_penalty * T
    q = occupancy[occupancy > 0] / T
    persisted = T - moves
    row = dict(
        steps=T,
        success=success,
        reward=reward,
        reward01=(reward + step_penalty * horizon) / (reward_goal + step_penalty * horizon - step_penalty * distance),
        optimal_steps=distance,
        optimal_over_steps=distance / T,
        path_efficiency=success * distance / T,
        coverage=len(visited) / len(free),
        coverage_auc=float(np.mean(cover)),
        coverage_auc_50=float(np.mean(cover + [cover[-1]] * (horizon - T))),
        unique_states=len(visited),
        novelty_rate=(len(visited) - 1) / T,
        exploration_efficiency=(len(visited) - 1) / min(T, len(free) - 1),
        backtracking_rate=back / max(moves, 1),
        backtracking_per_action=back / T,
        collision_rate=collisions / T,
        revisit_rate=revisits / T,
        state_action_coverage=len(pairs) / (4 * len(free)),
        spatial_entropy=float(-np.sum(q * np.log(q)) / np.log(len(free))),
        loop2_rate=float(np.mean([path[k] == path[k - 2] for k in range(2, len(path))])) if T >= 2 else 0.0,
        loop4_rate=float(np.mean([path[k] == path[k - 4] for k in range(4, len(path))])) if T >= 4 else 0.0,
        entropy=float(np.mean(entropies)),
        mean_gate=float(np.mean(gates)),
        motor_silent_fraction=float(np.mean(silent)),
        prediction_accuracy=correct / T,
        prediction_move_accuracy=move_correct / moves if moves else None,
        prediction_collision_accuracy=collision_correct / persisted if persisted else None,
        prediction_balanced_accuracy=0.5 * (move_correct / moves + collision_correct / persisted)
        if moves and persisted
        else correct / T,
        prediction_mse=float(np.mean(errors)),
        prediction_rmse=float(np.sqrt(np.mean(errors))),
        persistence_accuracy=persisted / T,
        action_discrimination=float(np.mean(discrimination)),
        prediction_move_count=moves,
        prediction_collision_count=persisted,
    )
    row.update(
        latent_activity_mean=float(np.mean(latent_rates)),
        latent_activity_variance=float(np.var(np.asarray(latent_rates), axis=0).mean()),
    )
    for key in (
        "prediction_accuracy",
        "prediction_move_accuracy",
        "prediction_collision_accuracy",
        "prediction_balanced_accuracy",
    ):
        row[key] = None
    if neural_diagnostics:
        row.update({k: float(np.mean([d[k] for d in neural_diagnostics])) for k in neural_diagnostics[0]})
    assert 0 <= row["reward01"] <= 1 + 1e-9 and 0 < row["optimal_over_steps"] <= 1 + 1e-9
    return (
        row,
        records,
        dict(
            occupancy=occupancy,
            arrivals=arrivals,
            pred_count=pred_count,
            pred_error=pred_error,
            actions=np.asarray(action_log, dtype=np.uint8),
        ),
        path,
    )
