"""Tables and training figures for composed MDP experiments."""

import csv
import gzip
import json
from pathlib import Path
import numpy as np
from astrorl.logging import read_json, write_json
from .modified import identity

METRICS = ["reward01", "coverage", "path_efficiency", "auc_reward01", "auc_coverage", "auc_path_efficiency"]


def clustered_summary(records, settings):
    """Maps are outer replication units; seeds are resampled within maps."""
    maps = sorted({r["map"] for r in records})
    seeds = sorted({r["seed"] for r in records})
    lookup = {(r["map"], r["seed"]): r for r in records}
    if len(lookup) != len(maps) * len(seeds) or len(records) != len(lookup):
        raise ValueError("Clustered summaries require complete paired map/seed conditions")
    values = np.array([[[lookup[m, s][k] for k in METRICS] for s in seeds] for m in maps])
    rng = np.random.default_rng(settings["seed"])
    draws = settings["bootstrap_draws"]
    outer = rng.integers(len(maps), size=(draws, len(maps)))
    inner = rng.integers(len(seeds), size=(draws, len(maps), len(seeds)))
    boot = values[outer[:, :, None], inner].mean((1, 2))
    tail = (1 - settings["interval"]) / 2
    lo, hi = np.quantile(boot, [tail, 1 - tail], axis=0)
    avg = values.mean((0, 1))
    result = dict(maps=len(maps), seeds_per_map=len(seeds))
    for i, k in enumerate(METRICS):
        result.update({k: float(avg[i]), k + "_lo": float(lo[i]), k + "_hi": float(hi[i])})
    return result


def report(output):
    root = Path(output)
    if read_json(root / "status.json")["stage"] != "complete":
        raise ValueError("Complete the declared experiment before aggregating results")
    manifest = read_json(root / "manifest.json")
    settings = manifest["configuration"]["protocol"]["statistics"]
    rows = read_json(root / "index.json")
    records = []
    for row in rows:
        task = row["task"]
        active = row["evaluation"].get("learned", row["evaluation"]["off"])
        records.append(
            dict(
                family=task["family"],
                model=task["model"],
                group=task["group"],
                mode=task["task_mode"],
                map=task["map"]["name"],
                seed=task["seed"],
                **{k: active[k] for k in ["reward01", "coverage", "path_efficiency"]},
                **{"auc_" + k: v for k, v in row["training_auc"].items()},
            )
        )
    keys = list(records[0])
    with (root / "summary_per_run.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(records)
    conditions = sorted({(r["family"], r["model"], r["group"], r["mode"]) for r in records})
    summary = []
    for family, model, group, mode in conditions:
        selected = [
            r for r in records if (r["family"], r["model"], r["group"], r["mode"]) == (family, model, group, mode)
        ]
        summary.append(
            dict(
                family=family,
                model=model,
                group=group,
                mode=mode,
                runs=len(selected),
                **clustered_summary(selected, settings),
            )
        )
    write_json(root / "summary.json", summary)
    aliases = {"full_mixed_pool", "full_mixed_single", "local_equal_pool", "local_equal_single"}
    if aliases <= {r["model"] for r in records}:
        effects = []
        for family, group, mode in sorted({(r["family"], r["group"], r["mode"]) for r in records}):
            rr = [
                r
                for r in records
                if (r["family"], r["group"], r["mode"]) == (family, group, mode) and r["model"] in aliases
            ]
            lookup = {(r["model"], r["map"], r["seed"]): r for r in rr}
            pairs = sorted({(r["map"], r["seed"]) for r in rr})
            contrasts = {
                "pool_effect_full": {"full_mixed_pool": 1, "full_mixed_single": -1},
                "pool_effect_local": {"local_equal_pool": 1, "local_equal_single": -1},
                "pool_interaction": {
                    "full_mixed_pool": 1,
                    "full_mixed_single": -1,
                    "local_equal_pool": -1,
                    "local_equal_single": 1,
                },
            }
            for name, weights in contrasts.items():
                differences = [
                    dict(
                        map=m,
                        seed=s,
                        **{k: sum(w * lookup[model, m, s][k] for model, w in weights.items()) for k in METRICS},
                    )
                    for m, s in pairs
                ]
                effects.append(
                    dict(
                        family=family, group=group, mode=mode, contrast=name, **clustered_summary(differences, settings)
                    )
                )
        write_json(root / "factorial_effects.json", effects)
    with (root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    families = sorted({r["family"] for r in records})
    modes = sorted({(r["group"], r["mode"]) for r in records})
    models = sorted({r["model"] for r in records})
    histories = {}
    for row in rows:
        with gzip.open(root / "runs" / identity(row["task"]) / "training.json.gz", "rt") as f:
            histories[identity(row["task"])] = json.load(f)
    for metric in ["reward01", "coverage", "path_efficiency"]:
        fig, axes = plt.subplots(
            len(modes),
            len(families),
            figsize=(5 * len(families), 3.5 * len(modes)),
            squeeze=False,
            layout="constrained",
        )
        for i, (group, mode) in enumerate(modes):
            for j, family in enumerate(families):
                ax = axes[i, j]
                for model in models:
                    values = []
                    for row in rows:
                        t = row["task"]
                        if (t["group"], t["task_mode"], t["family"], t["model"]) != (group, mode, family, model):
                            continue
                        v = np.array([h[metric] for h in histories[identity(t)]])
                        values.append([v[max(0, k - 19) : k + 1].mean() for k in range(len(v))])
                    if values:
                        ax.plot(np.arange(1, len(values[0]) + 1), np.mean(values, axis=0), label=model, lw=1.3)
                ax.set(title=f"{family}/{group}/{mode}", xlabel="Training episode", ylabel=metric)
                ax.grid(alpha=0.2)
                ax.legend(fontsize=7)
        for ext in ["png", "pdf", "svg"]:
            fig.savefig(root / f"training_{metric}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)
    return summary
