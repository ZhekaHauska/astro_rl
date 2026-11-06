"""Command line entry point for training, tuning, scientific checks and plotting."""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "tune", "plot", "check", "plan"])
    parser.add_argument(
        "--config", help="Top-level experiment path relative to configs/, e.g. experiments/mdp_factorial.json"
    )
    parser.add_argument("--legacy", action="store_true", help="Explicitly run the original four-map article protocol")
    parser.add_argument(
        "--workers", type=int, help="Parallel training workers (1–30; default from experiment protocol)."
    )
    parser.add_argument("--output", help="Directory for generated metrics, checkpoints and figures.")
    args = parser.parse_args()
    if args.workers is not None and not 1 <= args.workers <= 30:
        parser.error("--workers must be between 1 and 30")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = "1"
    if args.output:
        os.environ["ASTRORL_OUTPUT_DIR"] = args.output
    if args.config:
        if args.legacy or args.command == "tune":
            parser.error(
                "Composed experiments use fixed selected parameters; --legacy/tune cannot be combined with --config"
            )
        from pathlib import Path
        from .modified import run

        output = args.output or str(Path("results") / Path(args.config).stem)
        if args.command == "plot":
            from .modified_analysis import report

            report(output)
        else:
            run(args.config, output, args.workers, plan=args.command in ("plan", "check"))
        return
    if args.command == "plan" or args.command == "run" and not args.legacy:
        parser.error("Choose --config experiments/<name>.json; use --legacy explicitly for the original article study")
    if args.workers is None:
        args.workers = 30
    # Numerical packages are imported after their thread budgets are configured.
    if args.command in ("run", "tune"):
        from . import runner

        getattr(runner, args.command)(workers=args.workers)
    elif args.command == "plot":
        from .analysis import report

        report()
    else:
        from .checks import main as check

        check()


if __name__ == "__main__":
    main()
