"""
Command-line interface for ORTHON Prediction Module.

Usage:
    python -m orthon.prediction rul /path/to/prism/output
    python -m orthon.prediction health /path/to/prism/output
    python -m orthon.prediction anomaly /path/to/prism/output --method zscore
"""

import argparse
import json
import sys
from pathlib import Path

from .anomaly import AnomalyDetector, AnomalyMethod
from .health import HealthScorer
from .rul import RULPredictor


def cmd_rul(args: argparse.Namespace) -> int:
    """Run RUL prediction."""
    try:
        predictor = RULPredictor(
            args.prism_dir,
            failure_threshold=args.threshold,
            min_history=args.min_history,
        )

        result = predictor.predict(args.unit)

        if args.explain and args.unit:
            explanation = predictor.explain(args.unit)
            output = {"prediction": result.to_dict(), "explanation": explanation}
        else:
            output = result.to_dict()

        print(json.dumps(output, indent=2, default=str))
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_health(args: argparse.Namespace) -> int:
    """Run health scoring."""
    try:
        scorer = HealthScorer(
            args.prism_dir,
            baseline_mode=args.baseline,
        )

        result = scorer.predict(args.unit)

        if args.explain and args.unit:
            explanation = scorer.explain(args.unit)
            output = {"prediction": result.to_dict(), "explanation": explanation}
        else:
            output = result.to_dict()

        print(json.dumps(output, indent=2, default=str))
        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_anomaly(args: argparse.Namespace) -> int:
    """Run anomaly detection."""
    try:
        method = AnomalyMethod(args.method)
        detector = AnomalyDetector(
            args.prism_dir,
            method=method,
            threshold=args.threshold,
            contamination=args.contamination,
        )

        result = detector.predict(args.unit)

        if args.explain and args.unit:
            explanation = detector.explain(args.unit)
            output = {"prediction": result.to_dict(), "explanation": explanation}
        else:
            output = result.to_dict()

        print(json.dumps(output, indent=2, default=str))

        # Optionally output anomaly indices
        if args.indices:
            indices = detector.get_anomaly_indices(args.unit)
            print(f"\nAnomaly indices: {indices[:50]}", file=sys.stderr)
            if len(indices) > 50:
                print(f"  ... and {len(indices) - 50} more", file=sys.stderr)

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="ORTHON Prediction Module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict RUL for all units
  python -m orthon.prediction rul /path/to/prism/output

  # Score health for specific unit with explanation
  python -m orthon.prediction health /path/to/prism/output --unit unit_1 --explain

  # Detect anomalies using isolation forest
  python -m orthon.prediction anomaly /path/to/prism/output --method isolation_forest
        """,
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # RUL subcommand
    rul_parser = subparsers.add_parser("rul", help="Predict Remaining Useful Life")
    rul_parser.add_argument("prism_dir", type=Path, help="PRISM output directory")
    rul_parser.add_argument("--unit", type=str, help="Specific unit to predict")
    rul_parser.add_argument(
        "--threshold", type=float, default=0.8,
        help="Failure threshold (default: 0.8)"
    )
    rul_parser.add_argument(
        "--min-history", type=int, default=10,
        help="Minimum observations for trend (default: 10)"
    )
    rul_parser.add_argument("--explain", action="store_true", help="Include explanation")
    rul_parser.set_defaults(func=cmd_rul)

    # Health subcommand
    health_parser = subparsers.add_parser("health", help="Compute health score")
    health_parser.add_argument("prism_dir", type=Path, help="PRISM output directory")
    health_parser.add_argument("--unit", type=str, help="Specific unit to score")
    health_parser.add_argument(
        "--baseline", type=str, default="first_10_percent",
        choices=["first_10_percent", "global_mean"],
        help="Baseline mode (default: first_10_percent)"
    )
    health_parser.add_argument("--explain", action="store_true", help="Include explanation")
    health_parser.set_defaults(func=cmd_health)

    # Anomaly subcommand
    anomaly_parser = subparsers.add_parser("anomaly", help="Detect anomalies")
    anomaly_parser.add_argument("prism_dir", type=Path, help="PRISM output directory")
    anomaly_parser.add_argument("--unit", type=str, help="Specific unit to analyze")
    anomaly_parser.add_argument(
        "--method", type=str, default="zscore",
        choices=["zscore", "isolation_forest", "lof", "combined"],
        help="Detection method (default: zscore)"
    )
    anomaly_parser.add_argument(
        "--threshold", type=float, default=3.0,
        help="Z-score threshold (default: 3.0)"
    )
    anomaly_parser.add_argument(
        "--contamination", type=float, default=0.1,
        help="Expected anomaly proportion (default: 0.1)"
    )
    anomaly_parser.add_argument("--explain", action="store_true", help="Include explanation")
    anomaly_parser.add_argument("--indices", action="store_true", help="Output anomaly indices")
    anomaly_parser.set_defaults(func=cmd_anomaly)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
