#!/usr/bin/env python3
"""CLI tool for GPU per-pipeline CPI stack analysis.

Analyzes HPCToolkit GPU profile databases and produces per-pipeline
breakdown showing Issue% and Stall% for each GPU pipeline.

By default, generates both JSON (for LLM) and PNG (for human).

Usage:
    python scripts/pipeline_analysis.py <database_path>
    python scripts/pipeline_analysis.py <database_path> -o output_prefix
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from leo.analysis.pipeline import pipeline_analysis


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GPU Per-Pipeline CPI Stack Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("database", type=Path, help="Path to HPCToolkit database")
    parser.add_argument("-o", "--output", type=str, help="Output prefix (default: <db_name>_pipeline)")
    parser.add_argument("--scope", choices=["i", "e"], default="i",
                        help="Metric scope: i=inclusive (default), e=exclusive")
    parser.add_argument("--json-only", action="store_true", help="Output JSON only (no figure)")
    parser.add_argument("--figure-only", action="store_true", help="Output figure only (no JSON)")
    parser.add_argument("--show-all", action="store_true",
                        help="Show all 11 pipelines even if 100%% idle (for consistent cross-workload comparison)")

    args = parser.parse_args()

    db_path = args.database.resolve()
    if not db_path.exists():
        print(f"Error: Database not found: {db_path}", file=sys.stderr)
        return 1

    # Determine output prefix
    if args.output:
        prefix = args.output
    else:
        # Default to outputs/ directory
        outputs_dir = Path(__file__).parent.parent / "outputs"
        outputs_dir.mkdir(exist_ok=True)
        db_name = db_path.name.replace("hpctoolkit-", "").replace("-database", "")
        prefix = str(outputs_dir / f"{db_name}_pipeline")

    # Run analysis
    try:
        result = pipeline_analysis(str(db_path), scope=args.scope)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Print summary
    print(result.format_summary())

    # Save outputs
    if not args.figure_only:
        json_path = f"{prefix}.json"
        result.save_json(json_path)
        print(f"\nJSON saved: {json_path}")

    if not args.json_only:
        png_path = f"{prefix}.png"
        result.save_figure(png_path, show_all=args.show_all)
        print(f"Figure saved: {png_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
