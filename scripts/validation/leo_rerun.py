"""Docker-based batch Leo re-run using the universal image."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


DOCKER_IMAGE = "leo-base-universal:latest"


@dataclass
class RerunTask:
    """A single Leo re-run job for the batch Docker runner."""

    label: str  # e.g., "Apps_LTIMES/nvidia"
    meas_dir: Path  # host path to hpctoolkit-*-measurements
    arch: str  # e.g., "h100"
    output_file: Path  # host path for leo_output_latest.txt


def batch_rerun_leo_docker(tasks: list[RerunTask], leo_root: Path) -> None:
    """Re-run Leo on all tasks in a single Docker container.

    Generates an inner shell script, mounts the entire results tree + Leo
    source, and runs all analyses sequentially inside one container.
    The universal image has nvdisasm + llvm-objdump + Intel GTPin.
    """
    if not tasks:
        return

    leo_root = leo_root.resolve()
    # Mount the results root so all measurement/database dirs are visible.
    results_root = leo_root / "results"
    results_root = results_root.resolve()
    container_results = "/data/results"
    total = len(tasks)

    print(f"\n  Starting Docker container ({DOCKER_IMAGE})...", file=sys.stderr)
    print(f"  {total} analyses to run", file=sys.stderr)

    # Build the inner script
    # Use umask 000 so output files are world-writable (rootless Docker maps
    # container root to host 'nobody', so chown doesn't work)
    script_lines = [
        "#!/bin/bash",
        "set -e",
        "umask 000",
        "cd /opt/leo",
        "cp -r /opt/leo-host/src/leo/* src/leo/",
        "cp -r /opt/leo-host/scripts/* scripts/",
        "export UV_PROJECT_ENVIRONMENT=/tmp/leo-venv",
        'uv run python -c "print(\'Leo environment ready\')" 2>&1',
        "",
    ]

    for i, task in enumerate(tasks, 1):
        # Convert host path to container path
        try:
            rel = task.meas_dir.resolve().relative_to(results_root)
        except ValueError:
            print(f"  SKIP {task.label}: meas_dir not under results/", file=sys.stderr)
            continue
        container_meas = f"{container_results}/{rel}"

        out_rel = task.output_file.resolve().relative_to(results_root)
        container_out = f"{container_results}/{out_rel}"

        script_lines.append(f'echo "[{i}/{total}] {task.label}"')
        script_lines.append(
            f"UV_PROJECT_ENVIRONMENT=/tmp/leo-venv "
            f"uv run python scripts/analyze_benchmark.py "
            f"{container_meas} --arch {task.arch} --top-n 2 "
            f"> {container_out} 2>&1 || true"
        )
        script_lines.append("")

    script_lines.append(f'echo "Done: {total} analyses"')

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, prefix="leo-rerun-"
    ) as f:
        f.write("\n".join(script_lines))
        inner_script = f.name

    try:
        os.chmod(inner_script, 0o755)
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{results_root}:{container_results}",
                "-v", f"{leo_root}:/opt/leo-host:ro",
                "-v", f"{inner_script}:/tmp/run.sh:ro",
                "--entrypoint", "",
                DOCKER_IMAGE,
                "bash", "/tmp/run.sh",
            ],
            text=True, timeout=1800,
        )
    except subprocess.TimeoutExpired:
        print("  ERROR: Docker container timed out (30 min)", file=sys.stderr)
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
    finally:
        os.unlink(inner_script)

    # Post-check: remove output files without valid analysis to avoid
    # overriding good leo_output.txt with empty/error results.
    ok = 0
    bad = 0
    for t in tasks:
        if not t.output_file.exists():
            continue
        content = t.output_file.read_text()
        if "STALL ANALYSIS" in content:
            ok += 1
        else:
            t.output_file.unlink()
            bad += 1

    print(f"  Container finished: {ok}/{total} valid, {bad} failed (removed)",
          file=sys.stderr)
