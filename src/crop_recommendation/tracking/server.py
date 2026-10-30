"""Localhost-only MLflow tracking server launcher for repository storage."""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Sequence

from crop_recommendation.tracking.mlflow_tracking import MlflowTrackingConfig


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Start the repository-local MLflow UI and tracking server."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Localhost port (default: 5000).",
    )
    arguments = parser.parse_args(argv)
    if not 1 <= arguments.port <= 65_535:
        parser.error("--port must be between 1 and 65535")

    config = MlflowTrackingConfig()
    config.artifact_root.mkdir(parents=True, exist_ok=True)
    config.artifact_root.parent.mkdir(parents=True, exist_ok=True)

    upgrade = subprocess.run(
        [
            sys.executable,
            "-m",
            "mlflow",
            "db",
            "upgrade",
            config.tracking_uri,
        ],
        check=False,
    )
    if upgrade.returncode != 0:
        return upgrade.returncode

    command = [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--backend-store-uri",
        config.tracking_uri,
        "--default-artifact-root",
        config.artifact_root_uri,
        "--no-serve-artifacts",
        "--host",
        "127.0.0.1",
        "--port",
        str(arguments.port),
        "--workers",
        "1",
    ]
    try:
        return subprocess.run(command, check=False).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
