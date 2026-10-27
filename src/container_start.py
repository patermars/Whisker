"""Download the Whisker checkpoint if needed, then start the web server."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_checkpoint(checkpoint: Path) -> None:
    if checkpoint.is_file() and checkpoint.stat().st_size > 0:
        print(f"Using local checkpoint: {checkpoint}", flush=True)
        return

    bucket = os.getenv("WHISKER_CHECKPOINT_BUCKET")
    key = os.getenv("WHISKER_CHECKPOINT_KEY")
    if not bucket or not key:
        raise SystemExit(
            "Checkpoint is missing. Set WHISKER_CHECKPOINT_BUCKET and "
            "WHISKER_CHECKPOINT_KEY."
        )

    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint.with_name(f"{checkpoint.name}.download")
    temporary_path.unlink(missing_ok=True)
    print(f"Downloading checkpoint from s3://{bucket}/{key}", flush=True)
    try:
        import boto3

        boto3.client("s3").download_file(bucket, key, str(temporary_path))
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise RuntimeError("S3 returned an empty checkpoint file")
        os.replace(temporary_path, checkpoint)
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        print(f"Checkpoint download failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
    print(f"Checkpoint downloaded to {checkpoint}", flush=True)


def main() -> None:
    checkpoint = Path(
        os.getenv("WHISKER_CHECKPOINT_PATH", "/tmp/whisker/model.pt")
    )
    _ensure_checkpoint(checkpoint)

    port = os.getenv("PORT", "8080")
    threads = os.getenv("WHISKER_THREADS", "1")
    os.environ.setdefault("WHISKER_DEVICE", "cpu")
    os.environ.setdefault("WHISKER_HOST", "0.0.0.0")
    os.environ.setdefault("WHISKER_REQUIRE_MODEL", "1")

    command = [
        sys.executable,
        "-m",
        "src.serve",
        "--checkpoint",
        str(checkpoint),
        "--device",
        "cpu",
        "--host",
        "0.0.0.0",
        "--port",
        port,
        "--threads",
        threads,
    ]
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
