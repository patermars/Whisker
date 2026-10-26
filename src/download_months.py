"""Download and preprocess a sequence of monthly Lichess game archives.

Example:
    python -m whisker.download_months --year 2024 --min-elo 1000 --max-elo 1099

This creates ``data/processed/elo-1000-1099/2024-01.h5`` through
``data/processed/elo-1000-1099/2024-12.h5``.
Completed files are skipped on reruns; incomplete or differently filtered files
are rebuilt.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import sys
from pathlib import Path

import h5py

from .paths import PROJECT_ROOT

def file_matches_request(
    path: Path,
    month: str,
    min_elo: int,
    max_elo: int | None,
    max_games: int | None,
) -> bool:
    """Return whether a monthly HDF5 file is complete and matches the filters."""
    if not path.is_file():
        return False

    try:
        with h5py.File(path, "r") as dataset:
            return (
                bool(dataset.attrs.get("complete", False))
                and dataset.attrs.get("month") == month
                and int(dataset.attrs.get("min_elo", -1)) == min_elo
                and int(dataset.attrs.get("max_elo", -1))
                == (-1 if max_elo is None else max_elo)
                and int(dataset.attrs.get("max_games", -1))
                == (-1 if max_games is None else max_games)
                and all(name in dataset for name in ("boards", "policies", "values"))
                and dataset["boards"].shape[0] == dataset["policies"].shape[0]
                and dataset["boards"].shape[0] == dataset["values"].shape[0]
            )
    except (OSError, KeyError, TypeError, ValueError):
        return False


def prepare_month(month: str, command: list[str], output_path: Path) -> tuple[str, int, Path]:
    """Run one monthly preparation job and return its exit status."""
    print(f"\n[start] {month} → {output_path}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(f"[{month}] {line}", end="", flush=True)
    return month, process.wait(), output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and preprocess monthly Lichess games for one Elo band."
    )
    parser.add_argument("--year", type=int, required=True, help="Year to process, e.g. 2024")
    parser.add_argument("--min-elo", type=int, required=True)
    parser.add_argument(
        "--max-elo", type=int, default=None,
        help="Optional inclusive maximum rating for both players",
    )
    parser.add_argument("--start-month", type=int, default=1, help="First month (1-12)")
    parser.add_argument("--end-month", type=int, default=12, help="Last month (1-12)")
    parser.add_argument(
        "--max-games", type=int, default=None,
        help="Optional cap per month; by default process every qualifying game",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=(
            "Output folder (default: "
            "data/processed/elo-<min-elo>-<max-elo-or-plus>)"
        ),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Rebuild even complete monthly files",
    )
    parser.add_argument(
        "--jobs", type=int, default=4,
        help="Maximum number of months to process concurrently (default: 4)",
    )
    args = parser.parse_args()

    if args.min_elo < 0:
        parser.error("--min-elo must be non-negative")
    if args.max_elo is not None and args.max_elo < args.min_elo:
        parser.error("--max-elo must be greater than or equal to --min-elo")
    if not 1 <= args.start_month <= 12 or not 1 <= args.end_month <= 12:
        parser.error("--start-month and --end-month must be from 1 through 12")
    if args.start_month > args.end_month:
        parser.error("--start-month must not be after --end-month")
    if args.max_games is not None and args.max_games < 1:
        parser.error("--max-games must be positive")
    if args.jobs < 1:
        parser.error("--jobs must be positive")

    rating_max = args.max_elo if args.max_elo is not None else "plus"
    output_dir = args.output_dir or (
        PROJECT_ROOT / "data" / "processed" / f"elo-{args.min_elo}-{rating_max}"
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    months = [f"{args.year}-{month:02d}" for month in range(args.start_month, args.end_month + 1)]
    failures: list[str] = []
    pending_jobs: list[tuple[str, list[str], Path]] = []
    completed = 0

    print(f"Writing monthly HDF5 files to: {output_dir}", flush=True)
    print(
        f"Rating band: {args.min_elo}–{args.max_elo if args.max_elo is not None else '∞'}; "
        f"months: {months[0]} through {months[-1]}",
        flush=True,
    )
    print(f"Concurrent monthly jobs: up to {args.jobs}", flush=True)

    try:
        for month in months:
            output_path = output_dir / f"{month}.h5"
            if (
                not args.overwrite
                and file_matches_request(
                    output_path,
                    month,
                    args.min_elo,
                    args.max_elo,
                    args.max_games,
                )
            ):
                print(f"[skip] {output_path} is already complete", flush=True)
                completed += 1
                continue

            command = [
                sys.executable,
                "-u",
                "-m",
                "whisker.prepare_data",
                "--month",
                month,
                "--min-elo",
                str(args.min_elo),
                "--output",
                str(output_path),
            ]
            if args.max_elo is not None:
                command.extend(["--max-elo", str(args.max_elo)])
            if args.max_games is not None:
                command.extend(["--max-games", str(args.max_games)])

            pending_jobs.append((month, command, output_path))

        if pending_jobs:
            with ThreadPoolExecutor(max_workers=args.jobs) as executor:
                future_months = {
                    executor.submit(prepare_month, month, command, output_path): month
                    for month, command, output_path in pending_jobs
                }

                for future in as_completed(future_months):
                    month = future_months[future]
                    try:
                        _, return_code, output_path = future.result()
                    except Exception as exc:
                        failures.append(f"{month} (could not run prepare_data.py: {exc})")
                        print(f"[failed] {month}; {exc}", flush=True)
                        continue

                    if return_code != 0:
                        failures.append(f"{month} (prepare_data.py exited {return_code})")
                        print(f"[failed] {month}; continuing with other months", flush=True)
                        continue

                    if file_matches_request(
                        output_path,
                        month,
                        args.min_elo,
                        args.max_elo,
                        args.max_games,
                    ):
                        completed += 1
                        print(f"[done] {output_path}", flush=True)
                    else:
                        failures.append(
                            f"{month} (output was incomplete or had no completion metadata)"
                        )
                        print(f"[failed] {month}; output was not marked complete", flush=True)
    except KeyboardInterrupt:
        print("\nInterrupted. Completed months are preserved; rerun to continue.", flush=True)
        raise SystemExit(130)

    print(f"\nFinished: {completed}/{len(months)} monthly files complete.")
    if failures:
        print("Months needing attention:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print(
        "Train on the full period with: "
        f"python -m whisker.train --data \"{output_dir}\" ..."
    )


if __name__ == "__main__":
    main()
