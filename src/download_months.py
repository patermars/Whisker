"""Download monthly Lichess archives and prepare Elo-band datasets."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

import h5py
import requests
from tqdm import tqdm

from .paths import PROJECT_ROOT
from .prepare_data import (
    DEFAULT_ELO_SAMPLE_RATES,
    ELO_BANDS,
    LICHESS_URL,
    available_cpu_count,
    prepare_elo_bands,
)


def _handle_sigterm(_signum, _frame) -> None:
    raise KeyboardInterrupt


def file_matches_request(
    path: Path,
    month: str,
    min_elo: int,
    max_elo: int,
    max_games: int | None,
    sample_rate: float,
) -> bool:
    """Return whether an Elo-band HDF5 file is complete and matches filters."""
    if not path.is_file():
        return False

    try:
        with h5py.File(path, "r") as dataset:
            return (
                bool(dataset.attrs.get("complete", False))
                and dataset.attrs.get("month") == month
                and int(dataset.attrs.get("min_elo", -1)) == min_elo
                and int(dataset.attrs.get("max_elo", -1)) == max_elo
                and int(dataset.attrs.get("max_games", -1))
                == (-1 if max_games is None else max_games)
                and float(dataset.attrs.get("sample_rate", 1.0)) == sample_rate
                and all(name in dataset for name in ("boards", "policies", "values"))
                and dataset["boards"].shape[0] == dataset["policies"].shape[0]
                and dataset["boards"].shape[0] == dataset["values"].shape[0]
            )
    except (OSError, KeyError, TypeError, ValueError):
        return False


def month_is_complete(
    output_root: Path,
    month: str,
    max_games: int | None,
    sample_rates: dict[tuple[int, int], float],
) -> bool:
    return all(
        file_matches_request(
            output_root / f"elo-{min_elo}-{max_elo}" / f"{month}.h5",
            month,
            min_elo,
            max_elo,
            max_games,
            sample_rates[(min_elo, max_elo)],
        )
        for min_elo, max_elo in ELO_BANDS
    )


def _content_range_total(value: str | None) -> tuple[int | None, int | None] | None:
    """Return (range_start, archive_size) from an HTTP Content-Range header."""
    if not value:
        return None
    unsatisfied = re.fullmatch(r"bytes \*/(\d+)", value.strip())
    if unsatisfied:
        return None, int(unsatisfied.group(1))
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+|\*)", value.strip())
    if not match:
        return None
    start = int(match.group(1))
    total = None if match.group(3) == "*" else int(match.group(3))
    return start, total


class RangeRequestsUnsupported(RuntimeError):
    """Raised when the archive host does not honor byte-range requests."""


def _download_archive_single(
    month: str,
    archive_path: Path,
    partial_path: Path,
    url: str,
    retries: int,
) -> Path:
    """Fallback downloader for hosts that do not support byte ranges."""
    with requests.Session() as session:
        for attempt in range(1, retries + 1):
            existing_bytes = partial_path.stat().st_size if partial_path.exists() else 0
            headers = {"Accept-Encoding": "identity"}
            if existing_bytes:
                headers["Range"] = f"bytes={existing_bytes}-"

            response = None
            try:
                response = session.get(
                    url,
                    headers=headers,
                    stream=True,
                    timeout=(20, 120),
                )

                if response.status_code == 416:
                    content_range = _content_range_total(
                        response.headers.get("Content-Range")
                    )
                    response.close()
                    if content_range and content_range[1] == existing_bytes:
                        os.replace(partial_path, archive_path)
                        return archive_path
                    partial_path.unlink(missing_ok=True)
                    continue

                response.raise_for_status()
                range_info = _content_range_total(response.headers.get("Content-Range"))
                if response.status_code == 206:
                    if range_info is None or range_info[0] != existing_bytes:
                        response.close()
                        partial_path.unlink(missing_ok=True)
                        continue
                    mode = "ab" if existing_bytes else "wb"
                    initial_bytes = existing_bytes
                    total_bytes = range_info[1]
                else:
                    # A server that ignores Range returns 200; restart cleanly.
                    mode = "wb"
                    initial_bytes = 0
                    length_header = response.headers.get("Content-Length")
                    total_bytes = (
                        int(length_header)
                        if length_header and length_header.isdigit()
                        else None
                    )

                with tqdm(
                    total=total_bytes,
                    initial=initial_bytes,
                    desc=f"{month} download",
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    dynamic_ncols=True,
                ) as progress:
                    with partial_path.open(mode) as archive_file:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                archive_file.write(chunk)
                                progress.update(len(chunk))
                        archive_file.flush()
                        os.fsync(archive_file.fileno())

                downloaded_bytes = partial_path.stat().st_size
                if total_bytes is not None and downloaded_bytes != total_bytes:
                    raise IOError(
                        f"incomplete response: got {downloaded_bytes:,} of "
                        f"{total_bytes:,} bytes"
                    )

                os.replace(partial_path, archive_path)
                print(f"[downloaded] {archive_path} ({downloaded_bytes / 1e9:.2f} GB)", flush=True)
                return archive_path
            except (requests.RequestException, OSError) as exc:
                if attempt >= retries:
                    raise RuntimeError(
                        f"could not finish downloading {month} after {retries} attempts; "
                        f"partial data is retained at {partial_path}: {exc}"
                    ) from exc
                wait_seconds = min(2 ** (attempt - 1), 30)
                print(
                    f"[retry] {month} download interrupted ({exc}); "
                    f"resuming in {wait_seconds}s ({attempt}/{retries})",
                    flush=True,
                )
                time.sleep(wait_seconds)
            finally:
                if response is not None:
                    response.close()

    raise RuntimeError(f"could not download {month}")


def _merge_ranges(ranges: list[list[int]]) -> list[list[int]]:
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def _missing_ranges(
    total_bytes: int,
    chunk_bytes: int,
    completed: list[list[int]],
) -> list[tuple[int, int]]:
    """Return byte intervals not already recorded as complete."""
    missing: list[tuple[int, int]] = []
    completed = _merge_ranges(completed)
    for chunk_start in range(0, total_bytes, chunk_bytes):
        chunk_end = min(total_bytes - 1, chunk_start + chunk_bytes - 1)
        cursor = chunk_start
        for done_start, done_end in completed:
            if done_end < cursor:
                continue
            if done_start > chunk_end:
                break
            if done_start > cursor:
                missing.append((cursor, min(chunk_end, done_start - 1)))
            cursor = max(cursor, done_end + 1)
            if cursor > chunk_end:
                break
        if cursor <= chunk_end:
            missing.append((cursor, chunk_end))
    return missing


def _write_range_metadata(
    path: Path,
    url: str,
    total_bytes: int,
    chunk_bytes: int,
    completed: list[list[int]],
) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    payload = {
        "url": url,
        "total_bytes": total_bytes,
        "chunk_bytes": chunk_bytes,
        "completed_ranges": _merge_ranges(completed),
    }
    with temporary_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(payload, metadata_file)
        metadata_file.flush()
        os.fsync(metadata_file.fileno())
    os.replace(temporary_path, path)


def _download_archive_parallel(
    month: str,
    archive_path: Path,
    partial_path: Path,
    metadata_path: Path,
    url: str,
    total_bytes: int,
    retries: int,
    connections: int,
    chunk_bytes: int,
) -> Path:
    """Download independent byte ranges concurrently and resume completed ranges."""
    completed: list[list[int]] = []
    metadata = None
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = None

    part_size = partial_path.stat().st_size if partial_path.exists() else 0
    metadata_is_valid = (
        isinstance(metadata, dict)
        and metadata.get("url") == url
        and metadata.get("total_bytes") == total_bytes
        and isinstance(metadata.get("chunk_bytes"), int)
        and metadata["chunk_bytes"] > 0
        and isinstance(metadata.get("completed_ranges"), list)
        and part_size == total_bytes
    )

    if metadata_is_valid:
        try:
            completed = [
                [int(start), int(end)]
                for start, end in metadata["completed_ranges"]
                if 0 <= int(start) <= int(end) < total_bytes
            ]
            chunk_bytes = int(metadata["chunk_bytes"])
        except (TypeError, ValueError):
            metadata_is_valid = False

    if not metadata_is_valid:
        # Older versions left a contiguous, resumable .part file without metadata.
        # Keep that verified prefix and download the remaining ranges in parallel.
        if metadata_path.exists():
            partial_path.unlink(missing_ok=True)
            part_size = 0
        elif part_size == total_bytes:
            os.replace(partial_path, archive_path)
            return archive_path
        elif part_size > total_bytes:
            partial_path.unlink(missing_ok=True)
            part_size = 0

        completed = [[0, part_size - 1]] if part_size else []
        _write_range_metadata(metadata_path, url, total_bytes, chunk_bytes, completed)

    fd = os.open(partial_path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        if os.fstat(fd).st_size != total_bytes:
            os.ftruncate(fd, total_bytes)

        metadata_lock = threading.Lock()
        fallback_write_lock = threading.Lock()
        stop_download = threading.Event()
        completed = _merge_ranges(completed)
        completed_bytes = sum(end - start + 1 for start, end in completed)
        ranges = _missing_ranges(total_bytes, chunk_bytes, completed)
        worker_count = min(connections, max(1, len(ranges)))
        print(
            f"[download] {month}: {worker_count} parallel range connection(s), "
            f"{len(ranges)} range(s) remaining",
            flush=True,
        )

        def write_at(offset: int, data: bytes) -> None:
            view = memoryview(data)
            while view:
                if hasattr(os, "pwrite"):
                    written = os.pwrite(fd, view, offset)
                else:
                    with fallback_write_lock:
                        os.lseek(fd, offset, os.SEEK_SET)
                        written = os.write(fd, view)
                if written <= 0:
                    raise OSError("could not write downloaded range to archive")
                offset += written
                view = view[written:]

        def download_range(start: int, end: int, progress: tqdm) -> None:
            expected_bytes = end - start + 1
            for attempt in range(1, retries + 1):
                if stop_download.is_set():
                    return
                response = None
                received = 0
                try:
                    response = requests.get(
                        url,
                        headers={
                            "Accept-Encoding": "identity",
                            "Range": f"bytes={start}-{end}",
                        },
                        stream=True,
                        timeout=(20, 120),
                    )
                    if response.status_code >= 400:
                        response.raise_for_status()
                    if response.status_code != 206:
                        raise RangeRequestsUnsupported(
                            f"server returned HTTP {response.status_code} for a byte range"
                        )
                    range_info = _content_range_total(
                        response.headers.get("Content-Range")
                    )
                    if range_info != (start, total_bytes):
                        raise IOError(
                            f"unexpected Content-Range for bytes {start}-{end}: "
                            f"{response.headers.get('Content-Range')}"
                        )

                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if stop_download.is_set():
                            if received:
                                progress.update(-received)
                            return
                        if not chunk:
                            continue
                        if received + len(chunk) > expected_bytes:
                            raise IOError("server sent more bytes than requested")
                        write_at(start + received, chunk)
                        received += len(chunk)
                        progress.update(len(chunk))

                    if received != expected_bytes:
                        raise IOError(
                            f"incomplete range: got {received:,} of "
                            f"{expected_bytes:,} bytes"
                        )
                    os.fsync(fd)
                    with metadata_lock:
                        completed.append([start, end])
                        completed[:] = _merge_ranges(completed)
                        _write_range_metadata(
                            metadata_path, url, total_bytes, chunk_bytes, completed
                        )
                    return
                except RangeRequestsUnsupported:
                    if received:
                        progress.update(-received)
                    raise
                except (requests.RequestException, OSError) as exc:
                    if received:
                        progress.update(-received)
                    if attempt >= retries:
                        raise RuntimeError(
                            f"range {start}-{end} failed after {retries} attempts: {exc}"
                        ) from exc
                    wait_seconds = min(2 ** (attempt - 1), 30)
                    print(
                        f"[retry] {month} bytes {start}-{end} interrupted; "
                        f"resuming in {wait_seconds}s ({attempt}/{retries})",
                        flush=True,
                    )
                    time.sleep(wait_seconds)
                finally:
                    if response is not None:
                        response.close()

        with tqdm(
            total=total_bytes,
            initial=completed_bytes,
            desc=f"{month} download",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
        ) as progress:
            executor = ThreadPoolExecutor(max_workers=worker_count)
            futures = []
            try:
                futures = [
                    executor.submit(download_range, start, end, progress)
                    for start, end in ranges
                ]
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                stop_download.set()
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=True, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)

        completed_bytes = sum(end - start + 1 for start, end in _merge_ranges(completed))
        if completed_bytes != total_bytes:
            raise IOError(
                f"incomplete archive: completed {completed_bytes:,} of "
                f"{total_bytes:,} bytes"
            )
        os.fsync(fd)
    finally:
        os.close(fd)

    os.replace(partial_path, archive_path)
    metadata_path.unlink(missing_ok=True)
    print(f"[downloaded] {archive_path} ({total_bytes / 1e9:.2f} GB)", flush=True)
    return archive_path


def download_archive(
    month: str,
    archive_path: Path,
    retries: int,
    connections: int,
    chunk_bytes: int,
) -> Path:
    """Download an archive with parallel ranges and resumable per-range state."""
    if archive_path.is_file():
        print(f"[reuse] Found downloaded archive: {archive_path}", flush=True)
        return archive_path

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
    metadata_path = partial_path.with_name(partial_path.name + ".ranges.json")
    url = LICHESS_URL.format(month=month)

    range_info = None
    supports_ranges = False
    for attempt in range(1, retries + 1):
        response = None
        try:
            response = requests.get(
                url,
                headers={"Accept-Encoding": "identity", "Range": "bytes=0-0"},
                stream=True,
                timeout=(20, 120),
            )
            range_info = _content_range_total(response.headers.get("Content-Range"))
            supports_ranges = (
                response.status_code == 206
                and range_info is not None
                and range_info[0] == 0
                and range_info[1] is not None
                and range_info[1] > 0
            )
            if response.status_code >= 400 and response.status_code != 416:
                response.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt >= retries:
                raise RuntimeError(
                    f"could not check range support for {month} after {retries} "
                    f"attempts: {exc}"
                ) from exc
            wait_seconds = min(2 ** (attempt - 1), 30)
            print(
                f"[retry] {month} range check failed; resuming in "
                f"{wait_seconds}s ({attempt}/{retries})",
                flush=True,
            )
            time.sleep(wait_seconds)
        finally:
            if response is not None:
                response.close()

    if not supports_ranges:
        if metadata_path.exists():
            raise RuntimeError(
                f"the server no longer supports byte ranges for {month}; "
                f"parallel partial data is retained at {partial_path}"
            )
        print(
            f"[download] {month}: server does not support byte ranges; "
            "using one connection",
            flush=True,
        )
        return _download_archive_single(
            month, archive_path, partial_path, url, retries
        )

    assert range_info is not None and range_info[1] is not None
    return _download_archive_parallel(
        month,
        archive_path,
        partial_path,
        metadata_path,
        url,
        range_info[1],
        retries,
        connections,
        chunk_bytes,
    )


def prepare_month(
    month: str,
    archive_path: Path,
    output_root: Path,
    max_games: int | None,
    sample_rates: dict[tuple[int, int], float],
    workers: int,
    overwrite: bool,
) -> int:
    """Parse the local archive into all Elo bands, showing its live progress."""
    print(f"\n[parse] {month}: building all Elo bands from {archive_path}", flush=True)
    prepare_elo_bands(
        month,
        archive_path,
        output_root,
        max_games=max_games,
        sample_rates=sample_rates,
        workers=workers,
        overwrite=overwrite,
    )
    return 0


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    parser = argparse.ArgumentParser(
        description=(
            "Download monthly Lichess archives one at a time and parse each "
            "into four Elo-band datasets (up to 2200 Elo)."
        )
    )
    parser.add_argument("--year", type=int, required=True, help="Year to process, e.g. 2024")
    parser.add_argument("--start-month", type=int, default=1, help="First month (1-12)")
    parser.add_argument("--end-month", type=int, default=12, help="Last month (1-12)")
    parser.add_argument(
        "--month-count", type=int, default=None,
        help="Process at most this many months from the selected range (for example, 1)",
    )
    parser.add_argument(
        "--max-games", type=int, default=None,
        help="Optional cap per Elo band per month; by default process every qualifying game",
    )
    parser.add_argument(
        "--workers", type=int, default=available_cpu_count(),
        help=f"CPU worker processes for game encoding (default: {available_cpu_count()})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output root (default: data/processed; creates one directory per Elo band)",
    )
    parser.add_argument(
        "--archive-dir", type=Path, default=None,
        help="Keep in-progress archives here (default: <output-dir>/.archives)",
    )
    parser.add_argument(
        "--download-retries", type=int, default=8,
        help="Attempts per month download; partial files resume between attempts (default: 8)",
    )
    parser.add_argument(
        "--download-connections", type=int, default=10,
        help="Parallel HTTP range connections per archive (default: 10)",
    )
    parser.add_argument(
        "--download-chunk-mb", type=int, default=64,
        help="Size of each independently resumable range in MiB (default: 64)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Rebuild complete monthly band files",
    )
    args = parser.parse_args()

    if not 1 <= args.start_month <= 12 or not 1 <= args.end_month <= 12:
        parser.error("--start-month and --end-month must be from 1 through 12")
    if args.start_month > args.end_month:
        parser.error("--start-month must not be after --end-month")
    if args.month_count is not None and args.month_count < 1:
        parser.error("--month-count must be positive")
    if args.max_games is not None and args.max_games < 1:
        parser.error("--max-games must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.download_retries < 1:
        parser.error("--download-retries must be positive")
    if args.download_connections < 1:
        parser.error("--download-connections must be positive")
    if args.download_chunk_mb < 1:
        parser.error("--download-chunk-mb must be positive")

    output_root = args.output_dir or PROJECT_ROOT / "data" / "processed"
    output_root = output_root.resolve()
    archive_dir = args.archive_dir or output_root / ".archives"
    archive_dir = archive_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    archive_dir.mkdir(parents=True, exist_ok=True)
    print(f"Controller PID: {os.getpid()}", flush=True)

    months = [
        f"{args.year}-{number:02d}"
        for number in range(args.start_month, args.end_month + 1)
    ]
    if args.month_count is not None:
        months = months[:args.month_count]
    print(f"Writing Elo-band datasets under: {output_root}", flush=True)
    print(f"Temporary monthly archives: {archive_dir}", flush=True)
    print(f"Elo bands: {', '.join(f'{low}-{high}' for low, high in ELO_BANDS)}", flush=True)
    print(
        f"Months: {months[0]} through {months[-1]} ({len(months)} month(s); "
        "next archive downloads during parsing)",
        flush=True,
    )
    print(f"CPU encoding workers: {args.workers}", flush=True)

    completed_months = 0
    failures: list[str] = []
    sample_rates = DEFAULT_ELO_SAMPLE_RATES.copy()
    print(
        "Game sample rates: "
        + ", ".join(
            f"{low}-{high}:{sample_rates[(low, high)]:.0%}"
            for low, high in ELO_BANDS
        ),
        flush=True,
    )

    def archive_paths(month: str) -> tuple[Path, Path, Path]:
        archive_path = archive_dir / f"lichess_db_standard_rated_{month}.pgn.zst"
        partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
        metadata_path = partial_path.with_name(partial_path.name + ".ranges.json")
        return archive_path, partial_path, metadata_path

    def download_month_archive(month: str) -> Path:
        archive_path, _, _ = archive_paths(month)
        return download_archive(
            month,
            archive_path,
            args.download_retries,
            args.download_connections,
            args.download_chunk_mb * 1024 * 1024,
        )

    def cleanup_archive(month: str) -> None:
        archive_path, partial_path, metadata_path = archive_paths(month)
        archive_path.unlink(missing_ok=True)
        partial_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

    download_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="month-download")
    pending_download: tuple[str, Future[Path]] | None = None
    try:
        for index, month in enumerate(months):
            archive_path, _, _ = archive_paths(month)
            if not args.overwrite and month_is_complete(
                output_root, month, args.max_games, sample_rates
            ):
                cleanup_archive(month)
                print(f"[skip] All band files for {month} are already complete", flush=True)
                completed_months += 1
                continue

            try:
                if pending_download is not None and pending_download[0] == month:
                    pending_download[1].result()
                    pending_download = None
                else:
                    download_month_archive(month)

                # Start fetching the next needed month while this one is parsed.
                next_month = next(
                    (
                        candidate
                        for candidate in months[index + 1:]
                        if args.overwrite
                        or not month_is_complete(
                            output_root, candidate, args.max_games, sample_rates
                        )
                    ),
                    None,
                )
                if next_month is not None:
                    pending_download = (
                        next_month,
                        download_pool.submit(download_month_archive, next_month),
                    )

                return_code = prepare_month(
                    month,
                    archive_path,
                    output_root,
                    args.max_games,
                    sample_rates,
                    args.workers,
                    args.overwrite,
                )
                if return_code != 0:
                    failures.append(f"{month} (prepare_data.py exited {return_code})")
                    print(f"[failed] {month}; archive retained for rerun", flush=True)
                    break

                if not month_is_complete(
                    output_root, month, args.max_games, sample_rates
                ):
                    failures.append(f"{month} (one or more Elo-band files are incomplete)")
                    print(f"[failed] {month}; archive retained for rerun", flush=True)
                    break

                cleanup_archive(month)
                completed_months += 1
                print(f"[done] {month}: all Elo bands complete; archive removed", flush=True)
            except (OSError, RuntimeError, requests.RequestException) as exc:
                failures.append(f"{month} ({exc})")
                print(f"[failed] {month}; archive/partial download retained for rerun", flush=True)
                break
            except KeyboardInterrupt:
                print("\nInterrupted. The current archive and completed band files are retained.", flush=True)
                raise SystemExit(130)
    finally:
        if pending_download is not None:
            pending_download[1].cancel()
        download_pool.shutdown(wait=True, cancel_futures=True)

    print(f"\nFinished: {completed_months}/{len(months)} months complete.")
    if failures:
        print("Resume by running the same command again:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    print(
        "Train on a band with: "
        f"python -m src.train --data \"{output_root / 'elo-1000-1200'}\" ..."
    )


if __name__ == "__main__":
    main()
