"""Prepare Lichess PGN archives as HDF5 chess datasets."""

import argparse
import hashlib
import io
import os
import signal
import sys
import time
from collections import deque
from concurrent.futures import (
    ALL_COMPLETED,
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    wait,
)
from pathlib import Path
from typing import Iterator, Optional, Tuple

import chess
import chess.pgn
import h5py
import numpy as np
import requests
import zstandard as zstd
from tqdm import tqdm

from .paths import PROJECT_ROOT

LICHESS_URL = (
    "https://database.lichess.org/standard/"
    "lichess_db_standard_rated_{month}.pgn.zst"
)

# The monthly downloader builds these bands from one archive scan.
ELO_BANDS = (
    (1000, 1200),
    (1200, 1500),
    (1500, 1800),
    (1800, 2200),
)

# Sampling is deterministic per month so reruns produce the same data.
DEFAULT_ELO_SAMPLE_RATES = {
    (1000, 1200): 1.0,
    (1200, 1500): 1.0,
    (1500, 1800): 1.0,
    (1800, 2200): 1.0,
}

# Piece types in python-chess: PAWN=1 .. KING=6
# We map (color, piece_type) to a plane index.
PIECE_PLANE = {
    (chess.WHITE, chess.PAWN):   0,
    (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK):   3,
    (chess.WHITE, chess.QUEEN):  4,
    (chess.WHITE, chess.KING):   5,
    (chess.BLACK, chess.PAWN):   6,
    (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK):   9,
    (chess.BLACK, chess.QUEEN):  10,
    (chess.BLACK, chess.KING):   11,
}

FLUSH_BATCH = 50_000


def move_to_index(move: chess.Move) -> int:
    """Convert a chess.Move to a flat index in [0, 4095].

    Encoding: from_square * 64 + to_square.

    Simplification: underpromotions (to knight, bishop, rook) are mapped
    into the same 0–4095 range as normal moves because the policy head
    outputs 4096 logits.  Queen promotions are the default and are also
    handled identically.  This loses a small amount of information for
    the rare underpromotion moves, but keeps the action space compact.
    """
    return move.from_square * 64 + move.to_square


def index_to_move(index: int, promotion: Optional[int] = None) -> chess.Move:
    """Convert a flat index back to a chess.Move.

    Args:
        index: Integer in [0, 4095].
        promotion: Optional piece type for promotion moves (e.g. chess.QUEEN).
    """
    from_sq = index // 64
    to_sq = index % 64
    return chess.Move(from_sq, to_sq, promotion=promotion)


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode a chess.Board as an (18, 8, 8) float16 tensor.

    See module docstring for the full plane layout.
    """
    tensor = np.zeros((18, 8, 8), dtype=np.float16)

    # --- Planes 0–11: piece placement ---
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            plane = PIECE_PLANE[(piece.color, piece.piece_type)]
            # python-chess squares: a1=0 .. h8=63
            # Map to (row, col) where row 0 = rank 1
            row = square // 8
            col = square % 8
            tensor[plane, row, col] = 1.0

    # --- Planes 12–15: castling rights ---
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[12, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[14, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0

    # --- Plane 16: en passant target square ---
    if board.ep_square is not None:
        row = board.ep_square // 8
        col = board.ep_square % 8
        tensor[16, row, col] = 1.0

    # --- Plane 17: side to move ---
    if board.turn == chess.WHITE:
        tensor[17, :, :] = 1.0
    # Black to move → plane stays all zeros

    return tensor


class ProgressRawReader:
    """Count compressed bytes as the Zstandard reader pulls them from its source."""

    def __init__(self, source: object, progress: tqdm) -> None:
        self._source = source
        self._progress = progress

    def read(self, size: int = -1) -> bytes:
        data = self._source.read(size)  # type: ignore[attr-defined]
        if data:
            self._progress.update(len(data))
        return data

    def close(self) -> None:
        self._source.close()  # type: ignore[attr-defined]


class ZstdPgnStream(io.RawIOBase):
    """A read-only binary stream that decompresses a Zstandard archive.

    Wraps an HTTP response or local file and a zstd streaming decompressor to
    produce bytes suitable for TextIOWrapper → chess.pgn.read_game().
    """

    def __init__(self, source: object, progress: tqdm, on_close=None) -> None:
        self._dctx = zstd.ZstdDecompressor()
        self._reader = self._dctx.stream_reader(ProgressRawReader(source, progress))
        self._on_close = on_close

    def readable(self) -> bool:
        return True

    def readinto(self, b: bytearray) -> int:
        """Read decompressed bytes into buffer *b*."""
        data = self._reader.read(len(b))
        n = len(data)
        b[:n] = data
        return n

    def close(self) -> None:
        if self.closed:
            return
        try:
            self._reader.close()
        finally:
            try:
                if self._on_close is not None:
                    self._on_close()
            finally:
                super().close()


def open_pgn_stream(
    month: str,
    archive_path: Optional[Path] = None,
) -> tuple[io.TextIOWrapper, tqdm]:
    """Open a streaming, decompressed PGN text reader for the given month.

    If archive_path is provided, read a previously downloaded local .zst file.
    Otherwise stream the archive directly from Lichess.
    """
    if archive_path is not None:
        archive_path = Path(archive_path)
        source = archive_path.open("rb")
        total_bytes = archive_path.stat().st_size
        progress = tqdm(
            total=total_bytes,
            desc=f"{month} archive read",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
        )
        binary_stream = ZstdPgnStream(source, progress)
        return io.TextIOWrapper(binary_stream, encoding="utf-8", errors="replace"), progress

    url = LICHESS_URL.format(month=month)
    print(f"[download] Connecting to {url}")

    session = requests.Session()
    resp = session.get(
        url,
        headers={"Accept-Encoding": "identity"},
        stream=True,
        timeout=60,
    )
    resp.raise_for_status()

    total_header = resp.headers.get("Content-Length")
    total_bytes = int(total_header) if total_header and total_header.isdigit() else None
    download_progress = tqdm(
        total=total_bytes,
        desc=f"{month} download",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
    )
    binary_stream = ZstdPgnStream(
        resp.raw,
        download_progress,
        on_close=lambda: (resp.close(), session.close()),
    )
    text_stream = io.TextIOWrapper(binary_stream, encoding="utf-8", errors="replace")
    return text_stream, download_progress


def parse_elo(header_value: Optional[str]) -> int:
    """Safely parse an ELO rating from a PGN header, returning 0 on failure."""
    if header_value is None:
        return 0
    try:
        return int(header_value)
    except (ValueError, TypeError):
        return 0


def is_rapid_or_classical(event: str) -> bool:
    """Check if the Event header indicates a rapid or classical game.

    Lichess PGN Event headers contain the time-control category, e.g.:
      "Rated Rapid game", "Rated Classical game"
    We reject bullet, ultraBullet, blitz, and correspondence.
    """
    event_lower = event.lower()
    return "rapid" in event_lower or "classical" in event_lower


def passes_result_time_filter(game: chess.pgn.Game) -> bool:
    """Check the inexpensive result and time-control filters."""
    headers = game.headers

    # Result must be decisive or draw (not ongoing / abandoned)
    result = headers.get("Result", "*")
    if result == "*":
        return False

    # Time control category
    event = headers.get("Event", "")
    return is_rapid_or_classical(event)


def has_minimum_game_length(game: chess.pgn.Game) -> bool:
    """Require at least ten half-moves."""
    return len(list(game.mainline_moves())) >= 10


def passes_quality_filter(game: chess.pgn.Game) -> bool:
    """Return True if a game meets the shared quality filters."""
    return passes_result_time_filter(game) and has_minimum_game_length(game)


def game_matches_elo_band(
    game: chess.pgn.Game,
    min_elo: int,
    max_elo: Optional[int] = None,
) -> bool:
    """Return whether both players' ratings are within an inclusive Elo band."""
    white_elo = parse_elo(game.headers.get("WhiteElo"))
    black_elo = parse_elo(game.headers.get("BlackElo"))
    if white_elo < min_elo or black_elo < min_elo:
        return False
    if max_elo is not None and (white_elo > max_elo or black_elo > max_elo):
        return False
    return True


def passes_filter(
    game: chess.pgn.Game,
    min_elo: int,
    max_elo: Optional[int] = None,
) -> bool:
    """Return True if a game passes the quality and Elo filters."""
    # Check the inexpensive Elo headers before walking the move list.
    if not game_matches_elo_band(game, min_elo, max_elo):
        return False
    return passes_quality_filter(game)


def outcome_value(result: str, side: chess.Color) -> int:
    """Compute the game outcome from *side*'s perspective.

    Returns +1 (win), 0 (draw), or −1 (loss).
    """
    if result == "1/2-1/2":
        return 0
    elif result == "1-0":
        return 1 if side == chess.WHITE else -1
    elif result == "0-1":
        return 1 if side == chess.BLACK else -1
    return 0  # fallback


def process_game(
    game: chess.pgn.Game,
    moves: Optional[list[chess.Move]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract all (board, policy, value) triples from a single game.

    Returns:
        boards:   float16 array of shape (M, 18, 8, 8)
        policies: int16 array of shape (M,)
        values:   int8 array of shape (M,)
    where M is the number of positions (= number of moves in the game).
    """
    result = game.headers.get("Result", "*")
    if moves is None:
        moves = list(game.mainline_moves())

    boards_list = []
    policies_list = []
    values_list = []

    board = game.board()  # starting position

    for move in moves:
        # Encode the current position *before* the move is played
        tensor = encode_board(board)
        move_idx = move_to_index(move)
        value = outcome_value(result, board.turn)

        boards_list.append(tensor)
        policies_list.append(move_idx)
        values_list.append(value)

        board.push(move)

    boards = np.stack(boards_list, axis=0)                       # (M, 18, 8, 8)
    policies = np.array(policies_list, dtype=np.int16)           # (M,)
    values = np.array(values_list, dtype=np.int8)                # (M,)

    return boards, policies, values


def available_cpu_count() -> int:
    """Return CPUs available to this process, respecting container limits."""
    cpu_count = getattr(os, "process_cpu_count", os.cpu_count)()
    return max(1, cpu_count or 1)


def _handle_sigterm(_signum, _frame) -> None:
    raise KeyboardInterrupt


def _keep_sample(
    month: str,
    game_number: int,
    band: tuple[int, int],
    rate: float,
) -> bool:
    """Make a stable per-game sampling decision independent of worker order."""
    if rate >= 1.0:
        return True
    sample_key = f"{month}:{game_number}:{band[0]}-{band[1]}".encode("ascii")
    sample_value = int.from_bytes(
        hashlib.blake2b(sample_key, digest_size=8).digest(), "big"
    )
    return sample_value < rate * (1 << 64)


def _process_game_batch_worker(
    records: list[tuple[int, str]],
    month: str,
    bands: tuple[tuple[int, int], ...],
    sample_rates: dict[tuple[int, int], float],
) -> list[
    tuple[
        int,
        list[tuple[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]],
    ]
]:
    """Parse, filter, sample, and encode raw PGN records in a worker process."""
    results = []
    for game_number, record in records:
        game = chess.pgn.read_game(io.StringIO(record))
        encoded_bands = []
        if game is not None and passes_result_time_filter(game):
            moves = list(game.mainline_moves())
            if len(moves) >= 10:
                matching_bands = [
                    band
                    for band in bands
                    if game_matches_elo_band(game, band[0], band[1])
                    and _keep_sample(month, game_number, band, sample_rates[band])
                ]
                if matching_bands:
                    encoded = process_game(game, moves)
                    encoded_bands = [(band, encoded) for band in matching_bands]
        results.append((game_number, encoded_bands))
    return results


def _process_single_batch_worker(
    games: list[chess.pgn.Game],
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Encode a batch of games for the single-band preprocessing path."""
    return [process_game(game) for game in games]


class HDF5Writer:
    """Incrementally writes position data to a chunked, gzip-compressed HDF5 file."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.f = h5py.File(path, "w")
        self.f.attrs["complete"] = False

        self.ds_boards = self.f.create_dataset(
            "boards",
            shape=(0, 18, 8, 8),
            maxshape=(None, 18, 8, 8),
            dtype="float16",
            chunks=(256, 18, 8, 8),
            compression="gzip",
            compression_opts=4,
        )
        self.ds_policies = self.f.create_dataset(
            "policies",
            shape=(0,),
            maxshape=(None,),
            dtype="int16",
            chunks=(4096,),
            compression="gzip",
            compression_opts=4,
        )
        self.ds_values = self.f.create_dataset(
            "values",
            shape=(0,),
            maxshape=(None,),
            dtype="int8",
            chunks=(4096,),
            compression="gzip",
            compression_opts=4,
        )

        self._buf_boards: list[np.ndarray] = []
        self._buf_policies: list[np.ndarray] = []
        self._buf_values: list[np.ndarray] = []
        self._buf_count = 0
        self.total_positions = 0

    def add(self, boards: np.ndarray, policies: np.ndarray, values: np.ndarray) -> None:
        """Buffer position data from one game."""
        self._buf_boards.append(boards)
        self._buf_policies.append(policies)
        self._buf_values.append(values)
        self._buf_count += len(boards)

        if self._buf_count >= FLUSH_BATCH:
            self.flush()

    def flush(self) -> None:
        """Flush buffered positions to HDF5."""
        if self._buf_count == 0:
            return

        all_boards = np.concatenate(self._buf_boards, axis=0)
        all_policies = np.concatenate(self._buf_policies, axis=0)
        all_values = np.concatenate(self._buf_values, axis=0)

        n = len(all_boards)
        old_len = self.ds_boards.shape[0]
        new_len = old_len + n

        self.ds_boards.resize(new_len, axis=0)
        self.ds_policies.resize(new_len, axis=0)
        self.ds_values.resize(new_len, axis=0)

        self.ds_boards[old_len:new_len] = all_boards
        self.ds_policies[old_len:new_len] = all_policies
        self.ds_values[old_len:new_len] = all_values

        self.total_positions = new_len

        self._buf_boards.clear()
        self._buf_policies.clear()
        self._buf_values.clear()
        self._buf_count = 0

    def close(self, complete: bool = True) -> None:
        """Flush remaining data, mark completion status, and close the file."""
        self.flush()
        self.f.attrs["complete"] = complete
        self.f.close()

    @property
    def file_size_mb(self) -> float:
        """Return current file size in megabytes."""
        try:
            return os.path.getsize(self.path) / (1024 * 1024)
        except OSError:
            return 0.0


def iterate_games(text_stream: io.TextIOWrapper) -> Iterator[chess.pgn.Game]:
    """Yield chess.pgn.Game objects from a PGN text stream."""
    while True:
        game = chess.pgn.read_game(text_stream)
        if game is None:
            break
        yield game


def iterate_pgn_records(text_stream: io.TextIOWrapper) -> Iterator[str]:
    """Yield raw Lichess PGN records without parsing them in the parent process."""
    record_lines: list[str] = []
    for line in text_stream:
        if line.startswith("[Event ") and record_lines:
            record = "".join(record_lines)
            if record.strip():
                yield record
            record_lines = []
        record_lines.append(line)

    record = "".join(record_lines)
    if record.strip():
        yield record


def output_matches_request(
    path: Path,
    month: str,
    min_elo: int,
    max_elo: int,
    max_games: Optional[int],
    sample_rate: float = 1.0,
) -> bool:
    """Check whether a segment output is complete for the requested filters."""
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


def prepare_elo_bands(
    month: str,
    archive_path: Path,
    output_root: Path,
    max_games: Optional[int] = None,
    sample_rates: Optional[dict[tuple[int, int], float]] = None,
    workers: Optional[int] = None,
    overwrite: bool = False,
) -> None:
    """Parse one local monthly archive into all configured Elo-band datasets."""
    rates = DEFAULT_ELO_SAMPLE_RATES.copy()
    if sample_rates:
        rates.update(sample_rates)
    if any(not 0.0 < rate <= 1.0 for rate in rates.values()):
        raise ValueError("Elo-band sample rates must be greater than 0 and at most 1")
    worker_count = available_cpu_count() if workers is None else workers
    if worker_count < 1:
        raise ValueError("workers must be positive")
    output_root.mkdir(parents=True, exist_ok=True)
    output_paths = {
        band: output_root / f"elo-{band[0]}-{band[1]}" / f"{month}.h5"
        for band in ELO_BANDS
    }
    writers: dict[tuple[int, int], HDF5Writer] = {}
    kept_games = {band: 0 for band in ELO_BANDS}
    total_positions = {band: 0 for band in ELO_BANDS}

    for min_elo, max_elo in ELO_BANDS:
        output_path = output_paths[(min_elo, max_elo)]
        if (
            not overwrite
            and output_matches_request(
                output_path,
                month,
                min_elo,
                max_elo,
                max_games,
                rates[(min_elo, max_elo)],
            )
        ):
            print(f"[skip] {output_path} is already complete", flush=True)
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = HDF5Writer(str(output_path))
        writer.f.attrs["month"] = month
        writer.f.attrs["min_elo"] = min_elo
        writer.f.attrs["max_elo"] = max_elo
        writer.f.attrs["max_games"] = -1 if max_games is None else max_games
        writer.f.attrs["sample_rate"] = rates[(min_elo, max_elo)]
        writers[(min_elo, max_elo)] = writer

    if not writers:
        print(f"[skip] All Elo bands for {month} are already complete", flush=True)
        return

    print(
        f"[info] Parsing and encoding PGN records with {worker_count} CPU worker(s).",
        flush=True,
    )

    try:
        text_stream, archive_progress = open_pgn_stream(month, archive_path)
    except BaseException:
        for writer in writers.values():
            writer.close(complete=False)
        raise
    parsed_progress = tqdm(
        desc=f"{month} parsed",
        unit="games",
        position=1,
        dynamic_ncols=True,
    )
    total_games = 0
    completed = False
    scan_finished = False
    stop_early = False
    start_time = time.time()
    pending: deque[Future] = deque()
    batch: list[tuple[int, str]] = []
    active_bands = tuple(writers)
    active_rates = {band: rates[band] for band in active_bands}
    executor = (
        ProcessPoolExecutor(max_workers=worker_count) if worker_count > 1 else None
    )
    max_pending = max(1, worker_count * 2)
    batch_size = min(8, worker_count)

    def write_results(
        results: list[
            tuple[
                int,
                list[
                    tuple[
                        tuple[int, int],
                        tuple[np.ndarray, np.ndarray, np.ndarray],
                    ]
                ],
            ]
        ],
    ) -> None:
        for _game_number, encoded_bands in results:
            for band, (boards, policies, values) in encoded_bands:
                if max_games is None or kept_games[band] < max_games:
                    writers[band].add(boards, policies, values)
                    kept_games[band] += 1
                    total_positions[band] += len(boards)

    def submit_batch() -> None:
        if executor is not None and batch:
            pending.append(
                executor.submit(
                    _process_game_batch_worker,
                    batch.copy(),
                    month,
                    active_bands,
                    active_rates,
                )
            )
            batch.clear()

    def drain_one() -> None:
        if not pending:
            return
        future = pending.popleft()
        write_results(future.result())

    try:
        for record in iterate_pgn_records(text_stream):
            total_games += 1
            parsed_progress.update(1)

            if executor is None:
                write_results(
                    _process_game_batch_worker(
                        [(total_games, record)], month, active_bands, active_rates
                    )
                )
            else:
                batch.append((total_games, record))
                if len(batch) >= batch_size:
                    submit_batch()

            if executor is not None and len(pending) >= max_pending:
                drain_one()

            if total_games % 1_000 == 0:
                parsed_progress.set_postfix(
                    kept=" ".join(
                        f"{low}-{high}:{kept_games[(low, high)]:,}"
                        for low, high in writers
                    ),
                    queued_batches=len(pending),
                    refresh=False,
                )

            if max_games is not None and all(
                kept_games[band] >= max_games for band in writers
            ):
                print(f"[info] Reached --max-games cap ({max_games}) for all bands.")
                scan_finished = True
                stop_early = True
                batch.clear()
                break
        else:
            scan_finished = True

        if executor is not None and not stop_early:
            submit_batch()
            while pending:
                drain_one()
        completed = scan_finished
    except KeyboardInterrupt:
        completed = False
        stop_early = True
        batch.clear()
        print(
            "\n[interrupted] Saving completed batches and stopping pending workers; "
            "keep the archive to resume.",
            flush=True,
        )
    finally:
        try:
            if executor is not None:
                executor.shutdown(
                    wait=True,
                    cancel_futures=(not completed or stop_early),
                )
            for writer in writers.values():
                writer.close(complete=completed)
        finally:
            text_stream.close()
            parsed_progress.close()
            archive_progress.close()

    print(f"[done] {month}: scanned {total_games:,} games in {time.time() - start_time:.1f}s")
    for band, writer in writers.items():
        print(
            f"  {band[0]}-{band[1]}: {kept_games[band]:,} games, "
            f"{total_positions[band]:,} positions → {output_paths[band]}"
        )


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    parser = argparse.ArgumentParser(
        description="Preprocess Lichess PGN data for AlphaZero training."
    )
    parser.add_argument(
        "--month", required=True,
        help="Month identifier, e.g. '2024-01'",
    )
    parser.add_argument(
        "--archive", type=Path, default=None,
        help="Read this already-downloaded .pgn.zst archive instead of fetching it",
    )
    parser.add_argument(
        "--all-elo-bands", action="store_true",
        help="Parse one local archive into all four Elo-band datasets (up to 2200 Elo)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Root folder for --all-elo-bands outputs (default: data/processed)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Rebuild complete outputs when using --all-elo-bands",
    )
    parser.add_argument(
        "--min-elo", type=int, default=2000,
        help="Minimum ELO for both players (default: 2000)",
    )
    parser.add_argument(
        "--max-elo", type=int, default=None,
        help="Optional maximum ELO for both players, inclusive",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output HDF5 file path (default: data/processed/chess_data_{month}.h5)",
    )
    parser.add_argument(
        "--max-games", type=int, default=None,
        help="Maximum number of games to process (for testing)",
    )
    parser.add_argument(
        "--workers", type=int, default=available_cpu_count(),
        help=f"CPU worker processes for game encoding (default: {available_cpu_count()})",
    )
    args = parser.parse_args()
    if args.max_elo is not None and args.max_elo < args.min_elo:
        parser.error("--max-elo must be greater than or equal to --min-elo")
    if args.max_games is not None and args.max_games < 1:
        parser.error("--max-games must be positive")
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.all_elo_bands:
        if args.archive is None:
            parser.error("--all-elo-bands requires --archive")
        if args.output is not None:
            parser.error("--output cannot be used with --all-elo-bands")
        output_root = args.output_dir or PROJECT_ROOT / "data" / "processed"
        if not output_root.is_absolute():
            output_root = PROJECT_ROOT / output_root
        prepare_elo_bands(
            args.month,
            args.archive,
            output_root,
            max_games=args.max_games,
            workers=args.workers,
            overwrite=args.overwrite,
        )
        return
    elif (
        args.output_dir is not None
        or args.overwrite
    ):
        parser.error(
            "--output-dir and --overwrite require --all-elo-bands"
        )

    output_path = (
        Path(args.output)
        if args.output
        else PROJECT_ROOT / "data" / "processed" / f"chess_data_{args.month}.h5"
    )
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Lichess Data Preparation Pipeline")
    print("=" * 70)
    print(f"  Month       : {args.month}")
    print(f"  Min ELO     : {args.min_elo}")
    print(f"  Max ELO     : {args.max_elo if args.max_elo is not None else 'unlimited'}")
    print(f"  Output      : {output_path}")
    print(f"  Max games   : {args.max_games or 'unlimited'}")
    print("=" * 70)

    text_stream, download_progress = open_pgn_stream(args.month, args.archive)

    writer = HDF5Writer(str(output_path))
    writer.f.attrs["month"] = args.month
    writer.f.attrs["min_elo"] = args.min_elo
    writer.f.attrs["max_elo"] = -1 if args.max_elo is None else args.max_elo
    writer.f.attrs["max_games"] = -1 if args.max_games is None else args.max_games
    writer.f.attrs["sample_rate"] = 1.0

    total_games = 0       # all games seen
    kept_games = 0        # games passing filters
    total_positions = 0   # running position count
    start_time = time.time()
    completed = False
    scan_finished = False
    parsed_progress = tqdm(
        desc=f"{args.month} parsed",
        unit="games",
        position=1,
        dynamic_ncols=True,
    )
    worker_count = args.workers
    print(f"[info] Encoding games with {worker_count} CPU worker(s).", flush=True)
    executor = (
        ProcessPoolExecutor(max_workers=worker_count) if worker_count > 1 else None
    )
    pending: dict[Future, None] = {}
    batch: list[chess.pgn.Game] = []
    reserved_games = 0
    max_pending = max(1, worker_count * 2)
    batch_size = min(8, worker_count)

    def save_single_result(future: Future) -> None:
        nonlocal kept_games, total_positions, reserved_games
        pending.pop(future)
        results = future.result()
        reserved_games -= len(results)
        for boards, policies, values in results:
            writer.add(boards, policies, values)
            kept_games += 1
            total_positions += len(boards)

    def submit_single_batch() -> None:
        if executor is not None and batch:
            pending[executor.submit(_process_single_batch_worker, batch.copy())] = None
            batch.clear()

    def drain_single_pending(wait_for_all: bool = False) -> None:
        if not pending:
            return
        done, _ = wait(
            tuple(pending),
            return_when=ALL_COMPLETED if wait_for_all else FIRST_COMPLETED,
        )
        for future in done:
            save_single_result(future)

    try:
        for game in iterate_games(text_stream):
            total_games += 1
            parsed_progress.update(1)

            if not passes_filter(game, args.min_elo, args.max_elo):
                if total_games % 1_000 == 0:
                    parsed_progress.set_postfix(
                        kept=f"{kept_games:,}",
                        positions=f"{total_positions:,}",
                        refresh=False,
                    )
                continue

            # Keep a bounded queue so workers stay busy without growing memory use.
            if executor is None:
                boards, policies, values = process_game(game)
                writer.add(boards, policies, values)
                kept_games += 1
                total_positions += len(boards)
            else:
                if args.max_games is None or kept_games + reserved_games < args.max_games:
                    batch.append(game)
                    reserved_games += 1
                    if len(batch) >= batch_size:
                        submit_single_batch()
                if len(pending) >= max_pending:
                    submit_single_batch()
                    drain_single_pending()

            if total_games % 1_000 == 0:
                parsed_progress.set_postfix(
                    kept=f"{kept_games:,}",
                    positions=f"{total_positions:,}",
                    refresh=False,
                )

            if args.max_games and kept_games + reserved_games >= args.max_games:
                if executor is not None:
                    submit_single_batch()
                    drain_single_pending(wait_for_all=True)
                if kept_games >= args.max_games:
                    print(f"[info] Reached --max-games cap ({args.max_games}). Stopping.")
                    scan_finished = True
                    break
        else:
            scan_finished = True

        if executor is not None:
            submit_single_batch()
            drain_single_pending(wait_for_all=True)
        completed = scan_finished

    except KeyboardInterrupt:
        print(
            "\n[interrupted] Saving completed batches and stopping pending workers...",
            flush=True,
        )

    finally:
        try:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=not completed)
            writer.close(complete=completed)
        finally:
            text_stream.close()
            parsed_progress.close()
            download_progress.close()

    elapsed = time.time() - start_time

    print()
    print("=" * 70)
    print("  Processing Complete" if completed else "  Partial Data Saved")
    print("=" * 70)
    print(f"  Total games scanned : {total_games:,}")
    print(f"  Games kept          : {kept_games:,}")
    print(f"  Total positions     : {writer.total_positions:,}")
    print(f"  Output file         : {output_path}")
    print(f"  File size           : {writer.file_size_mb:.1f} MB")
    print(f"  Elapsed time        : {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
