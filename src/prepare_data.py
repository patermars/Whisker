"""
prepare_data.py — Download & Preprocess Lichess Open Database for AlphaZero Training
=====================================================================================

Streams a month's PGN archive from https://database.lichess.org, filters for
high-quality rapid/classical games, encodes every board position into an
(18, 8, 8) tensor, and writes the result to a compressed HDF5 file.

HDF5 Schema
-----------
  boards   : float16, shape (N, 18, 8, 8)  — board state tensors
  policies : int16,   shape (N,)            — UCI move index (from_sq * 64 + to_sq)
  values   : int8,    shape (N,)            — game outcome from side-to-move (+1/0/−1)

Tensor Layout (18 planes, each 8×8)
------------------------------------
  Plane  0 : white pawns          Plane  6 : black pawns
  Plane  1 : white knights        Plane  7 : black knights
  Plane  2 : white bishops        Plane  8 : black bishops
  Plane  3 : white rooks          Plane  9 : black rooks
  Plane  4 : white queens         Plane 10 : black queens
  Plane  5 : white king           Plane 11 : black king
  Plane 12 : white kingside castling  (all 1s if right exists)
  Plane 13 : white queenside castling
  Plane 14 : black kingside castling
  Plane 15 : black queenside castling
  Plane 16 : en passant target square (1 on the square, 0 elsewhere)
  Plane 17 : side to move (all 1s = white, all 0s = black)

CLI Usage
---------
  python -m whisker.prepare_data --month 2024-01 --min-elo 2000 --output data.h5
  python -m whisker.prepare_data --month 2024-01 --max-games 5000   # quick test run
"""

import argparse
import io
import os
import sys
import time
from pathlib import Path
from typing import Iterator, Optional, Tuple

import chess
import chess.pgn
import h5py
import numpy as np
import requests
import zstandard as zstd

from .paths import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Lichess archive URL template
LICHESS_URL = (
    "https://database.lichess.org/standard/"
    "lichess_db_standard_rated_{month}.pgn.zst"
)

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

# In-memory batch size before flushing to HDF5
FLUSH_BATCH = 50_000


# ---------------------------------------------------------------------------
# UCI Move Encoding
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Board Encoding
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Streaming PGN Reader
# ---------------------------------------------------------------------------

class ZstdPgnStream(io.RawIOBase):
    """A read-only binary stream that decompresses a Zstandard HTTP response.

    Wraps a requests streaming response and a zstd streaming decompressor
    to produce raw bytes suitable for TextIOWrapper → chess.pgn.read_game().
    """

    def __init__(self, response: requests.Response) -> None:
        self._dctx = zstd.ZstdDecompressor()
        # stream_reader wraps the raw socket iterator
        self._reader = self._dctx.stream_reader(response.raw)

    def readable(self) -> bool:
        return True

    def readinto(self, b: bytearray) -> int:
        """Read decompressed bytes into buffer *b*."""
        data = self._reader.read(len(b))
        n = len(data)
        b[:n] = data
        return n


def open_pgn_stream(month: str) -> io.TextIOWrapper:
    """Open a streaming, decompressed PGN text reader for the given month.

    The raw .zst file is never stored on disk; everything is streamed.
    """
    url = LICHESS_URL.format(month=month)
    print(f"[download] Connecting to {url}")

    session = requests.Session()
    resp = session.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    # Content-Length may be available for progress estimation
    total_bytes = resp.headers.get("Content-Length")
    if total_bytes:
        print(f"[download] Compressed size: {int(total_bytes) / 1e9:.2f} GB")
    else:
        print("[download] Compressed size: unknown (streaming)")

    binary_stream = ZstdPgnStream(resp)
    text_stream = io.TextIOWrapper(binary_stream, encoding="utf-8", errors="replace")
    return text_stream


# ---------------------------------------------------------------------------
# Game Filtering
# ---------------------------------------------------------------------------

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


def passes_filter(
    game: chess.pgn.Game,
    min_elo: int,
    max_elo: Optional[int] = None,
) -> bool:
    """Return True if a game passes all quality filters."""
    headers = game.headers

    # Result must be decisive or draw (not ongoing / abandoned)
    result = headers.get("Result", "*")
    if result == "*":
        return False

    # Time control category
    event = headers.get("Event", "")
    if not is_rapid_or_classical(event):
        return False

    # ELO filter
    white_elo = parse_elo(headers.get("WhiteElo"))
    black_elo = parse_elo(headers.get("BlackElo"))
    if white_elo < min_elo or black_elo < min_elo:
        return False
    if max_elo is not None and (white_elo > max_elo or black_elo > max_elo):
        return False

    # Minimum game length (at least 10 half-moves)
    moves = list(game.mainline_moves())
    if len(moves) < 10:
        return False

    return True


# ---------------------------------------------------------------------------
# Game Processing
# ---------------------------------------------------------------------------

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract all (board, policy, value) triples from a single game.

    Returns:
        boards:   float16 array of shape (M, 18, 8, 8)
        policies: int16 array of shape (M,)
        values:   int8 array of shape (M,)
    where M is the number of positions (= number of moves in the game).
    """
    result = game.headers.get("Result", "*")
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


# ---------------------------------------------------------------------------
# HDF5 Writer
# ---------------------------------------------------------------------------

class HDF5Writer:
    """Incrementally writes position data to a chunked, gzip-compressed HDF5 file."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.f = h5py.File(path, "w")
        self.f.attrs["complete"] = False

        # Create resizable datasets with chunked + gzip compression
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

        # In-memory buffers for batching writes
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

        # Resize all datasets
        self.ds_boards.resize(new_len, axis=0)
        self.ds_policies.resize(new_len, axis=0)
        self.ds_values.resize(new_len, axis=0)

        # Write the batch
        self.ds_boards[old_len:new_len] = all_boards
        self.ds_policies[old_len:new_len] = all_policies
        self.ds_values[old_len:new_len] = all_values

        self.total_positions = new_len

        # Clear buffers
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


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def iterate_games(text_stream: io.TextIOWrapper) -> Iterator[chess.pgn.Game]:
    """Yield chess.pgn.Game objects from a PGN text stream."""
    while True:
        game = chess.pgn.read_game(text_stream)
        if game is None:
            break
        yield game


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and preprocess Lichess PGN data for AlphaZero training."
    )
    parser.add_argument(
        "--month", required=True,
        help="Month to download, e.g. '2024-01'",
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
    args = parser.parse_args()
    if args.max_elo is not None and args.max_elo < args.min_elo:
        parser.error("--max-elo must be greater than or equal to --min-elo")
    if args.max_games is not None and args.max_games < 1:
        parser.error("--max-games must be positive")

    # Default output filename
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

    # Open the streaming PGN reader
    text_stream = open_pgn_stream(args.month)

    # Open HDF5 writer
    writer = HDF5Writer(str(output_path))
    writer.f.attrs["month"] = args.month
    writer.f.attrs["min_elo"] = args.min_elo
    writer.f.attrs["max_elo"] = -1 if args.max_elo is None else args.max_elo
    writer.f.attrs["max_games"] = -1 if args.max_games is None else args.max_games

    total_games = 0       # all games seen
    kept_games = 0        # games passing filters
    total_positions = 0   # running position count
    start_time = time.time()
    completed = False

    try:
        for game in iterate_games(text_stream):
            total_games += 1

            # Apply quality filters
            if not passes_filter(game, args.min_elo, args.max_elo):
                # Progress report every 10k games (including skipped)
                if total_games % 10_000 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"[progress] {total_games:>10,} games scanned | "
                        f"{kept_games:>8,} kept | "
                        f"{total_positions:>10,} positions | "
                        f"{elapsed:.0f}s elapsed"
                    )
                continue

            # Process the game
            boards, policies, values = process_game(game)
            writer.add(boards, policies, values)
            kept_games += 1
            total_positions += len(boards)

            # Progress report every 10k total games scanned
            if total_games % 10_000 == 0:
                elapsed = time.time() - start_time
                print(
                    f"[progress] {total_games:>10,} games scanned | "
                    f"{kept_games:>8,} kept | "
                    f"{total_positions:>10,} positions | "
                    f"{elapsed:.0f}s elapsed"
                )

            # Early stop for testing
            if args.max_games and kept_games >= args.max_games:
                print(f"[info] Reached --max-games cap ({args.max_games}). Stopping.")
                break
        completed = True

    except KeyboardInterrupt:
        print("\n[interrupted] Saving partial data and exiting...")

    finally:
        # Ensure all buffered data is written
        writer.close(complete=completed)
        text_stream.close()

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
