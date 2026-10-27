"""Run paired local gauntlets against Stockfish's UCI_Elo strength limits."""

from __future__ import annotations

import argparse
import math
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

import chess
import chess.pgn

from .paths import PROJECT_ROOT

def find_executable(command: str, label: str) -> str:
    """Resolve a command name or executable path, with a useful error."""
    resolved = shutil.which(command)
    if resolved is None:
        raise SystemExit(
            f"{label} executable {command!r} was not found. "
            f"Install it or pass its full path with the matching option."
        )
    return resolved


def get_uci_elo_range(stockfish: str) -> tuple[int, int]:
    """Read this Stockfish binary's advertised UCI_Elo bounds."""
    try:
        result = subprocess.run(
            [stockfish],
            input="uci\nquit\n",
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"Could not query Stockfish UCI options: {exc}") from exc

    output = result.stdout + "\n" + result.stderr
    match = re.search(
        r"(?im)^option name UCI_Elo type spin .*?\bmin\s+(\d+)\s+max\s+(\d+)\s*$",
        output,
    )
    if match is None:
        raise SystemExit(
            "This Stockfish binary did not advertise a UCI_Elo spin option. "
            "Use a Stockfish build that supports UCI_LimitStrength/UCI_Elo."
        )
    return int(match.group(1)), int(match.group(2))


def write_openings(path: Path, count: int, plies: int, seed: int) -> None:
    """Create a deterministic set of varied, legal opening lines."""
    rng = random.Random(seed)
    with path.open("w", encoding="utf-8") as out:
        for number in range(count):
            board = chess.Board()
            game = chess.pgn.Game()
            game.headers["Event"] = "Whisker local benchmark openings"
            game.headers["Site"] = "Local"
            game.headers["Round"] = str(number + 1)
            node = game
            for _ in range(plies):
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break
                move = rng.choice(legal_moves)
                board.push(move)
                node = node.add_variation(move)
            out.write(str(game))
            out.write("\n\n")


def score_pgn(path: Path, engine_name: str) -> tuple[int, float, float, float]:
    """Return completed games, points, raw score, and smoothed score."""
    points: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as pgn:
        while game := chess.pgn.read_game(pgn):
            result = game.headers.get("Result", "*")
            if result not in {"1-0", "0-1", "1/2-1/2"}:
                continue

            white = game.headers.get("White", "")
            black = game.headers.get("Black", "")
            if engine_name == white:
                points.append({"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}[result])
            elif engine_name == black:
                points.append({"0-1": 1.0, "1/2-1/2": 0.5, "1-0": 0.0}[result])

    count = len(points)
    if count == 0:
        raise RuntimeError(f"no completed {engine_name} games found in {path}")

    raw_score = sum(points) / count
    # Half-point smoothing avoids infinite ratings after a clean sweep.
    adjusted_score = (sum(points) + 0.5) / (count + 1)
    return count, sum(points), raw_score, adjusted_score


def performance_rating(opponent_elo: int, score: float) -> float:
    return opponent_elo + 400.0 * math.log10(score / (1.0 - score))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play Whisker against Stockfish at several calibrated UCI_Elo settings."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/checkpoint_step_1000.pt",
    )
    parser.add_argument("--stockfish", default="stockfish", help="Stockfish executable")
    parser.add_argument("--cutechess", default="cutechess-cli", help="Cute Chess CLI executable")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for the engine")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads")
    parser.add_argument("--elos", type=int, nargs="+", default=[1500, 1800, 2100, 2400])
    parser.add_argument("--games", type=int, default=100, help="games per Stockfish target; use an even number")
    parser.add_argument("--tc", default="60+0.6", help="Cute Chess time control, e.g. 60+0.6")
    parser.add_argument("--plies", type=int, default=8, help="random opening length")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "benchmark-results"
    )
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()

    if args.games < 2 or args.games % 2:
        parser.error("--games must be an even number of at least 2")
    if not args.elos or any(elo <= 0 for elo in args.elos):
        parser.error("--elos must contain positive ratings")
    if args.threads < 1 or args.plies < 1:
        parser.error("--threads and --plies must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    cutechess = find_executable(args.cutechess, "Cute Chess CLI")
    stockfish = find_executable(args.stockfish, "Stockfish")
    elo_min, elo_max = get_uci_elo_range(stockfish)
    out_of_range = [elo for elo in args.elos if not elo_min <= elo <= elo_max]
    if out_of_range:
        invalid = ", ".join(map(str, out_of_range))
        raise SystemExit(
            f"Stockfish accepts UCI_Elo values from {elo_min} to {elo_max}; "
            f"out of range: {invalid}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    opening_file = args.output_dir / "openings.pgn"
    write_openings(opening_file, args.games // 2, args.plies, args.seed)

    engine_name = "Whisker"
    engine_args = [
        f"cmd={args.python}",
        f"dir={PROJECT_ROOT}",
        "arg=-m",
        "arg=src.engine",
        "arg=--checkpoint",
        f"arg={args.checkpoint.resolve()}",
        "arg=--device",
        f"arg={args.device}",
        "arg=--threads",
        f"arg={args.threads}",
        f"name={engine_name}",
        "proto=uci",
    ]

    for elo in args.elos:
        pgn_path = args.output_dir / f"stockfish-{elo}.pgn"
        engine_log_path = args.output_dir / f"whisker-{elo}.stderr.log"
        command = [
            cutechess,
            "-engine",
            *engine_args,
            f"stderr={engine_log_path}",
            "-engine",
            f"cmd={stockfish}",
            f"name=Stockfish-{elo}",
            "proto=uci",
            "option.UCI_LimitStrength=true",
            f"option.UCI_Elo={elo}",
            "-each",
            f"tc={args.tc}",
            "-rounds",
            str(args.games // 2),
            "-games",
            "2",
            "-concurrency",
            "1",
            "-openings",
            f"file={opening_file}",
            "format=pgn",
            "order=sequential",
            f"plies={args.plies}",
            "-repeat",
            "-pgnout",
            str(pgn_path),
        ]

        pgn_path.unlink(missing_ok=True)
        engine_log_path.unlink(missing_ok=True)
        print(f"\n[{elo}] Running {args.games} games at tc={args.tc} ...", flush=True)
        try:
            subprocess.run(command, check=True, cwd=PROJECT_ROOT)
        except subprocess.CalledProcessError as exc:
            engine_error = (
                engine_log_path.read_text(encoding="utf-8", errors="replace").strip()
                if engine_log_path.is_file()
                else ""
            )
            raise SystemExit(
                f"Cute Chess exited with status {exc.returncode}; see its output above."
                + (f"\nWhisker stderr ({engine_log_path}):\n{engine_error}" if engine_error else "")
            ) from exc

        if not pgn_path.is_file():
            engine_error = (
                engine_log_path.read_text(encoding="utf-8", errors="replace").strip()
                if engine_log_path.is_file()
                else ""
            )
            raise SystemExit(
                f"Cute Chess did not create {pgn_path}. Check that both engines started "
                "and that the match completed."
                + (f"\nWhisker stderr ({engine_log_path}):\n{engine_error}" if engine_error else "")
            )

        try:
            completed, points, raw_score, adjusted_score = score_pgn(pgn_path, engine_name)
        except RuntimeError as exc:
            raise SystemExit(f"Could not score the match: {exc}") from exc
        if completed != args.games:
            raise SystemExit(
                f"Expected {args.games} completed games, found {completed} in {pgn_path}. "
                "Check the Cute Chess output before trusting this estimate."
            )
        estimate = performance_rating(elo, adjusted_score)
        print(
            f"[{elo}] {points:g}/{completed} points ({raw_score:.1%}); "
            f"rough performance estimate {estimate:.0f} Elo; PGN: {pgn_path}",
            flush=True,
        )

    print(
        "\nTreat these as local estimates. Stockfish's UCI_Elo targets are calibrated "
        "to engine rating lists and time controls; they are not Lichess human ratings.",
        flush=True,
    )


if __name__ == "__main__":
    main()
