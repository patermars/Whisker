"""Small UCI adapter for the ChessNet policy network.

This engine selects the highest-scoring legal move from the policy head. It
does not search ahead, so its playing strength is limited by the trained
network and its inference speed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch

from .prepare_data import encode_board
from .paths import PROJECT_ROOT
from .train import ChessNet


def load_model(checkpoint_path: Path, device: torch.device) -> ChessNet:
    """Load either this project's training checkpoint or a plain state dict."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    # These checkpoints are local files created by train.py. Its checkpoint
    # includes optimizer/scaler metadata in addition to the model weights.
    payload: Any = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a state dict or a checkpoint dictionary")

    state = payload
    for key in ("model", "model_state_dict", "state_dict"):
        if key in payload and isinstance(payload[key], dict):
            state = payload[key]
            break

    # Also accept DataParallel and torch.compile state dicts.
    normalized_state = {}
    for key, value in state.items():
        while key.startswith("module.") or key.startswith("_orig_mod."):
            key = key.split(".", 1)[1]
        normalized_state[key] = value

    try:
        channels = int(normalized_state["stem_conv.weight"].shape[0])
        policy_size = int(normalized_state["policy_fc.weight"].shape[0])
        policy_channels = int(normalized_state["policy_conv.weight"].shape[0])
        value_channels = int(normalized_state["value_conv.weight"].shape[0])
        block_ids = {
            int(key.split(".")[1])
            for key in normalized_state
            if key.startswith("tower.") and key.endswith(".conv1.weight")
        }
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError("checkpoint does not match the ChessNet architecture") from exc

    if not block_ids:
        raise ValueError("checkpoint has no ChessNet residual blocks")

    model = ChessNet(
        channels=channels,
        num_blocks=max(block_ids) + 1,
        policy_size=policy_size,
        policy_channels=policy_channels,
        value_channels=value_channels,
        use_se=any(key.startswith("tower.0.se_fc1.") for key in normalized_state),
    )
    model.load_state_dict(normalized_state)
    model.to(device)
    model.eval()
    return model


def select_move(
    board: chess.Board,
    model: ChessNet,
    device: torch.device,
    candidates: list[chess.Move] | None = None,
) -> chess.Move | None:
    """Return the candidate move with the largest policy logit."""
    legal_moves = candidates if candidates is not None else list(board.legal_moves)
    if not legal_moves:
        return None

    board_array = np.asarray(encode_board(board), dtype=np.float32)
    board_tensor = torch.from_numpy(board_array).unsqueeze(0).to(device)

    with torch.inference_mode():
        policy_logits, _ = model(board_tensor)
        logits = policy_logits[0]

    def score(move: chess.Move) -> tuple[float, int]:
        # Dataset labels encode from_square * 64 + to_square. Promotions share
        # that label, so prefer queen promotion if their logits are identical.
        index = move.from_square * 64 + move.to_square
        promotion_tiebreak = int(move.promotion == chess.QUEEN)
        return float(logits[index].item()), promotion_tiebreak

    return max(legal_moves, key=score)


def parse_position(command: str) -> chess.Board:
    """Parse a UCI position command, including an optional move list."""
    tokens = command.split()
    if len(tokens) < 2 or tokens[0] != "position":
        raise ValueError("expected a UCI position command")

    if tokens[1] == "startpos":
        board = chess.Board()
        moves_at = 2
    elif tokens[1] == "fen":
        try:
            moves_at = tokens.index("moves", 2)
        except ValueError:
            moves_at = len(tokens)
        fen = " ".join(tokens[2:moves_at])
        board = chess.Board(fen)
    else:
        raise ValueError("position must use startpos or fen")

    if moves_at < len(tokens):
        if tokens[moves_at] != "moves":
            raise ValueError("unexpected data after position")
        for uci_move in tokens[moves_at + 1 :]:
            board.push_uci(uci_move)

    return board


def allowed_search_moves(command: str, board: chess.Board) -> set[chess.Move] | None:
    """Read an optional UCI `searchmoves` restriction."""
    tokens = command.split()
    try:
        start = tokens.index("searchmoves") + 1
    except ValueError:
        return None

    legal = set(board.legal_moves)
    selected: set[chess.Move] = set()
    for token in tokens[start:]:
        try:
            move = board.parse_uci(token)
        except ValueError:
            continue
        if move in legal:
            selected.add(move)
    return selected


def emit(line: str) -> None:
    print(line, flush=True)


def run_uci(model: ChessNet, device: torch.device) -> None:
    board = chess.Board()

    for raw_line in sys.stdin:
        command = raw_line.strip()
        if not command:
            continue

        if command == "uci":
            emit("id name ChessNet Policy")
            emit("id author local")
            emit("uciok")
        elif command == "isready":
            emit("readyok")
        elif command.startswith("position "):
            try:
                board = parse_position(command)
            except (ValueError, IndexError) as exc:
                emit(f"info string invalid position: {exc}")
        elif command == "go" or command.startswith("go "):
            moves = list(board.legal_moves)
            restricted = allowed_search_moves(command, board)
            if restricted is not None:
                moves = [move for move in moves if move in restricted]

            if not moves:
                emit("bestmove 0000")
                continue

            move = select_move(board, model, device, candidates=moves)

            emit(f"bestmove {move.uci()}")
        elif command in {"ucinewgame", "stop", "ponderhit"}:
            continue
        elif command == "quit":
            break
        # UCI option commands are ignored because this model has no runtime
        # search or evaluation options.


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ChessNet as a UCI chess engine.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints/checkpoint_step_1000.pt",
        help="training checkpoint (.pt) or model state dict (.pth)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="inference device (default: CUDA when available, otherwise CPU)",
    )
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads")
    args = parser.parse_args()

    if args.threads < 1:
        parser.error("--threads must be at least 1")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested, but CUDA is unavailable")

    torch.set_num_threads(args.threads)
    try:
        model = load_model(args.checkpoint, device)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ChessNet: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc

    run_uci(model, device)


if __name__ == "__main__":
    main()
