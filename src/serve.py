"""Serve the Whisker web app and local ChessNet API."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .paths import PROJECT_ROOT

WEB_ROOT = Path(__file__).resolve().parent / "web"
STATIC_FILES = {
    "/": (WEB_ROOT / "index.html", "text/html; charset=utf-8"),
    "/index.html": (WEB_ROOT / "index.html", "text/html; charset=utf-8"),
    "/styles.css": (WEB_ROOT / "styles.css", "text/css; charset=utf-8"),
    "/app.js": (WEB_ROOT / "app.js", "text/javascript; charset=utf-8"),
}
for color in ("white", "black"):
    for piece in ("pawn", "rook", "knight", "bishop", "queen", "king"):
        filename = f"{color}_{piece}.svg"
        STATIC_FILES[f"/assets/pieces/cburnett/{filename}"] = (
            WEB_ROOT / "assets" / "pieces" / "cburnett" / filename,
            "image/svg+xml; charset=utf-8",
        )
    STATIC_FILES[f"/assets/pieces/wintrchess/standard/{color}_king.svg"] = (
        WEB_ROOT / "assets" / "pieces" / "wintrchess" / "standard" / f"{color}_king.svg",
        "image/svg+xml; charset=utf-8",
    )
for sound in ("move", "capture", "castle", "check", "promote", "gameend"):
    STATIC_FILES[f"/assets/audio/{sound}.mp3"] = (
        WEB_ROOT / "assets" / "audio" / f"{sound}.mp3",
        "audio/mpeg",
    )
TIME_CONTROLS = {
    "bullet": {(1, 0), (2, 1)},
    "blitz": {(3, 0), (5, 0), (5, 3)},
    "rapid": {(10, 0), (15, 10)},
}


def find_checkpoint() -> Path | None:
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    if not checkpoint_dir.is_dir():
        return None
    candidates = [*checkpoint_dir.rglob("*.pt"), *checkpoint_dir.rglob("*.pth")]
    if not candidates:
        return None

    def rank(path: Path) -> tuple[int, str]:
        match = re.search(r"(?:step|iter)[_-]?(\d+)", path.stem, re.IGNORECASE)
        return (int(match.group(1)) if match else -1, str(path))

    return max(candidates, key=rank)


class ChessApp:
    def __init__(self, checkpoint: Path | None, device_name: str, threads: int) -> None:
        self.lock = threading.RLock()
        self.session: dict[str, Any] | None = None
        self.checkpoint = checkpoint
        self.device_name = device_name
        self.threads = threads
        self.model: Any = None
        self.device: Any = None
        self.select_move: Any = None
        self.model_error: str | None = None
        self.chess: Any = None
        self._load_engine()

    @property
    def model_available(self) -> bool:
        return self.model is not None and self.select_move is not None

    def _load_engine(self) -> None:
        if self.checkpoint is None:
            self.model_error = (
                "No checkpoint found. Pass one with "
                "python -m src.serve --checkpoint path/to/checkpoint.pt"
            )
            return
        if not self.checkpoint.is_file():
            self.model_error = f"Checkpoint not found: {self.checkpoint}"
            return
        try:
            import chess
            import torch

            from .engine import load_model, select_move

            if self.device_name == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(self.device_name)
                if self.device.type == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("CUDA was requested, but it is unavailable")
            torch.set_num_threads(self.threads)
            self.model = load_model(self.checkpoint, self.device)
            self.select_move = select_move
            self.chess = chess
        except Exception as exc:  # Keep the page available to explain setup issues.
            self.model = None
            self.select_move = None
            self.model_error = f"Could not load the checkpoint: {exc}"

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "model": {
                "available": self.model_available,
                "checkpoint": str(self.checkpoint) if self.checkpoint else None,
                "device": str(self.device) if self.device is not None else None,
                "detail": self.model_error or "ChessNet is ready.",
            },
        }

    def state(self) -> dict[str, Any]:
        with self.lock:
            if self.session is None:
                board = self._preview_board()
                return {"state": board}
            self._tick_clock()
            self._update_outcome()
            return {"state": self._serialize_session()}

    def parse_pgn(self, payload: dict[str, Any]) -> dict[str, Any]:
        pgn_text = payload.get("pgn")
        if not isinstance(pgn_text, str) or not pgn_text.strip():
            raise ApiError(400, "Paste a PGN game to review.")
        if len(pgn_text.encode("utf-8")) > 1_000_000:
            raise ApiError(413, "The PGN is too large to review.")

        try:
            import chess
            import chess.pgn
        except ImportError as exc:
            raise ApiError(503, "PGN review requires the python-chess package.") from exc
        self.chess = chess

        try:
            game = chess.pgn.read_game(StringIO(pgn_text))
        except Exception as exc:
            raise ApiError(400, f"Could not read the PGN: {exc}") from None
        if game is None:
            raise ApiError(400, "No chess game was found in that PGN.")
        if game.errors:
            raise ApiError(400, f"The PGN contains an invalid move: {game.errors[0]}")

        board = game.board()
        history = []
        frames = [self._study_position(board)]
        for move in game.mainline_moves():
            if move not in board.legal_moves:
                raise ApiError(400, "The PGN contains a move that is illegal in its position.")
            color = "white" if board.turn == chess.WHITE else "black"
            number = board.fullmove_number
            san = board.san(move)
            board.push(move)
            history.append({"number": number, "color": color, "san": san})
            frames.append(self._study_position(board, move.uci()))
            if len(history) > 3000:
                raise ApiError(413, "The PGN has too many moves to review in one session.")
        if not history:
            raise ApiError(400, "That PGN does not contain any moves.")

        tag_names = ("Event", "Site", "Date", "Round", "White", "Black", "Result", "ECO")
        headers = {name: game.headers.get(name, "") for name in tag_names}
        frames[-1]["game_over"] = headers["Result"] not in ("", "*", "?")
        return {"study": {"headers": headers, "history": history, "frames": frames}}

    def _study_position(self, board: Any, last_move: str | None = None) -> dict[str, Any]:
        checked_king = board.king(board.turn) if board.is_check() else None
        return {
            "pieces": self._piece_map(board),
            "to_move": "white" if board.turn == self.chess.WHITE else "black",
            "last_move": last_move,
            "in_check": self.chess.square_name(checked_king) if checked_king is not None else None,
        }

    def _preview_board(self) -> dict[str, Any]:
        try:
            import chess

            self.chess = chess
            board = chess.Board()
            return {
                "started": False,
                "mode": None,
                "time_control": None,
                "player_color": "white",
                "to_move": "white",
                "pieces": self._piece_map(board),
                "legal_moves": [],
                "history": [],
                "clocks_ms": {"white": 300_000, "black": 300_000},
                "last_move": None,
                "in_check": None,
                "result": None,
            }
        except ImportError:
            return {"started": False, "pieces": None, "history": [], "result": None}

    def new_game(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.model_available:
            raise ApiError(503, self.model_error or "ChessNet is not ready.")
        mode = str(payload.get("mode", "blitz"))
        try:
            minutes = int(payload.get("minutes", 5))
            increment = int(payload.get("increment", 0))
        except (TypeError, ValueError):
            raise ApiError(400, "Choose a supported time control.") from None
        if mode not in TIME_CONTROLS or (minutes, increment) not in TIME_CONTROLS[mode]:
            raise ApiError(400, "Choose one of the available time controls.")
        player_color = str(payload.get("player_color", "white"))
        if player_color not in {"white", "black"}:
            raise ApiError(400, "Choose white or black.")

        with self.lock:
            self.session = {
                "board": self.chess.Board(),
                "mode": mode,
                "minutes": minutes,
                "increment": increment,
                "player_color": player_color,
                "clocks_ms": {"white": minutes * 60_000, "black": minutes * 60_000},
                "history": [],
                "last_move": None,
                "result": None,
                "last_tick": time.monotonic(),
            }
            if player_color == "black":
                self._play_engine_turn()
            self._update_outcome()
            return {"state": self._serialize_session()}

    def play_player_move(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            if self.session is None:
                raise ApiError(409, "Start a game first.")
            self._tick_clock()
            self._update_outcome()
            session = self.session
            if session["result"]:
                raise ApiError(409, "This game has ended.")
            board = session["board"]
            expected_turn = self.chess.WHITE if session["player_color"] == "white" else self.chess.BLACK
            if board.turn != expected_turn:
                raise ApiError(409, "Wait for ChessNet to finish its move.")
            source = str(payload.get("from", ""))
            target = str(payload.get("to", ""))
            promotion = str(payload.get("promotion") or "q").lower()
            if promotion not in {"q", "r", "b", "n"}:
                raise ApiError(400, "Choose a valid promotion piece.")
            try:
                uci = source + target
                source_square = self.chess.parse_square(source)
                moving_piece = board.piece_at(source_square)
                if moving_piece and moving_piece.piece_type == self.chess.PAWN and self._is_promotion_rank(target):
                    uci += promotion
                move = self.chess.Move.from_uci(uci)
            except (ValueError, TypeError):
                raise ApiError(400, "That move is not valid.") from None
            if move not in board.legal_moves:
                raise ApiError(400, "That move is not legal in this position.")

            self._push_move(move)
            self._add_increment("white" if expected_turn else "black")
            self._update_outcome()
            if not session["result"]:
                self._play_engine_turn()
                self._update_outcome()
            return {"state": self._serialize_session()}

    def resign(self) -> dict[str, Any]:
        with self.lock:
            if self.session is None or self.session["result"]:
                raise ApiError(409, "There is no active game to resign.")
            self._tick_clock()
            if self.session["result"]:
                return {"state": self._serialize_session()}
            winner = "black" if self.session["player_color"] == "white" else "white"
            self.session["result"] = {
                "winner": winner,
                "message": f"{winner.capitalize()} wins by resignation.",
                "termination": "resignation",
            }
            return {"state": self._serialize_session()}

    def _play_engine_turn(self) -> None:
        session = self.session
        board = session["board"]
        engine_color = "white" if board.turn == self.chess.WHITE else "black"
        if engine_color == session["player_color"]:
            return
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            self._update_outcome()
            return
        try:
            move = self.select_move(board, self.model, self.device, candidates=legal_moves)
        except Exception as exc:
            print(f"ChessNet inference error: {exc}")
            session["result"] = {
                "winner": session["player_color"],
                "message": "ChessNet hit an inference error. Start a fresh game to try again.",
                "termination": "engine_error",
            }
            return
        self._settle_clock_at(time.monotonic())
        if session["result"]:
            return
        if move is None or move not in board.legal_moves:
            session["result"] = {
                "winner": session["player_color"],
                "message": "ChessNet could not choose a legal move.",
                "termination": "engine_error",
            }
            return
        self._push_move(move)
        self._add_increment(engine_color)

    def _push_move(self, move: Any) -> None:
        session = self.session
        board = session["board"]
        color = "white" if board.turn == self.chess.WHITE else "black"
        number = board.fullmove_number
        san = board.san(move)
        board.push(move)
        session["history"].append({"number": number, "color": color, "san": san})
        session["last_move"] = move.uci()
        session["last_tick"] = time.monotonic()

    def _add_increment(self, color: str) -> None:
        self.session["clocks_ms"][color] += self.session["increment"] * 1000

    def _tick_clock(self) -> None:
        if self.session is not None and self.session["result"] is None:
            self._settle_clock_at(time.monotonic())

    def _settle_clock_at(self, now: float) -> None:
        session = self.session
        if session is None or session["result"] is not None:
            return
        color = "white" if session["board"].turn == self.chess.WHITE else "black"
        elapsed_ms = max(0, int((now - session["last_tick"]) * 1000))
        session["clocks_ms"][color] = max(0, session["clocks_ms"][color] - elapsed_ms)
        session["last_tick"] = now
        if session["clocks_ms"][color] <= 0:
            winner = "black" if color == "white" else "white"
            session["result"] = {
                "winner": winner,
                "message": f"{winner.capitalize()} wins on time.",
                "termination": "time",
            }

    def _update_outcome(self) -> None:
        session = self.session
        if session is None or session["result"] is not None:
            return
        outcome = session["board"].outcome()
        if outcome is None:
            return
        winner = None
        if outcome.winner is not None:
            winner = "white" if outcome.winner == self.chess.WHITE else "black"
        termination = outcome.termination.name.lower().replace("_", " ")
        if winner is None:
            message = f"Draw by {termination}."
        elif outcome.termination == self.chess.Termination.CHECKMATE:
            message = f"{winner.capitalize()} wins by checkmate."
        else:
            message = f"{winner.capitalize()} wins by {termination}."
        session["result"] = {"winner": winner, "message": message, "termination": termination}

    def _serialize_session(self) -> dict[str, Any]:
        session = self.session
        board = session["board"]
        legal_moves = []
        expected_turn = self.chess.WHITE if session["player_color"] == "white" else self.chess.BLACK
        if session["result"] is None and board.turn == expected_turn:
            legal_moves = [move.uci() for move in board.legal_moves]
        time_control = f"{session['minutes']}|{session['increment']}"
        return {
            "started": True,
            "mode": session["mode"],
            "time_control": time_control,
            "player_color": session["player_color"],
            "to_move": "white" if board.turn == self.chess.WHITE else "black",
            "pieces": self._piece_map(board),
            "legal_moves": legal_moves,
            "history": session["history"],
            "clocks_ms": dict(session["clocks_ms"]),
            "last_move": session["last_move"],
            "in_check": self.chess.square_name(board.king(board.turn)) if board.is_check() else None,
            "result": session["result"],
        }

    def _piece_map(self, board: Any) -> dict[str, dict[str, str]]:
        return {
            self.chess.square_name(square): {
                "color": "white" if piece.color == self.chess.WHITE else "black",
                "type": piece.symbol().lower(),
            }
            for square, piece in board.piece_map().items()
        }

    @staticmethod
    def _is_promotion_rank(square: str) -> bool:
        return len(square) == 2 and square[1] in {"1", "8"}


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def make_handler(app: ChessApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "Whisker/1.0"

        def do_GET(self) -> None:
            route = urlparse(self.path).path
            try:
                if route == "/api/health":
                    self._json(200, app.health())
                    return
                if route == "/api/state":
                    self._json(200, app.state())
                    return
                if route in STATIC_FILES:
                    path, content_type = STATIC_FILES[route]
                    self._file(path, content_type)
                    return
                self._json(404, {"error": "Not found"})
            except ApiError as exc:
                self._json(exc.status, {"error": str(exc)})
            except Exception as exc:
                self._json(500, {"error": str(exc)})

        def do_POST(self) -> None:
            route = urlparse(self.path).path
            try:
                payload = self._read_json()
                if route == "/api/new":
                    result = app.new_game(payload)
                elif route == "/api/move":
                    result = app.play_player_move(payload)
                elif route == "/api/pgn":
                    result = app.parse_pgn(payload)
                elif route == "/api/resign":
                    result = app.resign()
                else:
                    self._json(404, {"error": "Not found"})
                    return
                self._json(200, result)
            except ApiError as exc:
                self._json(exc.status, {"error": str(exc)})
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._json(400, {"error": "The request body must be valid JSON."})
            except Exception as exc:
                self._json(500, {"error": str(exc)})

        def _read_json(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise ApiError(400, "Invalid request length.") from None
            if length < 0 or length > 1_048_576:
                raise ApiError(413, "Request body exceeds 1 MB.")
            raw = self.rfile.read(length)
            value = json.loads(raw or b"{}")
            if not isinstance(value, dict):
                raise ApiError(400, "The request body must be an object.")
            return value

        def _json(self, status: int, value: dict[str, Any]) -> None:
            body = json.dumps(value, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _file(self, path: Path, content_type: str) -> None:
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type or mimetypes.guess_type(path.name)[0])
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format_string: str, *args: Any) -> None:
            print(f"[{self.log_date_time_string()}] {format_string % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Whisker local web app.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="ChessNet checkpoint (.pt or .pth)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    if args.port < 1 or args.port > 65535:
        parser.error("--port must be between 1 and 65535")

    checkpoint = args.checkpoint
    if checkpoint is not None and not checkpoint.is_absolute():
        checkpoint = (PROJECT_ROOT / checkpoint).resolve()
    elif checkpoint is None:
        checkpoint = find_checkpoint()

    app = ChessApp(checkpoint, args.device, args.threads)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"Whisker is serving at http://{args.host}:{args.port}")
    if app.model_available:
        print(f"ChessNet: {app.checkpoint} on {app.device}")
    else:
        print(f"ChessNet unavailable: {app.model_error}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Whisker.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
