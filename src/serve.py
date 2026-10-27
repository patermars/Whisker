"""Serve the Whisker web app and local Whisker engine API."""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import os
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from http.cookies import CookieError, SimpleCookie
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
SESSION_TTL_SECONDS = 6 * 60 * 60
SESSION_CLEANUP_INTERVAL_SECONDS = 10 * 60
SESSION_COOKIE_NAME = "whisker_session"
logger = logging.getLogger("whisker.serve")


@dataclass
class BrowserSession:
    game: dict[str, Any] | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)
    last_seen: float = field(default_factory=time.monotonic)


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
        self.sessions: dict[str, BrowserSession] = {}
        self.sessions_lock = threading.RLock()
        self.inference_lock = threading.Lock()
        self.checkpoint = checkpoint
        self.device_name = device_name
        self.threads = threads
        self.model: Any = None
        self.device: Any = None
        self.select_move: Any = None
        self.model_error: str | None = None
        self.chess: Any = None
        self.cookie_secure = os.getenv("WHISKER_COOKIE_SECURE", "0").lower() in {
            "1", "true", "yes", "on"
        }
        self._cleanup_stop = threading.Event()
        self._cleanup_thread = threading.Thread(
            target=self._session_cleanup_loop,
            name="whisker-session-cleanup",
            daemon=True,
        )
        self._cleanup_thread.start()
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
            logger.error("Whisker checkpoint is not configured")
            return
        if not self.checkpoint.is_file():
            self.model_error = "Checkpoint is missing."
            logger.error("Whisker checkpoint does not exist: %s", self.checkpoint)
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
            self.model_error = "Whisker could not load its model."
            logger.exception("Could not initialize Whisker model")

    def health(self) -> dict[str, Any]:
        available = self.model_available
        return {
            "ok": available,
            "model": {
                "available": available,
                "device": str(self.device) if self.device is not None else None,
            },
        }

    def get_browser_session(
        self, session_id: str | None = None
    ) -> tuple[str, BrowserSession, bool]:
        """Find or create one visitor's session and refresh its expiry time."""
        now = time.monotonic()
        with self.sessions_lock:
            self._expire_sessions_locked(now)
            if session_id is not None:
                session = self.sessions.get(session_id)
                if session is not None:
                    session.last_seen = now
                    return session_id, session, False
            new_id = secrets.token_urlsafe(32)
            session = BrowserSession(last_seen=now)
            self.sessions[new_id] = session
            return new_id, session, True

    def _expire_sessions_locked(self, now: float) -> None:
        expired_before = now - SESSION_TTL_SECONDS
        for session_id, session in list(self.sessions.items()):
            if session.last_seen >= expired_before:
                continue
            if not session.lock.acquire(blocking=False):
                continue
            try:
                if session.last_seen < expired_before:
                    self.sessions.pop(session_id, None)
            finally:
                session.lock.release()

    def _session_cleanup_loop(self) -> None:
        while not self._cleanup_stop.wait(SESSION_CLEANUP_INTERVAL_SECONDS):
            with self.sessions_lock:
                self._expire_sessions_locked(time.monotonic())

    def close(self) -> None:
        self._cleanup_stop.set()

    def _get_session(self, session_id: str) -> BrowserSession:
        with self.sessions_lock:
            session = self.sessions.get(session_id)
            if session is None:
                # A valid request can race session expiry; give it a fresh game slot.
                session = BrowserSession()
                self.sessions[session_id] = session
            session.last_seen = time.monotonic()
            return session

    def state(self, session_id: str) -> dict[str, Any]:
        session = self._get_session(session_id)
        with session.lock:
            if session.game is None:
                return {"state": self._preview_board()}
            self._tick_clock(session.game)
            self._update_outcome(session.game)
            return {"state": self._serialize_session(session.game)}

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
            logger.exception("PGN parser failed")
            raise ApiError(400, "Could not read that PGN.") from None
        if game is None:
            raise ApiError(400, "No chess game was found in that PGN.")
        if game.errors:
            raise ApiError(400, "The PGN contains an invalid move.")

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

    def new_game(
        self, session_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        if not self.model_available:
            raise ApiError(503, "Whisker is not ready. Try again later.")
        mode = str(payload.get("mode", "blitz"))
        try:
            minutes = int(payload.get("minutes", 5))
            increment = int(payload.get("increment", 0))
        except (TypeError, ValueError):
            raise ApiError(400, "Choose a supported time control.") from None
        if mode not in TIME_CONTROLS or (minutes, increment) not in TIME_CONTROLS[mode]:
            raise ApiError(400, "Choose one of the available time controls.")
        player_color = str(payload.get("player_color", "white"))
        if player_color == "random":
            player_color = secrets.choice(("white", "black"))
        if player_color not in {"white", "black"}:
            raise ApiError(400, "Choose white or black.")

        session = self._get_session(session_id)
        with session.lock:
            session.game = {
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
            return {"state": self._serialize_session(session.game)}

    def play_player_move(
        self, session_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        browser_session = self._get_session(session_id)
        with browser_session.lock:
            game = browser_session.game
            if game is None:
                raise ApiError(409, "Start a game first.")
            self._tick_clock(game)
            self._update_outcome(game)
            if game["result"]:
                raise ApiError(409, "This game has ended.")
            board = game["board"]
            expected_turn = self.chess.WHITE if game["player_color"] == "white" else self.chess.BLACK
            if board.turn != expected_turn:
                raise ApiError(409, "Wait for Whisker to finish its move.")
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

            self._push_move(game, move)
            self._add_increment(game, "white" if expected_turn else "black")
            self._update_outcome(game)
            return {"state": self._serialize_session(game)}

    def play_engine_move(self, session_id: str) -> dict[str, Any]:
        browser_session = self._get_session(session_id)
        with browser_session.lock:
            game = browser_session.game
            if game is None:
                raise ApiError(409, "Start a game first.")
            self._tick_clock(game)
            self._update_outcome(game)
            if game["result"]:
                return {"state": self._serialize_session(game)}
            expected_turn = self.chess.WHITE if game["player_color"] == "white" else self.chess.BLACK
            if game["board"].turn == expected_turn:
                raise ApiError(409, "It is your turn.")
            self._play_engine_turn(game)
            self._update_outcome(game)
            return {"state": self._serialize_session(game)}

    def resign(self, session_id: str) -> dict[str, Any]:
        browser_session = self._get_session(session_id)
        with browser_session.lock:
            game = browser_session.game
            if game is None or game["result"]:
                raise ApiError(409, "There is no active game to resign.")
            self._tick_clock(game)
            if game["result"]:
                return {"state": self._serialize_session(game)}
            winner = "black" if game["player_color"] == "white" else "white"
            game["result"] = {
                "winner": winner,
                "message": f"{winner.capitalize()} wins by resignation.",
                "termination": "resignation",
            }
            return {"state": self._serialize_session(game)}

    def _play_engine_turn(self, game: dict[str, Any]) -> None:
        board = game["board"]
        engine_color = "white" if board.turn == self.chess.WHITE else "black"
        if engine_color == game["player_color"]:
            return
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            self._update_outcome(game)
            return
        try:
            with self.inference_lock:
                move = self.select_move(board, self.model, self.device, candidates=legal_moves)
        except Exception as exc:
            logger.exception("Whisker inference failed")
            game["result"] = {
                "winner": game["player_color"],
                "message": "Whisker hit an inference error. Start a fresh game to try again.",
                "termination": "engine_error",
            }
            return
        self._settle_clock_at(game, time.monotonic())
        if game["result"]:
            return
        if move is None or move not in board.legal_moves:
            game["result"] = {
                "winner": game["player_color"],
                "message": "Whisker could not choose a legal move.",
                "termination": "engine_error",
            }
            return
        self._push_move(game, move)
        self._add_increment(game, engine_color)

    def _push_move(self, game: dict[str, Any], move: Any) -> None:
        board = game["board"]
        color = "white" if board.turn == self.chess.WHITE else "black"
        number = board.fullmove_number
        san = board.san(move)
        board.push(move)
        game["history"].append({"number": number, "color": color, "san": san})
        game["last_move"] = move.uci()
        game["last_tick"] = time.monotonic()

    def _add_increment(self, game: dict[str, Any], color: str) -> None:
        game["clocks_ms"][color] += game["increment"] * 1000

    def _tick_clock(self, game: dict[str, Any]) -> None:
        if game["result"] is None:
            self._settle_clock_at(game, time.monotonic())

    def _settle_clock_at(self, game: dict[str, Any], now: float) -> None:
        if game["result"] is not None:
            return
        color = "white" if game["board"].turn == self.chess.WHITE else "black"
        elapsed_ms = max(0, int((now - game["last_tick"]) * 1000))
        game["clocks_ms"][color] = max(0, game["clocks_ms"][color] - elapsed_ms)
        game["last_tick"] = now
        if game["clocks_ms"][color] <= 0:
            winner = "black" if color == "white" else "white"
            game["result"] = {
                "winner": winner,
                "message": f"{winner.capitalize()} wins on time.",
                "termination": "time",
            }

    def _update_outcome(self, game: dict[str, Any]) -> None:
        if game["result"] is not None:
            return
        outcome = game["board"].outcome()
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
        game["result"] = {"winner": winner, "message": message, "termination": termination}

    def _serialize_session(self, game: dict[str, Any]) -> dict[str, Any]:
        board = game["board"]
        legal_moves = []
        expected_turn = self.chess.WHITE if game["player_color"] == "white" else self.chess.BLACK
        if game["result"] is None and board.turn == expected_turn:
            legal_moves = [move.uci() for move in board.legal_moves]
        time_control = f"{game['minutes']}|{game['increment']}"
        return {
            "started": True,
            "mode": game["mode"],
            "time_control": time_control,
            "player_color": game["player_color"],
            "to_move": "white" if board.turn == self.chess.WHITE else "black",
            "pieces": self._piece_map(board),
            "legal_moves": legal_moves,
            "history": game["history"],
            "clocks_ms": dict(game["clocks_ms"]),
            "last_move": game["last_move"],
            "in_check": self.chess.square_name(board.king(board.turn)) if board.is_check() else None,
            "result": game["result"],
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

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._cookie_to_set: str | None = None
            super().__init__(*args, **kwargs)

        def _request_session_id(self) -> str:
            cookie_header = self.headers.get("Cookie", "")
            supplied_id = None
            if cookie_header:
                cookies = SimpleCookie()
                try:
                    cookies.load(cookie_header)
                    morsel = cookies.get(SESSION_COOKIE_NAME)
                    supplied_id = morsel.value if morsel is not None else None
                except CookieError:
                    supplied_id = None
            session_id, _session, created = app.get_browser_session(supplied_id)
            if created:
                self._cookie_to_set = session_id
            return session_id

        def _write_session_cookie(self) -> None:
            if self._cookie_to_set is None:
                return
            value = (
                f"{SESSION_COOKIE_NAME}={self._cookie_to_set}; HttpOnly; "
                "SameSite=Lax; Path=/; Max-Age=21600"
            )
            if app.cookie_secure:
                value += "; Secure"
            self.send_header("Set-Cookie", value)

        def do_GET(self) -> None:
            route = urlparse(self.path).path
            try:
                if route == "/api/health":
                    health = app.health()
                    self._json(200 if health["ok"] else 503, health)
                    return
                if route == "/api/state":
                    session_id = self._request_session_id()
                    self._json(200, app.state(session_id))
                    return
                if route in STATIC_FILES:
                    path, content_type = STATIC_FILES[route]
                    self._file(path, content_type)
                    return
                self._json(404, {"error": "Not found"})
            except ApiError as exc:
                self._json(exc.status, {"error": str(exc)})
            except Exception as exc:
                logger.exception("Unhandled GET API error for %s", route)
                self._json(500, {"error": "The server could not complete the request."})

        def do_POST(self) -> None:
            route = urlparse(self.path).path
            try:
                if route not in {
                    "/api/new", "/api/move", "/api/engine-move", "/api/pgn", "/api/resign"
                }:
                    self._json(404, {"error": "Not found"})
                    return
                payload = self._read_json()
                if route == "/api/new":
                    session_id = self._request_session_id()
                    result = app.new_game(session_id, payload)
                elif route == "/api/move":
                    session_id = self._request_session_id()
                    result = app.play_player_move(session_id, payload)
                elif route == "/api/engine-move":
                    session_id = self._request_session_id()
                    result = app.play_engine_move(session_id)
                elif route == "/api/pgn":
                    result = app.parse_pgn(payload)
                elif route == "/api/resign":
                    session_id = self._request_session_id()
                    result = app.resign(session_id)
                self._json(200, result)
            except ApiError as exc:
                self._json(exc.status, {"error": str(exc)})
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._json(400, {"error": "The request body must be valid JSON."})
            except Exception as exc:
                logger.exception("Unhandled POST API error for %s", route)
                self._json(500, {"error": "The server could not complete the request."})

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
            self._write_session_cookie()
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
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run the Whisker local web app.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Whisker checkpoint (.pt or .pth)")
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"),
        default=os.getenv("WHISKER_DEVICE", "auto"),
    )
    parser.add_argument("--threads", type=int, default=int(os.getenv("WHISKER_THREADS", "1")))
    parser.add_argument("--host", default=os.getenv("WHISKER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    if args.port < 1 or args.port > 65535:
        parser.error("--port must be between 1 and 65535")

    checkpoint = args.checkpoint
    if checkpoint is None:
        configured_checkpoint = os.getenv("WHISKER_CHECKPOINT_PATH")
        checkpoint = Path(configured_checkpoint) if configured_checkpoint else None
    if checkpoint is not None and not checkpoint.is_absolute():
        checkpoint = (PROJECT_ROOT / checkpoint).resolve()
    elif checkpoint is None:
        checkpoint = find_checkpoint()

    app = ChessApp(checkpoint, args.device, args.threads)
    if os.getenv("WHISKER_REQUIRE_MODEL", "0").lower() in {"1", "true", "yes", "on"}:
        if not app.model_available:
            app.close()
            raise SystemExit(
                "Whisker model initialization failed; refusing to start because "
                "WHISKER_REQUIRE_MODEL is enabled. See server logs for details."
            )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"Whisker is serving at http://{args.host}:{args.port}")
    if app.model_available:
        print(f"Whisker: {app.checkpoint} on {app.device}")
    else:
        print(f"Whisker unavailable: {app.model_error}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Whisker.")
    finally:
        server.server_close()
        app.close()


if __name__ == "__main__":
    main()
