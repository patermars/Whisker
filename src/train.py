"""Train the Whisker policy and value network."""

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Training uses PyTorch's built-in CPU/CUDA backends. Skip optional third-party
# backend discovery by default so an unreadable entry_points.txt in any installed
# package cannot prevent torch from importing. Set this to 1 before launching
# training when an out-of-tree device backend plugin is required.
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import h5py
import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    Subset,
    SubsetRandomSampler,
    WeightedRandomSampler,
)
from tqdm import tqdm

def setup_logging(output_dir: str) -> logging.Logger:
    """Configure a logger that writes to both console and a local file."""
    logger = logging.getLogger("chess_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    os.makedirs(output_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(output_dir, "training.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


class ExperimentTracker:
    """Thin wrapper that dispatches metric logging to wandb, TensorBoard, or nowhere."""

    def __init__(self, config: Dict[str, Any], output_dir: str) -> None:
        self.backend = "console"
        self._writer = None

        try:
            import wandb  # type: ignore
            wandb.init(project="chess-alphazero", config=config, dir=output_dir)
            self.backend = "wandb"
            self._wandb = wandb
            return
        except Exception:
            pass

        try:
            from torch.utils.tensorboard import SummaryWriter  # type: ignore
            self._writer = SummaryWriter(log_dir=os.path.join(output_dir, "tb_logs"))
            self.backend = "tensorboard"
            return
        except Exception:
            pass

    def log(self, metrics: Dict[str, float], step: int) -> None:
        """Log a dict of metrics at a given step."""
        if self.backend == "wandb":
            self._wandb.log(metrics, step=step)
        elif self.backend == "tensorboard" and self._writer is not None:
            for k, v in metrics.items():
                self._writer.add_scalar(k, v, step)

    def close(self) -> None:
        if self.backend == "wandb":
            self._wandb.finish()
        elif self.backend == "tensorboard" and self._writer is not None:
            self._writer.close()


class ResidualBlock(nn.Module):
    """Residual block with optional squeeze-and-excitation channel attention."""

    def __init__(self, channels: int, use_se: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.use_se = use_se
        if use_se:
            hidden = max(channels // 4, 16)
            self.se_fc1 = nn.Linear(channels, hidden)
            self.se_fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_se:
            gate = F.adaptive_avg_pool2d(out, 1).flatten(1)
            gate = F.relu(self.se_fc1(gate))
            gate = torch.sigmoid(self.se_fc2(gate)).view(out.size(0), -1, 1, 1)
            out = out * gate
        out = F.relu(out + residual)  # skip connection then ReLU
        return out


class Whisker(nn.Module):
    """AlphaZero-style dual-headed network for chess.

    Architecture:
        Stem   → N residual blocks → (policy head, value head)

    Args:
        channels:    Number of convolutional filters (default 128).
        num_blocks:  Number of residual blocks in the tower (default 8).
        policy_size: Number of policy output logits (default 4096).
        policy_channels: Width of the policy head (default 8).
        value_channels: Width of the value head (default 8).
        use_se:      Add global channel attention to each residual block.
    """

    def __init__(
        self,
        channels: int = 128,
        num_blocks: int = 8,
        policy_size: int = 4096,
        policy_channels: int = 8,
        value_channels: int = 8,
        use_se: bool = True,
    ) -> None:
        super().__init__()

        self.stem_conv = nn.Conv2d(18, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)

        self.tower = nn.Sequential(
            *[ResidualBlock(channels, use_se=use_se) for _ in range(num_blocks)]
        )

        self.policy_conv = nn.Conv2d(channels, policy_channels, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(policy_channels)
        self.policy_fc = nn.Linear(policy_channels * 8 * 8, policy_size)

        self.value_conv = nn.Conv2d(channels, value_channels, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_channels)
        self.value_fc1 = nn.Linear(value_channels * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

        self._init_weights()
        # Start each residual branch close to identity for stable deeper training.
        for block in self.tower:
            nn.init.zeros_(block.bn2.weight)

    def _init_weights(self) -> None:
        """Apply Kaiming init to conv layers and Xavier init to linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Board tensor of shape (B, 18, 8, 8).

        Returns:
            policy_logits: (B, 4096) raw logits (no softmax).
            value:         (B, 1)    scalar value in [−1, +1].
        """
        s = F.relu(self.stem_bn(self.stem_conv(x)))

        t = self.tower(s)

        p = F.relu(self.policy_bn(self.policy_conv(t)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)  # (B, 4096)

        v = F.relu(self.value_bn(self.value_conv(t)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))    # (B, 256)
        value = torch.tanh(self.value_fc2(v))  # (B, 1)

        return policy_logits, value

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ChessDataset(Dataset):
    """Lazily-loaded HDF5 chess position dataset from one file or a folder.

    Each sample returns:
        board  : float32 tensor of shape (18, 8, 8)
        policy : int64 scalar  (move index, 0–4095)
        value  : float32 scalar (−1, 0, or +1)

    For a folder, all accepted ``*.h5`` files are concatenated in filename
    order. Files are opened lazily so each DataLoader worker owns its handles.
    """

    def __init__(self, hdf5_path: str, allow_incomplete: bool = False) -> None:
        source = Path(hdf5_path)
        if source.is_dir():
            candidates = sorted(source.glob("*.h5"))
            if not candidates:
                raise FileNotFoundError(f"no .h5 files found in data folder: {source}")
        elif source.is_file():
            candidates = [source]
        else:
            raise FileNotFoundError(f"data file or folder not found: {source}")

        self.hdf5_paths: list[Path] = []
        self.lengths: list[int] = []
        game_id_parts: list[np.ndarray] = []
        self.skipped_files: list[Path] = []
        self.incomplete_files: list[Path] = []
        game_id_offset = 0
        for candidate in candidates:
            try:
                with h5py.File(candidate, "r") as f:
                    complete = f.attrs.get("complete")
                    valid_schema = all(name in f for name in ("boards", "policies", "values"))
                    length = int(f["boards"].shape[0]) if valid_schema else 0
                    if valid_schema:
                        valid_schema = (
                            f["policies"].shape[0] == length
                            and f["values"].shape[0] == length
                        )
                    if valid_schema and "game_ids" not in f:
                        raise ValueError(
                            f"dataset {candidate} has no game IDs; reprocess it with "
                            "the current src.prepare_data.py before training"
                        )
                    if valid_schema:
                        valid_schema = f["game_ids"].shape[0] == length
                    incomplete = complete is not None and not bool(complete)
                    if (
                        not valid_schema
                        or length == 0
                        or (incomplete and not allow_incomplete)
                    ):
                        if source.is_dir():
                            self.skipped_files.append(candidate)
                            continue
                        raise ValueError(f"incomplete or empty chess dataset: {candidate}")
                    self.hdf5_paths.append(candidate)
                    self.lengths.append(length)
                    game_ids = np.asarray(f["game_ids"][:], dtype=np.int64)
                    if game_ids.size and int(game_ids.min()) < 0:
                        raise ValueError(f"dataset {candidate} has invalid negative game IDs")
                    game_id_parts.append(game_ids + game_id_offset)
                    if game_ids.size:
                        game_id_offset += int(game_ids.max()) + 1
                    if incomplete:
                        self.incomplete_files.append(candidate)
            except OSError as exc:
                if source.is_dir():
                    self.skipped_files.append(candidate)
                    continue
                raise ValueError(f"could not read chess dataset: {candidate}") from exc

        if not self.hdf5_paths:
            raise ValueError(f"no usable, non-empty HDF5 datasets found in {source}")

        self._ends = np.cumsum(self.lengths)
        self._length = int(self._ends[-1])
        self.game_ids = np.concatenate(game_id_parts)
        self.game_count = int(np.unique(self.game_ids).size)
        self._files: Dict[int, h5py.File] = {}  # opened lazily per worker

    def __len__(self) -> int:
        return self._length

    def _ensure_open(self, file_index: int) -> h5py.File:
        """Open one HDF5 file lazily in the current DataLoader worker."""
        if file_index not in self._files:
            self._files[file_index] = h5py.File(self.hdf5_paths[file_index], "r")
        return self._files[file_index]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        file_index = int(np.searchsorted(self._ends, idx, side="right"))
        file_start = 0 if file_index == 0 else int(self._ends[file_index - 1])
        local_idx = idx - file_start
        data_file = self._ensure_open(file_index)

        board = data_file["boards"][local_idx].astype(np.float32)  # (18, 8, 8)
        policy = int(data_file["policies"][local_idx])             # scalar
        value = float(data_file["values"][local_idx])              # scalar

        return (
            torch.from_numpy(board),
            torch.tensor(policy, dtype=torch.long),
            torch.tensor(value, dtype=torch.float32),
        )


def concatenate_chess_datasets(datasets: list[ChessDataset]) -> Dataset:
    """Concatenate datasets while assigning unique IDs to all source games."""
    combined = ConcatDataset(datasets)
    game_id_parts = []
    offset = 0
    for dataset in datasets:
        game_id_parts.append(dataset.game_ids + offset)
        if dataset.game_ids.size:
            offset += int(dataset.game_ids.max()) + 1
    combined.game_ids = np.concatenate(game_id_parts)
    combined.game_count = sum(dataset.game_count for dataset in datasets)
    return combined


def make_data_loaders(
    dataset: ChessDataset,
    batch_size: int,
    num_workers: int,
    replay_dataset: Optional[Dataset] = None,
    replay_ratio: float = 0.25,
):
    """Create loaders, optionally replay-sampling older data while fine-tuning.

    The target dataset supplies the validation split. When replay data is
    provided, the training sampler draws the requested fraction from it while
    drawing the rest from the target dataset.
    """

    unique_games = np.unique(dataset.game_ids)
    if unique_games.size < 2:
        raise ValueError("each training dataset must contain at least two games")
    np.random.RandomState(42).shuffle(unique_games)
    val_game_count = min(max(1, int(0.20 * unique_games.size)), unique_games.size - 1)
    val_game_ids = unique_games[:val_game_count]
    val_positions = np.isin(dataset.game_ids, val_game_ids)
    train_indices = np.flatnonzero(~val_positions)
    val_indices = np.flatnonzero(val_positions)
    train_game_count = int(unique_games.size - val_game_count)
    val_sampler = SubsetRandomSampler(val_indices.tolist())

    if replay_dataset is None:
        train_dataset = dataset
        train_sampler = SubsetRandomSampler(train_indices.tolist())
    else:
        if not 0.0 < replay_ratio < 1.0:
            raise ValueError("--replay-ratio must be between 0 and 1 when --replay-data is set")
        if len(replay_dataset) < 1:
            raise ValueError("replay dataset must contain at least one position")
        replay_games = np.unique(replay_dataset.game_ids)
        if replay_games.size < 2:
            raise ValueError("replay dataset must contain at least two games")
        np.random.RandomState(43).shuffle(replay_games)
        replay_val_game_count = min(
            max(1, int(0.20 * replay_games.size)), replay_games.size - 1
        )
        replay_val_games = replay_games[:replay_val_game_count]
        replay_train_indices = np.flatnonzero(
            ~np.isin(replay_dataset.game_ids, replay_val_games)
        )
        train_dataset = ConcatDataset(
            [Subset(dataset, train_indices.tolist()),
             Subset(replay_dataset, replay_train_indices.tolist())]
        )
        weights = torch.cat(
            [
                torch.full(
                    (len(train_indices),),
                    (1.0 - replay_ratio) / len(train_indices),
                    dtype=torch.double,
                ),
                torch.full(
                    (len(replay_train_indices),),
                    replay_ratio / len(replay_train_indices),
                    dtype=torch.double,
                ),
            ]
        )
        train_sampler = WeightedRandomSampler(
            weights,
            num_samples=len(train_indices) + len(replay_train_indices),
            replacement=True,
            generator=torch.Generator().manual_seed(42),
        )

    val_workers = num_workers // 2
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False,
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=val_workers,
        pin_memory=True,
        prefetch_factor=2 if val_workers > 0 else None,
        drop_last=False,
    )

    return (
        train_loader,
        val_loader,
        len(train_indices),
        len(val_indices),
        train_game_count,
        val_game_count,
    )


def legal_move_indices_from_planes(planes: np.ndarray) -> list[int]:
    """Reconstruct legal move labels from the stored 18-plane board encoding."""
    board = chess.Board(None)
    piece_types = (
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    )
    for plane, piece_type in enumerate(piece_types):
        for row, col in np.argwhere(planes[plane] > 0.5):
            board.set_piece_at(chess.square(int(col), int(row)), chess.Piece(piece_type, chess.WHITE))
        for row, col in np.argwhere(planes[plane + 6] > 0.5):
            board.set_piece_at(chess.square(int(col), int(row)), chess.Piece(piece_type, chess.BLACK))

    castling_rights = 0
    if planes[12, 0, 0] > 0.5:
        castling_rights |= chess.BB_H1
    if planes[13, 0, 0] > 0.5:
        castling_rights |= chess.BB_A1
    if planes[14, 0, 0] > 0.5:
        castling_rights |= chess.BB_H8
    if planes[15, 0, 0] > 0.5:
        castling_rights |= chess.BB_A8
    board.castling_rights = castling_rights

    ep_squares = np.argwhere(planes[16] > 0.5)
    board.ep_square = (
        chess.square(int(ep_squares[0][1]), int(ep_squares[0][0]))
        if len(ep_squares)
        else None
    )
    board.turn = chess.WHITE if planes[17, 0, 0] > 0.5 else chess.BLACK
    return [move.from_square * 64 + move.to_square for move in board.legal_moves]


@torch.no_grad()
def validate(
    model: Whisker,
    val_loader: DataLoader,
    device: torch.device,
    max_samples: int = 10_000,
) -> Dict[str, float]:
    """Run validation and return a metrics dict.

    Metrics:
        val_policy_loss, val_value_loss, val_total_loss,
        raw_top1_accuracy, legal_top1_accuracy, top5_accuracy, value_mae
    """
    model.eval()

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_top1 = 0
    total_legal_top1 = 0
    total_top5 = 0
    total_value_ae = 0.0
    total_samples = 0

    for boards, policies, values in val_loader:
        batch_legal_indices = [
            legal_move_indices_from_planes(board.numpy()) for board in boards
        ]
        legal_mask = np.zeros((len(batch_legal_indices), 4096), dtype=np.bool_)
        for row, indices in enumerate(batch_legal_indices):
            legal_mask[row, indices] = True
        boards = boards.to(device, non_blocking=True)
        policies = policies.to(device, non_blocking=True)
        values = values.to(device, non_blocking=True).unsqueeze(1)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            policy_logits, value_pred = model(boards)
            p_loss = F.cross_entropy(policy_logits, policies, reduction="sum")
            v_loss = F.mse_loss(value_pred, values, reduction="sum")

        batch_n = boards.size(0)
        total_policy_loss += p_loss.item()
        total_value_loss += v_loss.item()

        pred_top1 = policy_logits.argmax(dim=1)
        total_top1 += (pred_top1 == policies).sum().item()
        legal_mask_tensor = torch.from_numpy(legal_mask).to(policy_logits.device)
        legal_top1 = policy_logits.masked_fill(~legal_mask_tensor, -torch.inf).argmax(dim=1)
        total_legal_top1 += (legal_top1 == policies).sum().item()

        _, pred_top5 = policy_logits.topk(5, dim=1)
        total_top5 += (pred_top5 == policies.unsqueeze(1)).any(dim=1).sum().item()

        total_value_ae += (value_pred.squeeze(1) - values.squeeze(1)).abs().sum().item()

        total_samples += batch_n
        if total_samples >= max_samples:
            break

    n = max(total_samples, 1)
    metrics = {
        "val_policy_loss": total_policy_loss / n,
        "val_value_loss": total_value_loss / n,
        "val_total_loss": (total_policy_loss + total_value_loss) / n,
        "raw_top1_accuracy": total_top1 / n * 100.0,
        "legal_top1_accuracy": total_legal_top1 / n * 100.0,
        "top5_accuracy": total_top5 / n * 100.0,
        "value_mae": total_value_ae / n,
    }

    model.train()
    return metrics


def save_checkpoint(
    path: str,
    model: Whisker,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    config: Dict[str, Any],
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Save a training checkpoint."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "config": config,
    }
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if metrics is not None:
        checkpoint["metrics"] = metrics
    torch.save(checkpoint, path)


def retain_only_checkpoint(
    output_dir: str,
    keep_path: str,
    logger: logging.Logger,
) -> None:
    """Delete older step checkpoints after a newer checkpoint is safely saved."""
    keep = Path(keep_path).resolve()
    for candidate in Path(output_dir).glob("checkpoint_step_*.pt"):
        if candidate.resolve() == keep:
            continue
        try:
            candidate.unlink()
        except OSError as exc:
            logger.warning("Could not remove old checkpoint %s: %s", candidate, exc)


def read_model_state(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """Read and normalize model weights from a training checkpoint or state dict."""
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("fine-tune checkpoint must contain a state dict or checkpoint dictionary")

    state = payload
    for key in ("model", "model_state_dict", "state_dict"):
        if key in payload and isinstance(payload[key], dict):
            state = payload[key]
            break

    normalized_state = {}
    for key, value in state.items():
        while key.startswith("module.") or key.startswith("_orig_mod."):
            key = key.split(".", 1)[1]
        normalized_state[key] = value
    return normalized_state


def infer_model_architecture(path: str) -> Dict[str, Any]:
    """Infer Whisker dimensions so resume/fine-tune use the checkpoint's model."""
    state = read_model_state(path, torch.device("cpu"))
    try:
        block_ids = {
            int(key.split(".")[1])
            for key in state
            if key.startswith("tower.") and key.endswith(".conv1.weight")
        }
        if not block_ids:
            raise ValueError("checkpoint has no residual blocks")
        return {
            "channels": int(state["stem_conv.weight"].shape[0]),
            "blocks": max(block_ids) + 1,
            "policy_channels": int(state["policy_conv.weight"].shape[0]),
            "value_channels": int(state["value_conv.weight"].shape[0]),
            "use_se": any(key.startswith("tower.0.se_fc1.") for key in state),
        }
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError("checkpoint does not match the Whisker architecture") from exc


def load_model_weights(path: str, model: Whisker, device: torch.device) -> None:
    """Load model weights only, for a fine-tuning stage with fresh optimizer state."""
    normalized_state = read_model_state(path, device)

    try:
        model.load_state_dict(normalized_state)
    except RuntimeError as exc:
        raise ValueError(
            "fine-tune checkpoint architecture does not match this model; "
            "use the same --blocks, --channels, --policy-channels, "
            "--value-channels, and SE setting as the first stage"
        ) from exc


def load_checkpoint(
    path: str,
    model: Whisker,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    expected_data: Optional[str] = None,
    expected_data_files: Optional[list[str]] = None,
) -> int:
    """Load a checkpoint and return the step to resume from."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    saved_data = ckpt.get("config", {}).get("data")
    if expected_data and saved_data:
        if os.path.abspath(saved_data) != os.path.abspath(expected_data):
            raise ValueError(
                "--resume checkpoint was trained with a different dataset; "
                "use --fine-tune to switch datasets"
            )
    saved_files = ckpt.get("config", {}).get("data_files")
    if expected_data_files and saved_files:
        normalized_saved = [os.path.abspath(path) for path in saved_files]
        normalized_expected = [os.path.abspath(path) for path in expected_data_files]
        if normalized_saved != normalized_expected:
            raise ValueError(
                "--resume dataset folder contains a different set of monthly files; "
                "finish downloading before training or use --fine-tune for a new stage"
            )
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    step = int(ckpt["step"])
    if scheduler is not None:
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        else:
            # Older checkpoints did not save scheduler state. Position the
            # cosine schedule directly instead of stepping through every past step.
            scheduler.step(step)
    return step


def infinite_loader(loader: DataLoader):
    """Yield batches from *loader* in an infinite loop (for step-based training)."""
    while True:
        yield from loader


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an AlphaZero-style chess network.",
    )
    parser.add_argument("--data", required=True, help="Path to HDF5 dataset")
    parser.add_argument(
        "--output-dir", default=None,
        help="Checkpoint directory (defaults to checkpoints or a source-specific fine-tune folder)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=512,
        help="Positions per optimizer step (default: 512; raise to use more GPU capacity)",
    )
    parser.add_argument(
        "--steps", type=int, default=500_000,
        help="Total target for --resume; number of steps for a new/fine-tune run",
    )
    parser.add_argument("--blocks", type=int, default=None, help="Residual blocks")
    parser.add_argument("--channels", type=int, default=None, help="Conv channels")
    parser.add_argument(
        "--policy-channels", type=int, default=None,
        help="Policy head width (default: 8; old checkpoints use 2)",
    )
    parser.add_argument(
        "--value-channels", type=int, default=None,
        help="Value head width (default: 8; old checkpoints use 1)",
    )
    parser.add_argument(
        "--no-se", action="store_true", default=None,
        help="Disable squeeze-and-excitation blocks",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="DataLoader subprocesses (default: 0, keeping training in one process)",
    )
    parser.add_argument(
        "--allow-incomplete-data",
        action="store_true",
        help="Train on non-empty HDF5 files marked incomplete after an interrupted parse",
    )
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--resume", type=str, default=None,
        help="Resume an interrupted run, restoring optimizer and scheduler state",
    )
    run_mode.add_argument(
        "--fine-tune", type=str, default=None,
        help="Initialize from model weights only; start a new training stage",
    )
    parser.add_argument(
        "--replay-data", nargs="+", default=None,
        help="One or more older HDF5 datasets to mix into a fine-tuning stage",
    )
    parser.add_argument(
        "--replay-ratio", type=float, default=0.25,
        help="Fraction of fine-tuning samples drawn from --replay-data (default: 0.25)",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument(
        "--fine-tune-lr", type=float, default=5e-5,
        help="Fresh learning rate for --fine-tune stages",
    )
    args = parser.parse_args()

    checkpoint_arch = None
    checkpoint_path = args.resume or args.fine_tune
    if checkpoint_path:
        checkpoint_arch = infer_model_architecture(checkpoint_path)

    architecture_defaults = {
        "blocks": 8,
        "channels": 128,
        "policy_channels": 8,
        "value_channels": 8,
    }
    for name, default in architecture_defaults.items():
        requested = getattr(args, name)
        saved = checkpoint_arch[name] if checkpoint_arch else default
        if requested is not None and checkpoint_arch and requested != saved:
            parser.error(
                f"--{name.replace('_', '-')}={requested} does not match checkpoint "
                f"value {saved}; this workflow requires the same architecture"
            )
        setattr(args, name, saved if requested is None else requested)

    saved_use_se = checkpoint_arch["use_se"] if checkpoint_arch else True
    if checkpoint_arch and args.no_se and saved_use_se:
        parser.error("--no-se does not match this checkpoint's squeeze-and-excitation architecture")
    args.use_se = False if args.no_se else saved_use_se

    if args.replay_data and not args.fine_tune:
        parser.error("--replay-data requires --fine-tune")
    if args.replay_data and not 0.0 < args.replay_ratio < 1.0:
        parser.error("--replay-ratio must be between 0 and 1")
    if args.replay_data:
        target_path = os.path.realpath(args.data)
        replay_paths = [os.path.realpath(path) for path in args.replay_data]
        if target_path in replay_paths:
            parser.error("--replay-data and --data must point to different datasets")
        if len(set(replay_paths)) != len(replay_paths):
            parser.error("--replay-data must not contain duplicate datasets")
    if args.steps < 1 or args.batch_size < 1:
        parser.error("--steps and --batch-size must be positive")
    if args.workers < 0:
        parser.error("--workers must be non-negative")
    if min(args.blocks, args.channels, args.policy_channels, args.value_channels) < 1:
        parser.error("all model dimensions must be positive")
    if args.lr <= 0 or args.fine_tune_lr <= 0:
        parser.error("--lr and --fine-tune-lr must be positive")
    if args.output_dir is None:
        if args.fine_tune:
            source = Path(args.fine_tune)
            source_tag = f"{source.parent.name or 'source'}_{source.stem}"
            args.output_dir = os.path.join("checkpoints", "fine_tune", source_tag)
        elif args.resume:
            args.output_dir = str(Path(args.resume).parent)
        else:
            args.output_dir = "checkpoints"

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)

    config: Dict[str, Any] = {
        "data": args.data,
        "allow_incomplete_data": args.allow_incomplete_data,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "blocks": args.blocks,
        "channels": args.channels,
        "policy_channels": args.policy_channels,
        "value_channels": args.value_channels,
        "use_se": args.use_se,
        "lr": args.fine_tune_lr if args.fine_tune else args.lr,
        "weight_decay": 1e-4,
        "policy_size": 4096,
        "fine_tune_from": args.fine_tune,
        "replay_data": args.replay_data,
        "replay_ratio": args.replay_ratio if args.replay_data else 0.0,
    }

    # is_available() checks both that this PyTorch build has CUDA support and
    # that a usable CUDA device/driver is visible to the process.
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info("Device: %s", device)
        logger.info("GPU: %s", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        logger.info("Device: %s", device)
        if not torch.backends.cuda.is_built():
            logger.warning(
                "CUDA is unavailable because this PyTorch installation was "
                "built without CUDA support; using CPU."
            )
        else:
            logger.warning(
                "This PyTorch build supports CUDA, but no usable CUDA GPU was "
                "detected; using CPU. Check GPU/driver visibility."
            )

    dataset = ChessDataset(args.data, allow_incomplete=args.allow_incomplete_data)
    config["data_files"] = [str(path.resolve()) for path in dataset.hdf5_paths]
    logger.info(
        f"Dataset: {len(dataset):,} positions from {dataset.game_count:,} games "
        f"in {args.data}"
    )
    if len(dataset.hdf5_paths) > 1:
        logger.info(f"Combined {len(dataset.hdf5_paths)} monthly HDF5 files")
    if dataset.skipped_files:
        logger.warning(
            "Ignored incomplete or empty HDF5 files: "
            + ", ".join(str(path) for path in dataset.skipped_files)
        )
    if dataset.incomplete_files:
        logger.warning(
            "Using partial HDF5 files saved by an interrupted parse: "
            + ", ".join(str(path) for path in dataset.incomplete_files)
        )

    replay_dataset = None
    if args.replay_data:
        replay_datasets = [
            ChessDataset(path, allow_incomplete=args.allow_incomplete_data)
            for path in args.replay_data
        ]
        replay_dataset = concatenate_chess_datasets(replay_datasets)
        logger.info(
            f"Replay dataset: {len(replay_dataset):,} positions from "
            f"{replay_dataset.game_count:,} games across {len(args.replay_data)} dataset(s) "
            f"({args.replay_ratio:.0%} of sampled training positions)"
        )
        for replay_source, replay_part in zip(args.replay_data, replay_datasets):
            if replay_part.skipped_files:
                logger.warning(
                    f"Ignored incomplete or empty replay files from {replay_source}: "
                    + ", ".join(str(path) for path in replay_part.skipped_files)
                )
            if replay_part.incomplete_files:
                logger.warning(
                    f"Using partial replay HDF5 files from {replay_source}: "
                    + ", ".join(str(path) for path in replay_part.incomplete_files)
                )

    (
        train_loader,
        val_loader,
        train_size,
        val_size,
        train_game_count,
        val_game_count,
    ) = make_data_loaders(
        dataset,
        args.batch_size,
        args.workers,
        replay_dataset=replay_dataset,
        replay_ratio=args.replay_ratio,
    )
    logger.info(
        f"Split: {train_size:,} train / {val_size:,} val positions; "
        f"{train_game_count:,} train / {val_game_count:,} validation games (80/20)"
    )

    model = Whisker(
        channels=args.channels,
        num_blocks=args.blocks,
        policy_size=config["policy_size"],
        policy_channels=args.policy_channels,
        value_channels=args.value_channels,
        use_se=config["use_se"],
    ).to(device)
    if args.fine_tune:
        load_model_weights(args.fine_tune, model, device)
        logger.info(
            f"Initialized model weights from {args.fine_tune}; "
            "optimizer and learning-rate schedule start fresh"
        )
    logger.info(
        f"Whisker: {args.blocks} blocks × {args.channels} channels, "
        f"policy/value heads={args.policy_channels}/{args.value_channels}, "
        f"SE={args.use_se}, {model.count_parameters():,} parameters"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(
            args.resume,
            model,
            optimizer,
            scaler,
            device,
            scheduler=scheduler,
            expected_data=args.data,
            expected_data_files=[str(path.resolve()) for path in dataset.hdf5_paths],
        )
        if scheduler.T_max != args.steps:
            raise ValueError(
                "--resume must use the same --steps total as the saved scheduler; "
                "use --fine-tune to start a new training stage"
            )
        logger.info(f"Resumed from {args.resume} at step {start_step:,}")

    tracker = ExperimentTracker(config, args.output_dir)
    logger.info(f"Logging backend: {tracker.backend}")

    best_checkpoint_path = Path(args.output_dir) / "best_checkpoint.pt"
    best_legal_top1 = float("-inf")
    best_policy_loss = float("inf")
    best_step = 0
    if best_checkpoint_path.is_file():
        try:
            best_checkpoint = torch.load(
                best_checkpoint_path, map_location="cpu", weights_only=False
            )
            best_metrics = best_checkpoint.get("metrics", {})
            best_legal_top1 = float(best_metrics.get("legal_top1_accuracy", float("-inf")))
            best_policy_loss = float(best_metrics.get("val_policy_loss", float("inf")))
            best_step = int(best_checkpoint.get("step", 0))
            logger.info(
                f"Best checkpoint so far: step={best_step:,}, "
                f"legal_top1={best_legal_top1:.2f}%"
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            logger.warning("Could not read existing best checkpoint metadata: %s", exc)

    last_validation_step = 0

    def evaluate_and_save_best(validation_step: int) -> Dict[str, float]:
        nonlocal best_legal_top1, best_policy_loss, best_step, last_validation_step
        val_metrics = validate(model, val_loader, device)
        last_validation_step = validation_step
        tracker.log(val_metrics, validation_step)
        logger.info(
            f"[VAL] step={validation_step:>7,} | "
            f"p_loss={val_metrics['val_policy_loss']:.4f} | "
            f"v_loss={val_metrics['val_value_loss']:.4f} | "
            f"legal_top1={val_metrics['legal_top1_accuracy']:.2f}% | "
            f"raw_top1={val_metrics['raw_top1_accuracy']:.2f}% | "
            f"top5={val_metrics['top5_accuracy']:.2f}% | "
            f"v_mae={val_metrics['value_mae']:.4f}"
        )
        current_legal_top1 = val_metrics["legal_top1_accuracy"]
        current_policy_loss = val_metrics["val_policy_loss"]
        improves = current_legal_top1 > best_legal_top1 or (
            current_legal_top1 == best_legal_top1
            and current_policy_loss < best_policy_loss
        )
        if improves:
            temp_path = best_checkpoint_path.with_suffix(".tmp")
            save_checkpoint(
                str(temp_path),
                model,
                optimizer,
                scaler,
                validation_step,
                config,
                scheduler,
                metrics=val_metrics,
            )
            os.replace(temp_path, best_checkpoint_path)
            best_legal_top1 = current_legal_top1
            best_policy_loss = current_policy_loss
            best_step = validation_step
            logger.info(
                f"[BEST] step={best_step:,} | legal_top1={best_legal_top1:.2f}% | "
                f"saved {best_checkpoint_path}"
            )
        return val_metrics

    logger.info(f"Starting training for {args.steps:,} steps (batch_size={args.batch_size})")
    model.train()

    data_iter = infinite_loader(train_loader)
    step = start_step
    batch_time = time.time()
    samples_since_log = 0
    interrupted = False

    pbar = tqdm(
        initial=start_step,
        total=args.steps,
        desc="Training",
        unit="step",
        smoothing=0.05,
    )

    try:
        while step < args.steps:
            boards, policies, values = next(data_iter)
            samples_since_log += boards.size(0)
            boards = boards.to(device, non_blocking=True)
            policies = policies.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True).unsqueeze(1)  # (B,) → (B,1)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                policy_logits, value_pred = model(boards)
                policy_loss = F.cross_entropy(policy_logits, policies)
                value_loss = F.mse_loss(value_pred, values)
                total_loss = policy_loss + value_loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            step += 1

            if step % 100 == 0:
                elapsed = time.time() - batch_time
                samples_per_sec = samples_since_log / max(elapsed, 1e-6)
                lr = scheduler.get_last_lr()[0]

                metrics = {
                    "step": step,
                    "lr": lr,
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "total_loss": total_loss.item(),
                    "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    "samples_per_sec": samples_per_sec,
                }
                tracker.log(metrics, step)
                logger.info(
                    f"step={step:>7,} | lr={lr:.2e} | "
                    f"p_loss={policy_loss.item():.4f} | v_loss={value_loss.item():.4f} | "
                    f"total={total_loss.item():.4f} | grad={grad_norm:.3f} | "
                    f"{samples_per_sec:.0f} samp/s"
                )
                batch_time = time.time()
                samples_since_log = 0

            if step % 5_000 == 0:
                evaluate_and_save_best(step)

            if step % 10_000 == 0:
                ckpt_path = os.path.join(args.output_dir, f"checkpoint_step_{step}.pt")
                save_checkpoint(ckpt_path, model, optimizer, scaler, step, config, scheduler)
                retain_only_checkpoint(args.output_dir, ckpt_path, logger)
                logger.info(f"[CKPT] Saved {ckpt_path}")

            pbar.update(1)

        if last_validation_step != step:
            evaluate_and_save_best(step)

    except KeyboardInterrupt:
        interrupted = True
        logger.info("Interrupted — saving emergency checkpoint...")
        ckpt_path = os.path.join(args.output_dir, f"checkpoint_step_{step}_emergency.pt")
        save_checkpoint(ckpt_path, model, optimizer, scaler, step, config, scheduler)
        retain_only_checkpoint(args.output_dir, ckpt_path, logger)
        logger.info(f"Emergency checkpoint saved: {ckpt_path}")

    finally:
        pbar.close()
        tracker.close()

    if interrupted:
        logger.info("Training interrupted; the emergency checkpoint is the retained model.")
    else:
        final_path = os.path.join(args.output_dir, f"checkpoint_step_{step}.pt")
        if not os.path.exists(final_path):
            save_checkpoint(final_path, model, optimizer, scaler, step, config, scheduler)
        retain_only_checkpoint(args.output_dir, final_path, logger)
        logger.info(f"Final checkpoint: {final_path}")
        logger.info(
            f"Best validation checkpoint: {best_checkpoint_path} "
            f"(step={best_step:,}, legal_top1={best_legal_top1:.2f}%)"
        )
        logger.info("Training complete.")


if __name__ == "__main__":
    main()
