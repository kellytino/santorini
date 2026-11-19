"""Shared feature extraction and model loading for the Phase 2 Santorini bot."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

FEATURE_VERSION = "santorini-v1"
# 25 heights + 25 own-worker occupancy + 25 opponent-worker occupancy
# + 6 strategic summary features.
INPUT_SIZE = 81


def _mobility(board, player: int) -> int:
    occupied = {(w.row, w.column) for group in board.workers for w in group}
    count = 0
    for worker in board.workers[player]:
        source = board.heights[worker.row][worker.column]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                r, c = worker.row + dr, worker.column + dc
                if 0 <= r < 5 and 0 <= c < 5 and (r, c) not in occupied:
                    h = board.heights[r][c]
                    if h < 4 and h <= source + 1:
                        count += 1
    return count


def _level_access(board, player: int, level: int) -> int:
    occupied = {(w.row, w.column) for group in board.workers for w in group}
    count = 0
    for worker in board.workers[player]:
        source = board.heights[worker.row][worker.column]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                r, c = worker.row + dr, worker.column + dc
                if 0 <= r < 5 and 0 <= c < 5 and (r, c) not in occupied:
                    if board.heights[r][c] == level and level <= source + 1:
                        count += 1
    return count


def encode_state(board, perspective: int) -> np.ndarray:
    """Encode a position from one player's perspective; output has 81 floats."""
    values = [board.heights[r][c] / 4.0 for r in range(5) for c in range(5)]
    own = np.zeros(25, dtype=np.float32)
    opp = np.zeros(25, dtype=np.float32)
    for w in board.workers[perspective]:
        own[w.row * 5 + w.column] = 1.0
    for w in board.workers[1 - perspective]:
        opp[w.row * 5 + w.column] = 1.0
    values.extend(own.tolist())
    values.extend(opp.tolist())

    own_heights = [board.heights[w.row][w.column] for w in board.workers[perspective]]
    opp_heights = [board.heights[w.row][w.column] for w in board.workers[1 - perspective]]
    summary = [
        (max(own_heights) - max(opp_heights)) / 3.0,
        (_mobility(board, perspective) - _mobility(board, 1 - perspective)) / 16.0,
        (_level_access(board, perspective, 3) - _level_access(board, 1 - perspective, 3)) / 4.0,
        (_level_access(board, perspective, 2) - _level_access(board, 1 - perspective, 2)) / 4.0,
        sum(own_heights) / 6.0,
        sum(opp_heights) / 6.0,
    ]
    values.extend(summary)
    return np.asarray(values, dtype=np.float32)


def model_paths(model_dir: str | Path):
    path = Path(model_dir)
    return path / "santorini_value_model.npz", path / "metadata.json"


class ValueModel:
    """Small 81-64-32-1 MLP with tanh output in approximately [-1, 1]."""
    def __init__(self, model_dir: str | Path):
        weights_path, metadata_path = model_paths(model_dir)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Trained model not found: {weights_path}. Run bot/train_model.py first."
            )
        data = np.load(weights_path)
        self.w1, self.b1 = data["w1"], data["b1"]
        self.w2, self.b2 = data["w2"], data["b2"]
        self.w3, self.b3 = data["w3"], data["b3"]
        if metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            self.metadata = {}

    def predict(self, features: np.ndarray) -> float:
        x = np.asarray(features, dtype=np.float32).reshape(1, -1)
        h1 = np.maximum(0.0, x @ self.w1 + self.b1)
        h2 = np.maximum(0.0, h1 @ self.w2 + self.b2)
        out = np.tanh(h2 @ self.w3 + self.b3)
        return float(out[0, 0])
