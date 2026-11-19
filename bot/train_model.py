"""Generate self-play positions and train a supervised board-value MLP.

Run from the submission directory:
    python bot/train_model.py --games 500 --seed 20261119
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BOT_DIR = ROOT / "bot"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(BOT_DIR))

import random_bot
import search_bot
from santorini_engine import apply_action, fingerprint, legal_actions_from, public_state_for, setup
from phase2_common import FEATURE_VERSION, INPUT_SIZE, encode_state, model_paths


def play_training_game(seed: int, search_player: int, search_time: float, max_turns: int, opponent_mode: str):
    """Play one game; retain each pre-move state and its eventual outcome."""
    board = setup(seed)
    history = [fingerprint(board)]
    samples = []
    reason = "turn_limit"
    winner = None

    for _ in range(max_turns):
        player = board.current_player
        actions = legal_actions_from(board)
        if not actions:
            winner = 1 - player
            reason = "no_legal_action"
            break

        samples.append((encode_state(board, player), player))
        public = public_state_for(board, player)
        if opponent_mode == "selfplay" or player == search_player:
            action = search_bot.choose_action(public, actions, search_time)
        else:
            action = random_bot.choose_action(public, actions, search_time)

        board = apply_action(board, action)
        if board.winner is not None:
            winner = board.winner
            reason = "win"
            break

        fp = fingerprint(board)
        history.append(fp)
        if history.count(fp) >= 3:
            reason = "repetition"
            break
    else:
        reason = "turn_limit"

    # Convert the eventual game result into a value for the player to move
    # in each recorded position: +1 win, -1 loss, 0 draw.
    labelled = []
    for features, side_to_move in samples:
        target = 0.0 if winner is None else (1.0 if winner == side_to_move else -1.0)
        labelled.append((features, target))
    return labelled, reason, len(samples), winner

def collect_games(
    num_games: int,
    start_seed: int,
    search_time: float,
    max_turns: int,
    opponent_mode: str,
    label: str,
):
    """Generate a dataset from a specified range of game seeds."""
    all_x, all_y, game_ids = [], [], []
    reasons = Counter()

    for i in range(num_games):
        search_player = i % 2

        game_samples, reason, n_positions, winner = play_training_game(
            start_seed + i,
            search_player,
            search_time,
            max_turns,
            opponent_mode,
        )

        reasons[reason] += 1

        for features, target in game_samples:
            all_x.append(features)
            all_y.append(target)
            game_ids.append(i)

        print(
            f"{label} game {i + 1:04d}/{num_games}: "
            f"{reason}, positions={n_positions}",
            flush=True,
        )

    x = np.asarray(all_x, dtype=np.float32)
    y = np.asarray(all_y, dtype=np.float32).reshape(-1, 1)
    game_ids = np.asarray(game_ids, dtype=np.int32)

    return x, y, game_ids, reasons

def initialize_network(rng: np.random.Generator):
    def weight(fan_in, fan_out):
        return (rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)).astype(np.float32)
    return {
        "w1": weight(INPUT_SIZE, 64), "b1": np.zeros((1, 64), np.float32),
        "w2": weight(64, 32), "b2": np.zeros((1, 32), np.float32),
        "w3": (rng.standard_normal((32, 1)) * np.sqrt(1.0 / 32)).astype(np.float32),
        "b3": np.zeros((1, 1), np.float32),
    }


def forward(x, p):
    z1 = x @ p["w1"] + p["b1"]
    a1 = np.maximum(z1, 0)
    z2 = a1 @ p["w2"] + p["b2"]
    a2 = np.maximum(z2, 0)
    z3 = a2 @ p["w3"] + p["b3"]
    y = np.tanh(z3)
    return y, (x, z1, a1, z2, a2, y)


def evaluate_predictions(predictions, targets):
    """Calculate several metrics for model evaluation."""
    mse = float(np.mean((predictions - targets) ** 2))

    # Determine whether the predicted sign matches the actual outcome.
    decisive = targets.flatten() != 0
    if np.any(decisive):
        accuracy = float(np.mean(
            np.sign(predictions.flatten()[decisive])
            == np.sign(targets.flatten()[decisive])
        ))
    else:
        accuracy = 0.0

    # Calculate correlation between predictions and actual outcomes.
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()

    if np.std(pred_flat) > 0 and np.std(target_flat) > 0:
        correlation = float(np.corrcoef(
            pred_flat, target_flat
        )[0, 1])
    else:
        correlation = 0.0

    return {
        "mse": mse,
        "accuracy": accuracy,
        "correlation": correlation,
        "prediction_mean": float(np.mean(predictions)),
        "prediction_std": float(np.std(predictions)),
    }

def train_network(
    x_train, y_train, x_val, y_val,
    seed: int, epochs: int, batch_size: int,
    lr: float, weight_decay: float,
):
    rng = np.random.default_rng(seed)
    p = initialize_network(rng)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(val) for k, val in p.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0
    history = []
    best_validation = float("inf")
    best_params = {k: value.copy() for k, value in p.items()}
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), batch_size):
            ids = order[start:start + batch_size]
            xb, yb = x_train[ids], y_train[ids]
            pred, cache = forward(xb, p)
            x, z1, a1, z2, a2, y = cache
            # MSE derivative, including the tanh output derivative.
            d = (2.0 / len(ids)) * (y - yb) * (1.0 - y * y)
            grads = {}
            grads["w3"] = a2.T @ d
            grads["b3"] = np.sum(d, axis=0, keepdims=True)
            da2 = d @ p["w3"].T
            dz2 = da2 * (z2 > 0)
            grads["w2"] = a1.T @ dz2
            grads["b2"] = np.sum(dz2, axis=0, keepdims=True)
            da1 = dz2 @ p["w2"].T
            dz1 = da1 * (z1 > 0)
            grads["w1"] = x.T @ dz1
            grads["b1"] = np.sum(dz1, axis=0, keepdims=True)

            # L2 regularization: penalize large weights.
            # Biases are not regularized.
            for key in ("w1", "w2", "w3"):
                grads[key] += 2.0 * weight_decay * p[key]

            step += 1
            for key in p:
                m[key] = beta1 * m[key] + (1 - beta1) * grads[key]
                v[key] = beta2 * v[key] + (1 - beta2) * (grads[key] ** 2)
                mhat = m[key] / (1 - beta1 ** step)
                vhat = v[key] / (1 - beta2 ** step)
                p[key] -= lr * mhat / (np.sqrt(vhat) + eps)

        
        train_pred, _ = forward(x_train, p)
        val_pred, _ = forward(x_val, p)

        train_metrics = evaluate_predictions(train_pred, y_train)
        val_metrics = evaluate_predictions(val_pred, y_val)

        train_loss = train_metrics["mse"]
        val_loss = val_metrics["mse"]

        history.append({
            "epoch": epoch,
            "train_mse": train_loss,
            "validation_mse": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "validation_accuracy": val_metrics["accuracy"],
            "train_correlation": train_metrics["correlation"],
            "validation_correlation": val_metrics["correlation"],
            "validation_prediction_mean": val_metrics["prediction_mean"],
            "validation_prediction_std": val_metrics["prediction_std"],
        })

        if val_loss < best_validation:
            best_validation = val_loss
            best_params = {k: value.copy() for k, value in p.items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"epoch {epoch:03d}: "
            f"train_mse={train_loss:.4f} "
            f"val_mse={val_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.3f} "
            f"val_corr={val_metrics['correlation']:.3f}",
            flush=True,
        )

        if epochs_without_improvement >= 12:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best validation MSE={best_validation:.4f}",
                flush=True,
            )
            break
    return best_params, history


def main():
    parser = argparse.ArgumentParser(description="Generate Santorini self-play data and train a value model.")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--search-time", type=float, default=0.20)
    parser.add_argument("--opponents", choices=("selfplay", "mixed"), default="selfplay")
    parser.add_argument("--max-turns", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--validation-games", type=int, default=400)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    args = parser.parse_args()
    if args.games < 10:
        parser.error("Use at least 10 training games.")
    if args.validation_games < 1:
        parser.error("Use at least 1 validation game.")
    if args.epochs < 1:
        parser.error("Use at least 1 training epoch.")
    if args.batch_size < 1:
        parser.error("Batch size must be at least 1.")
    if args.learning_rate <= 0:
        parser.error("Learning rate must be positive.")
    if args.weight_decay < 0:
        parser.error("Weight decay cannot be negative.")
    if args.search_time <= 0:
        parser.error("Search time must be positive.")
    if args.max_turns < 1:
        parser.error("Maximum turns must be at least 1.")
    random.seed(args.seed)
    random_bot._rng.seed(args.seed)

    # Generate training games.
    x_train, y_train, train_ids, train_reasons = collect_games(
        args.games,
        args.seed,
        args.search_time,
        args.max_turns,
        args.opponents,
        "Training",
    )
    
    # Reset random generators so validation data is independent
    # of how many training games were generated.
    random.seed(args.seed)
    random_bot._rng.seed(args.seed)

    # Generate a completely separate validation dataset.
    # The offset ensures validation games use different seeds.
    validation_seed = args.seed + 1_000_000

    x_val, y_val, val_ids, val_reasons = collect_games(
        args.validation_games,
        validation_seed,
        args.search_time,
        args.max_turns,
        args.opponents,
        "Validation",
    )

    print("\nTraining games:", dict(train_reasons))
    print("Validation games:", dict(val_reasons))

    print(f"\nTraining positions: {len(x_train)}")
    print(f"Validation positions: {len(x_val)}")

    if len(x_train) == 0 or len(x_val) == 0:
        raise RuntimeError("Training or validation dataset is empty.")

    # Combine datasets for saving, while retaining their game IDs.
    # Validation game IDs start after the training game IDs.
    all_x = np.concatenate((x_train, x_val))
    all_y = np.concatenate((y_train, y_val))

    all_game_ids = np.concatenate((
        train_ids,
        val_ids + args.games,
    ))

    validation_game_ids = np.arange(
        args.games,
        args.games + args.validation_games,
        dtype=np.int32,
    )

    data_out = ROOT / "results" / "training_data.npz"
    data_out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        data_out,
        features=all_x,
        targets=all_y,
        game_ids=all_game_ids,
        validation_game_ids=validation_game_ids,
    )

    print(f"Saved training examples: {data_out}")

    p, history = train_network(
        x_train, y_train, x_val, y_val,
        args.seed + 17,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
    )

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    weights_path, metadata_path = model_paths(model_dir)
    np.savez_compressed(weights_path, **p)
    metadata = {
        "feature_version": FEATURE_VERSION,
        "input_size": INPUT_SIZE,
        "architecture": [INPUT_SIZE, 64, 32, 1],
        "activation": "ReLU hidden layers; tanh output",
        "training_method": "supervised learning from game outcomes",
        "target_definition": "+1 if the player to move eventually wins; -1 if they lose; 0 for draw",
        "self_play_opponents": (
            "Phase 1 alpha-beta search bot versus itself"
            if args.opponents == "selfplay"
            else "Phase 1 alpha-beta search bot versus Phase 1 random bot, alternating seats"
        ),
        "opponent_mode": args.opponents,
        "games": args.games,
        "positions": len(all_x),
        "training_positions": len(x_train),
        "validation_positions": len(x_val),
        "validation_split": "Separate validation games generated using an offset seed",
        "validation_games": args.validation_games,
        "validation_seed": validation_seed,
        "weight_decay": args.weight_decay,        
        "seed": args.seed,
        "search_time_seconds": args.search_time,
        "max_turns": args.max_turns,
        "outcomes": {
            "training": dict(train_reasons),
            "validation": dict(val_reasons),
        },
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "last_epoch_train_mse": history[-1]["train_mse"],
        "last_epoch_validation_mse": history[-1]["validation_mse"],
        "best_validation_mse": min(item["validation_mse"] for item in history),
        "best_validation_epoch": min(history, key=lambda item: item["validation_mse"])["epoch"],
        "training_history": history,
        "epochs_trained": len(history),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved model: {weights_path}")
    print(f"Saved metadata and loss history: {metadata_path}")


if __name__ == "__main__":
    main()
