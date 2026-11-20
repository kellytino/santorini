"""Evaluate the learned Phase 2 bot in paired games using the official harness."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import random
import sys
from math import isfinite
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from santorini_harness import MatchConfig, TerminationReason, run_santorini_tournament

PHASE2 = Path(__file__).with_name("phase2_bot.py").resolve()
MAX_TURNS = 200
FORFEIT_REASONS = (
    TerminationReason.TIME_LIMIT.value,
    TerminationReason.INVALID_ACTION.value,
    TerminationReason.BOT_EXCEPTION.value,
)


def _load_bot(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load a Python bot from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    if not callable(getattr(module, "choose_action", None)):
        raise ValueError(f"{path} must define a callable choose_action")
    return module


def _file_record(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _assignments(games, seed):
    return [
        {"game": index + 1, "seed": seed + index // 2, "phase2_player": "A" if index % 2 == 0 else "B"}
        for index in range(games)
    ]


def _summarize(rows, names):
    summary = {
        "games": len(rows),
        "draws": sum(row["winner_id"] is None for row in rows),
        "average_game_length_plies": mean(row["plies"] for row in rows),
        "bots": {},
    }
    for bot, name in names.items():
        wins = sum(row["winner_bot"] == bot for row in rows)
        by_reason = {
            reason: sum(row["reason"] == reason and row["loser_bot"] == bot for row in rows)
            for reason in FORFEIT_REASONS
        }
        summary["bots"][bot] = {
            "name": name,
            "starts": sum(row["starting_bot"] == bot for row in rows),
            "wins": wins,
            "losses": len(rows) - wins - summary["draws"],
            "draws": summary["draws"],
            "forfeits": sum(by_reason.values()),
            "forfeits_by_reason": by_reason,
        }
    return summary


def _write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opponent", type=Path, required=True, help="Opponent Python file")
    parser.add_argument("--opponent-name", required=True)
    parser.add_argument("--games", type=int, required=True, help="Positive even number of games")
    parser.add_argument("--seed", type=int, required=True, help="First paired opening seed")
    parser.add_argument("--time-limit", type=float, required=True, help="Seconds per move")
    parser.add_argument("--out", type=Path, required=True, help="New or empty output directory")
    args = parser.parse_args(argv)
    if args.games <= 0 or args.games % 2:
        parser.error("--games must be a positive even number")
    if not isfinite(args.time_limit) or args.time_limit <= 0:
        parser.error("--time-limit must be positive and finite")

    out = args.out.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error(f"--out must be a new or empty directory: {out}")
    opponent_path = args.opponent.resolve()
    try:
        phase2 = _load_bot(PHASE2, "_evaluation_phase2")
        opponent = _load_bot(opponent_path, "_evaluation_opponent")
        names = {"phase2": "Phase 2", "opponent": args.opponent_name}
        assignments = _assignments(args.games, args.seed)
        opponent_rng = getattr(opponent, "_rng", None)
        seed_opponent = isinstance(opponent_rng, random.Random)
        config = {
            "games": args.games,
            "seed_start": args.seed,
            "assignments": assignments,
            "time_limit_seconds": args.time_limit,
            "max_turns": MAX_TURNS,
            "python_version": sys.version,
            "bots": {
                "phase2": {
                    "name": names["phase2"],
                    **_file_record(PHASE2),
                    "evaluator": _file_record(PHASE2.with_name("phase2_common.py")),
                    "model": _file_record(phase2.MODEL_DIR / "santorini_value_model.npz"),
                },
                "opponent": {
                    "name": names["opponent"],
                    **_file_record(opponent_path),
                    "rng_seed": "each game's seed, applied to module._rng before play" if seed_opponent else None,
                },
            },
        }
        out.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    bot_functions = {"phase2": phase2.choose_action, "opponent": opponent.choose_action}
    rows = []
    move_logs = []
    for assignment in assignments:
        seed = assignment["seed"]
        if seed_opponent:
            opponent_rng.seed(seed)
        seats = ("phase2", "opponent") if assignment["phase2_player"] == "A" else ("opponent", "phase2")
        match = MatchConfig(move_time_limit_seconds=args.time_limit, max_turns=MAX_TURNS, seed=seed)
        result, = run_santorini_tournament(*(bot_functions[bot] for bot in seats), config=match, num_games=1)
        winner_bot = None if result.winner is None else seats[result.winner]
        loser_bot = None if result.loser is None else seats[result.loser]
        row = {
            **assignment,
            "starting_bot": seats[0],
            "winner": "draw" if winner_bot is None else names[winner_bot],
            "winner_bot": winner_bot,
            "winner_id": result.winner,
            "loser_bot": loser_bot,
            "reason": result.reason.value,
            "turns_played": result.turns_played,
            "plies": len(result.moves),
            "detail": result.detail,
        }
        rows.append(row)
        move_logs.append({**row, "moves": [
            {"worker": move.worker, "move_to": [move.move_to.row, move.move_to.column],
             "build_at": [move.build_at.row, move.build_at.column]}
            for move in result.moves
        ]})
        print(
            f"Game {row['game']}/{args.games} seed={seed} Phase 2={row['phase2_player']} "
            f"winner={row['winner']} reason={row['reason']} plies={row['plies']}",
            flush=True,
        )

    summary = _summarize(rows, names)
    with (out / "match_results.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_json(out / "move_logs.json", move_logs)
    _write_json(out / "config.json", config)
    _write_json(out / "summary.json", summary)
    for bot in summary["bots"].values():
        print(f"{bot['name']} — {bot['wins']} wins, {bot['losses']} losses, "
              f"{bot['draws']} draws, {bot['forfeits']} forfeits")
    print(f"Average game length {summary['average_game_length_plies']:.2f} plies")
    print(f"Saved results to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
