"""Run reproducible paired Santorini bot matches and save raw results.

From the repository root:
  python run_experiment.py --games 50 --seed 20261028 --depth 2 --time-limit 1.5
The official harness enforces the per-move time limit and records full move logs.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import random_bot  # noqa: E402
import search_bot  # noqa: E402
from santorini_harness import MatchConfig, run_santorini_tournament  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--time-limit", type=float, default=1.5)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    if args.games < 1:
        parser.error("--games must be at least 1")

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    move_logs = []
    for i in range(args.games):
        seed = args.seed + i
        search_as_a = (i % 2 == 0)
        bot_a = search_bot.choose_action if search_as_a else random_bot.choose_action
        bot_b = random_bot.choose_action if search_as_a else search_bot.choose_action
        config = MatchConfig(move_time_limit_seconds=args.time_limit, max_turns=200, seed=seed)
        result = run_santorini_tournament(bot_a, bot_b, config=config, num_games=1)[0]
        if result.winner is None:
            winner_name = "draw"
        else:
            winner_is_search = (result.winner == 0) == search_as_a
            winner_name = "search" if winner_is_search else "random"
        row = {
            "game": i + 1,
            "seed": seed,
            "search_player": "A" if search_as_a else "B",
            "winner": winner_name,
            "winner_id": "draw" if result.winner is None else result.winner,
            "reason": result.reason.value,
            "turns_played": result.turns_played,
            "plies": len(result.moves),
            "detail": result.detail,
        }
        rows.append(row)
        move_logs.append({**row, "moves": [
            {"worker": m.worker, "move_to": [m.move_to.row, m.move_to.column],
             "build_at": [m.build_at.row, m.build_at.column]}
            for m in result.moves
        ]})
        print(f"Game {i+1:02d}/{args.games}: seed={seed} search_player={row['search_player']} "
              f"winner={winner_name} reason={row['reason']} turns={row['turns_played']}", flush=True)

    with (out_dir / "match_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "move_logs.json").write_text(json.dumps(move_logs, indent=2), encoding="utf-8")
    metadata = {
        "games_requested": args.games,
        "seed_start": args.seed,
        "seed_rule": "game i uses seed_start + (i - 1)",
        "player_assignment": "search is player A on odd-numbered games and B on even-numbered games",
        "time_limit_seconds": args.time_limit,
        "search_depth_setting": "iterative deepening depths 1-3; deepest completed iteration is used",
        "search_implementation": "iterative deepening alpha-beta, depths 1-3, returns last completed depth",
        "max_turns": 200,
        "python_version": sys.version,
    }
    (out_dir / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved {len(rows)} games to {out_dir}")
    print("Search wins:", sum(r["winner"] == "search" for r in rows))
    print("Random wins:", sum(r["winner"] == "random" for r in rows))
    print("Draws:", sum(r["winner"] == "draw" for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
