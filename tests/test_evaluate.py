import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))

import evaluate
from santorini_engine import legal_actions_from, setup
from santorini_harness import GameResult, TerminationReason


def _args(out, games="4"):
    return [
        "--opponent", str(ROOT / "src" / "random_bot.py"),
        "--opponent-name", "Phase 1 random",
        "--games", games, "--seed", "42", "--time-limit", "1.5",
        "--out", str(out),
    ]


def test_one_hundred_assignments_balance_each_opening():
    assignments = evaluate._assignments(100, 42)
    assert len(assignments) == 100
    for index in range(50):
        first, second = assignments[2 * index:2 * index + 2]
        assert first["seed"] == second["seed"] == 42 + index
        assert (first["phase2_player"], second["phase2_player"]) == ("A", "B")


def test_evaluation_records_results_and_reseeds_actual_random_bot(tmp_path, monkeypatch):
    calls = []
    action = legal_actions_from(setup(42))[0]
    outcomes = iter([
        GameResult(0, 1, TerminationReason.WIN, 1, (action,)),
        GameResult(0, 1, TerminationReason.TIME_LIMIT, 0, (), "late"),
        GameResult(None, None, TerminationReason.TURN_LIMIT, 1, (action,)),
        GameResult(1, 0, TerminationReason.NO_LEGAL_ACTION, 1, (action,)),
    ])

    def play(bot_a, bot_b, config, num_games):
        assert num_games == 1
        assert config.move_time_limit_seconds == 1.5
        assert config.max_turns == 200
        random_seat = 0 if bot_a.__module__ == "_evaluation_opponent" else 1
        rng = (bot_a, bot_b)[random_seat].__globals__["_rng"]
        assert rng.getstate() == random.Random(config.seed).getstate()
        rng.random()  # Make the next game fail unless the actual RNG is reset.
        calls.append((config.seed, random_seat))
        return [next(outcomes)]

    monkeypatch.setattr(evaluate, "run_santorini_tournament", play)
    assert evaluate.main(_args(tmp_path)) == 0
    assert calls == [(42, 1), (42, 0), (43, 1), (43, 0)]

    with (tmp_path / "match_results.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["winner_bot"] for row in rows] == ["phase2", "opponent", "", "phase2"]
    assert rows[1]["detail"] == "late"
    logs = json.loads((tmp_path / "move_logs.json").read_text(encoding="utf-8"))
    assert [len(game["moves"]) for game in logs] == [1, 0, 1, 1]
    assert logs[0]["moves"][0] == {
        "worker": action.worker,
        "move_to": [action.move_to.row, action.move_to.column],
        "build_at": [action.build_at.row, action.build_at.column],
    }

    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config["assignments"] == evaluate._assignments(4, 42)
    bots = config["bots"]
    for record in (bots["opponent"], bots["phase2"], bots["phase2"]["evaluator"], bots["phase2"]["model"]):
        assert record["sha256"] == hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["average_game_length_plies"] == 0.75
    phase2, opponent = (summary["bots"][key] for key in ("phase2", "opponent"))
    assert (phase2["starts"], phase2["wins"], phase2["losses"], phase2["draws"], phase2["forfeits"]) == (2, 2, 1, 1, 1)
    assert phase2["forfeits_by_reason"]["time_limit"] == 1
    assert opponent["forfeits"] == 0


@pytest.mark.parametrize("games", ["0", "-2", "3"])
def test_rejects_invalid_game_count(tmp_path, games):
    with pytest.raises(SystemExit) as error:
        evaluate.main(_args(tmp_path, games))
    assert error.value.code == 2
    assert not list(tmp_path.iterdir())


def test_preserves_existing_results(tmp_path):
    existing = tmp_path / "match_results.csv"
    existing.write_text("original", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        evaluate.main(_args(tmp_path))
    assert error.value.code == 2
    assert existing.read_text(encoding="utf-8") == "original"
