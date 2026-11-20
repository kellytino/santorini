from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bot"))

import numpy as np
import pytest
from santorini_engine import legal_actions_from, setup, public_state_for
from phase2_common import INPUT_SIZE, encode_state
import phase2_bot


def test_phase2_features_have_expected_size_and_range():
    board = setup(2026)
    features = encode_state(board, board.current_player)
    assert features.shape == (INPUT_SIZE,)
    assert np.isfinite(features).all()
    assert features.min() >= -1.0
    assert features.max() <= 1.0


def test_phase2_bot_returns_a_supplied_legal_action():
    # This test requires the trained model to exist in ../models.
    from phase2_bot import choose_action
    board = setup(2026)
    actions = legal_actions_from(board)
    state = public_state_for(board, board.current_player)
    action = choose_action(state, actions, 0.5)
    assert action in actions


@pytest.fixture
def clock(monkeypatch):
    class Clock:
        now = 0.0

    clock = Clock()
    monkeypatch.setattr(phase2_bot, "perf_counter", lambda: clock.now)
    return clock


@pytest.fixture
def opening():
    board = setup(2026)
    return board, public_state_for(board, board.current_player), legal_actions_from(board)[:2]


def test_expired_budget_skips_board_setup(monkeypatch, clock, opening):
    _, state, actions = opening
    monkeypatch.setattr(phase2_bot, "_as_board", lambda *_: pytest.fail("started after deadline"))
    assert phase2_bot.choose_action(state, actions, 0.0) == actions[0]


def test_root_stops_after_slow_inference(monkeypatch, clock, opening):
    _, state, actions = opening
    applied = []
    apply_action = phase2_bot.apply_action

    def track_apply(board, action):
        applied.append(action)
        return apply_action(board, action)

    def predict(_):
        clock.now = 0.002
        return 0.0

    monkeypatch.setattr(phase2_bot, "apply_action", track_apply)
    monkeypatch.setattr(phase2_bot._model, "predict", predict)
    assert phase2_bot.choose_action(state, actions, 0.001) == actions[0]
    assert applied == [actions[0]]


def test_expiration_during_encoding_skips_prediction(monkeypatch, clock, opening):
    board, _, _ = opening

    def encode(*_):
        clock.now = 1.0

    monkeypatch.setattr(phase2_bot, "encode_state", encode)
    monkeypatch.setattr(phase2_bot._model, "predict", lambda *_: pytest.fail("predicted after deadline"))
    with pytest.raises(phase2_bot.SearchTimeout):
        phase2_bot._minimax(board, 0, -float("inf"), float("inf"), 0, 1.0)


@pytest.mark.parametrize("operation", ["legal_actions_from", "apply_action"])
def test_recursive_expansion_stops_at_deadline(monkeypatch, clock, opening, operation):
    board, _, _ = opening
    original = getattr(phase2_bot, operation)
    calls = []

    def slow_call(*args):
        calls.append(args)
        clock.now = 1.0
        return original(*args)

    monkeypatch.setattr(phase2_bot, operation, slow_call)
    if operation == "legal_actions_from":
        monkeypatch.setattr(phase2_bot, "apply_action", lambda *_: pytest.fail("expanded after deadline"))
    monkeypatch.setattr(phase2_bot._model, "predict", lambda *_: pytest.fail("predicted after deadline"))
    with pytest.raises(phase2_bot.SearchTimeout):
        phase2_bot._minimax(board, 1, -float("inf"), float("inf"), 0, 1.0)
    assert len(calls) == 1


def test_late_final_score_keeps_last_completed_depth(monkeypatch, clock, opening):
    _, state, actions = opening
    predictions = iter((0.5, -0.5))  # Negated at the root, so depth one prefers actions[1].
    monkeypatch.setattr(phase2_bot._model, "predict", lambda _: next(predictions))
    scores = iter((0.0, 1.0))  # Depth two would prefer actions[0] if it completed in time.

    def minimax(board, depth, alpha, beta, root_player, deadline, ply):
        assert depth == 1
        score = next(scores)
        if score == 1.0:
            clock.now = deadline
        return score

    monkeypatch.setattr(phase2_bot, "_minimax", minimax)
    assert phase2_bot.choose_action(state, actions, 1.0) == actions[1]
