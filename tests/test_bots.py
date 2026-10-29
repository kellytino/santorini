from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import random_bot
import search_bot
from santorini_engine import legal_actions_from, setup
from santorini_types import Action, Coordinate


def public_state(board):
    from santorini_engine import public_state_for
    return public_state_for(board, board.current_player)


def test_random_bot_returns_supplied_legal_action():
    board = setup(123)
    actions = legal_actions_from(board)
    action = random_bot.choose_action(public_state(board), actions, 1.0)
    assert action in actions


def test_search_bot_returns_supplied_legal_action():
    board = setup(123)
    actions = legal_actions_from(board)
    action = search_bot.choose_action(public_state(board), actions, 0.5)
    assert action in actions


def test_search_bot_takes_immediate_level_three_win():
    # Worker 0 at (2, 1) can move to level 3 at (2, 2).
    heights = [[0] * 5 for _ in range(5)]
    heights[2][2] = 3
    from santorini_types import GameState
    state = GameState(
        heights=tuple(tuple(row) for row in heights),
        current_workers=(Coordinate(2, 1), Coordinate(0, 0)),
        opponent_workers=(Coordinate(4, 4), Coordinate(4, 3)),
        turn_number=10,
    )
    win_action = Action(0, Coordinate(2, 2), Coordinate(1, 2))
    action = search_bot.choose_action(state, (win_action,), 0.5)
    assert action == win_action
