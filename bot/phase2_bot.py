"""Phase 2 Santorini bot: iterative-deepening alpha-beta + learned value model."""
from __future__ import annotations

from math import inf
from pathlib import Path
import sys
from time import perf_counter

BOT_DIR = Path(__file__).resolve().parent
ROOT = BOT_DIR.parent
sys.path.insert(0, str(BOT_DIR))
sys.path.insert(0, str(ROOT / "src"))

from santorini_engine import BoardState, successors
from santorini_types import Action, GameState
from phase2_common import ValueModel, encode_state

WIN_SCORE = 100_000
MODEL_DIR = ROOT / "models"
_model = ValueModel(MODEL_DIR)


class SearchTimeout(Exception):
    pass


def _check_deadline(deadline: float) -> None:
    if perf_counter() >= deadline:
        raise SearchTimeout


def _as_board(state: GameState) -> BoardState:
    # Player 0 in this private search tree is always the bot that was called.
    return BoardState(
        heights=state.heights,
        workers=(state.current_workers, state.opponent_workers),
        current_player=0,
        turn_number=state.turn_number,
        winner=None,
    )


def _evaluate(board: BoardState, root_player: int) -> float:
    """Model predicts value for side-to-move; convert to root-player perspective."""
    side_to_move_value = _model.predict(encode_state(board, board.current_player))
    return side_to_move_value if board.current_player == root_player else -side_to_move_value


def _minimax(board, depth, alpha, beta, root_player, deadline, ply=0):
    _check_deadline(deadline)
    if board.winner is not None:
        return WIN_SCORE - ply if board.winner == root_player else -WIN_SCORE + ply
    if depth <= 0:
        return _evaluate(board, root_player)

    children = successors(board)
    if not children:
        return -WIN_SCORE + ply if board.current_player == root_player else WIN_SCORE - ply

    maximizing = board.current_player == root_player
    ordered = []
    for action, child in children:
        _check_deadline(deadline)
        score = (WIN_SCORE - ply - 1) if child.winner == root_player else (
            -WIN_SCORE + ply + 1 if child.winner is not None else _evaluate(child, root_player)
        )
        ordered.append((score, action, child))
    ordered.sort(key=lambda item: item[0], reverse=maximizing)

    if maximizing:
        value = -inf
        for _, _, child in ordered:
            value = max(value, _minimax(child, depth - 1, alpha, beta, root_player, deadline, ply + 1))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
    else:
        value = inf
        for _, _, child in ordered:
            value = min(value, _minimax(child, depth - 1, alpha, beta, root_player, deadline, ply + 1))
            beta = min(beta, value)
            if alpha >= beta:
                break
    return value


def choose_action(state: GameState, legal_actions: tuple[Action, ...], time_limit_seconds: float) -> Action:
    if not legal_actions:
        raise ValueError("choose_action was called without legal actions")

    board = _as_board(state)
    deadline = perf_counter() + max(0.01, time_limit_seconds * 0.88)
    # Never overlook an immediate winning move.
    for action in legal_actions:
        if board.heights[action.move_to.row][action.move_to.column] == 3:
            return action

    best_action = legal_actions[0]
    child_by_action = dict(successors(board))
    for depth in (1, 2, 3):
        try:
            _check_deadline(deadline)
            ordered = []
            for action in legal_actions:
                child = child_by_action[action]
                score = WIN_SCORE if child.winner == 0 else _evaluate(child, 0)
                ordered.append((score, action, child))
            ordered.sort(key=lambda item: item[0], reverse=True)

            best_score = -inf
            depth_best = best_action
            alpha = -inf
            for _, action, child in ordered:
                score = _minimax(child, depth - 1, alpha, inf, 0, deadline, 1)
                if score > best_score:
                    best_score, depth_best = score, action
                alpha = max(alpha, best_score)
            best_action = depth_best
        except SearchTimeout:
            break
    return best_action
