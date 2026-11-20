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

from santorini_engine import BoardState, apply_action, legal_actions_from
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


def _evaluate(board: BoardState, root_player: int, deadline: float) -> float:
    """Model predicts value for side-to-move; convert to root-player perspective."""
    _check_deadline(deadline)
    features = encode_state(board, board.current_player)
    _check_deadline(deadline)
    side_to_move_value = _model.predict(features)
    _check_deadline(deadline)
    return side_to_move_value if board.current_player == root_player else -side_to_move_value


def _ordered_children(board, actions, root_player, deadline, ply):
    ordered = []
    for action in actions:
        _check_deadline(deadline)
        child = apply_action(board, action)
        _check_deadline(deadline)
        if child.winner is None:
            score = _evaluate(child, root_player, deadline)
        else:
            win_score = WIN_SCORE - ply
            score = win_score if child.winner == root_player else -win_score
        ordered.append((score, action, child))
    _check_deadline(deadline)
    ordered.sort(key=lambda item: item[0], reverse=board.current_player == root_player)
    _check_deadline(deadline)
    return ordered


def _minimax(board, depth, alpha, beta, root_player, deadline, ply=0):
    _check_deadline(deadline)
    if board.winner is not None:
        return WIN_SCORE - ply if board.winner == root_player else -WIN_SCORE + ply
    if depth <= 0:
        return _evaluate(board, root_player, deadline)

    actions = legal_actions_from(board)
    _check_deadline(deadline)
    if not actions:
        return -WIN_SCORE + ply if board.current_player == root_player else WIN_SCORE - ply

    maximizing = board.current_player == root_player
    ordered = _ordered_children(board, actions, root_player, deadline, ply + 1)

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
    deadline = perf_counter() + max(0.0, time_limit_seconds * 0.88)
    if not legal_actions:
        raise ValueError("choose_action was called without legal actions")

    best_action = legal_actions[0]
    try:
        _check_deadline(deadline)
        board = _as_board(state)
        for action in legal_actions:
            _check_deadline(deadline)
            if board.heights[action.move_to.row][action.move_to.column] == 3:
                return action

        # The ordering scores are also the complete depth-one search.
        ordered = _ordered_children(board, legal_actions, 0, deadline, 1)
        best_action = ordered[0][1]
        for depth in (2, 3):
            _check_deadline(deadline)
            best_score = -inf
            depth_best = best_action
            alpha = -inf
            for _, action, child in ordered:
                score = _minimax(child, depth - 1, alpha, inf, 0, deadline, 1)
                if score > best_score:
                    best_score, depth_best = score, action
                alpha = max(alpha, best_score)
            _check_deadline(deadline)
            best_action = depth_best
    except SearchTimeout:
        pass
    return best_action
