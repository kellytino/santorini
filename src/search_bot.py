"""Depth-limited minimax with alpha-beta pruning for base-game Santorini.

The bot uses the official engine's successors() to generate legal complete turns.
All scores are from the perspective of the player who receives choose_action().
"""
from __future__ import annotations

from math import inf
from time import perf_counter

from santorini_engine import BoardState, successors
from santorini_types import Action, Coordinate, GameState

WIN_SCORE = 100_000
HEIGHT_WEIGHT = 3
MOBILITY_WEIGHT = 0.08
LEVEL3_ACCESS_WEIGHT = 12


class SearchTimeout(Exception):
    """Raised internally when the move's time budget is nearly exhausted."""


def _check_deadline(deadline: float) -> None:
    if perf_counter() >= deadline:
        raise SearchTimeout


def _as_board(state: GameState) -> BoardState:
    """Map the public perspective-relative state to a canonical search state."""
    return BoardState(
        heights=state.heights,
        workers=(state.current_workers, state.opponent_workers),
        current_player=0,
        turn_number=state.turn_number,
        winner=None,
    )


def _worker_height(board: BoardState, player: int) -> int:
    return max(board.heights[w.row][w.column] for w in board.workers[player])


def _mobility(board: BoardState, player: int) -> int:
    """Count legal move destinations, not complete build combinations."""
    occupied = {
        (worker.row, worker.column)
        for workers in board.workers
        for worker in workers
    }
    count = 0
    for worker in board.workers[player]:
        source_height = board.heights[worker.row][worker.column]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                row, col = worker.row + dr, worker.column + dc
                if not (0 <= row < 5 and 0 <= col < 5):
                    continue
                if (row, col) in occupied:
                    continue
                target_height = board.heights[row][col]
                if target_height < 4 and target_height <= source_height + 1:
                    count += 1
    return count


def _level3_access(board: BoardState, player: int) -> int:
    """Count level-3 squares that a worker could move onto next turn."""
    occupied = {
        (worker.row, worker.column)
        for workers in board.workers
        for worker in workers
    }
    count = 0
    for worker in board.workers[player]:
        source_height = board.heights[worker.row][worker.column]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                row, col = worker.row + dr, worker.column + dc
                if 0 <= row < 5 and 0 <= col < 5 and (row, col) not in occupied:
                    if board.heights[row][col] == 3 and 3 <= source_height + 1:
                        count += 1
    return count


def evaluate(board: BoardState, root_player: int = 0) -> float:
    """Hand-coded positional evaluation; terminal outcomes are handled by search."""
    opponent = 1 - root_player
    height_diff = _worker_height(board, root_player) - _worker_height(board, opponent)
    mobility_diff = _mobility(board, root_player) - _mobility(board, opponent)
    access_diff = _level3_access(board, root_player) - _level3_access(board, opponent)
    return (
        HEIGHT_WEIGHT * height_diff
        + MOBILITY_WEIGHT * mobility_diff
        + LEVEL3_ACCESS_WEIGHT * access_diff
    )


def _minimax(
    board: BoardState,
    depth: int,
    alpha: float,
    beta: float,
    root_player: int,
    deadline: float,
    ply: int = 0,
) -> float:
    _check_deadline(deadline)
    if board.winner is not None:
        return (WIN_SCORE - ply) if board.winner == root_player else (-WIN_SCORE + ply)

    if depth <= 0:
        return evaluate(board, root_player)

    pairs = successors(board)
    if not pairs:
        # No legal move means the player to move loses.
        return (-WIN_SCORE + ply) if board.current_player == root_player else (WIN_SCORE - ply)

    maximizing = board.current_player == root_player
    # Mild move ordering: evaluate children first, favoring immediate wins and
    # promising positions to improve alpha-beta pruning.
    scored = []
    for action, child in pairs:
        _check_deadline(deadline)
        if child.winner is not None:
            order_score = WIN_SCORE if child.winner == root_player else -WIN_SCORE
        else:
            order_score = evaluate(child, root_player)
        scored.append((order_score, action, child))
    scored.sort(key=lambda item: item[0], reverse=maximizing)

    if maximizing:
        value = -inf
        for _, _, child in scored:
            value = max(value, _minimax(child, depth - 1, alpha, beta, root_player, deadline, ply + 1))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value

    value = inf
    for _, _, child in scored:
        value = min(value, _minimax(child, depth - 1, alpha, beta, root_player, deadline, ply + 1))
        beta = min(beta, value)
        if alpha >= beta:
            break
    return value


def choose_action(
    state: GameState,
    legal_actions: tuple[Action, ...],
    time_limit_seconds: float,
) -> Action:
    """Choose an action using iterative deepening alpha-beta search."""
    if not legal_actions:
        raise ValueError("choose_action was called without legal actions")

    board = _as_board(state)
    # Keep a small safety margin for returning through the harness.
    budget = max(0.01, time_limit_seconds * 0.88)
    deadline = perf_counter() + budget
    best_action = legal_actions[0]

    # Immediate wins should always be selected without searching further.
    for action in legal_actions:
        if board.heights[action.move_to.row][action.move_to.column] == 3:
            return action

    # Iterative deepening retains the last fully completed depth if time expires.
    for depth in (1, 2, 3):
        try:
            _check_deadline(deadline)
            best_score = -inf
            depth_best = best_action
            child_by_action = dict(successors(board))
            ordered = []
            for action in legal_actions:
                child = child_by_action[action]
                score = evaluate(child, 0) if child.winner is None else WIN_SCORE
                ordered.append((score, action, child))
            ordered.sort(key=lambda item: item[0], reverse=True)
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


def next_state_for_action(board: BoardState, action: Action) -> BoardState:
    """Use the official successor generator to retrieve the child for action."""
    for candidate, child in successors(board):
        if candidate == action:
            return child
    raise ValueError("Action was not legal in this board state")
