"""Uniform random legal-action baseline for the Santorini course harness."""
import random
from santorini_types import Action, GameState

_rng = random.Random()


def choose_action(
    state: GameState,
    legal_actions: tuple[Action, ...],
    time_limit_seconds: float,
) -> Action:
    """Choose one complete action uniformly from the harness-provided legal set."""
    if not legal_actions:
        raise ValueError("choose_action was called without legal actions")
    return _rng.choice(legal_actions)
