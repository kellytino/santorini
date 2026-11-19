from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bot"))

import numpy as np
from santorini_engine import legal_actions_from, setup, public_state_for
from phase2_common import INPUT_SIZE, encode_state


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
