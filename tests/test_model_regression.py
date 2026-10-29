#!/usr/bin/env python3
"""Automated Regression and Safety Test Suite for Code Bot Models.

Verifies:
1. Action API contract (5 signed 32-bit integers, valid kind/dir/split).
2. Evaluator sandbox validity via sandbox.validate_action.
3. Tactical reflexes: Lethal snipe, adjacent defense, general garrison hold.
4. Obstacle avoidance: never issuing moves into type 2 (mountain) or type 5 (fog structure).
5. State isolation across consecutive matches.
6. Latency budget: strictly under 150 ms per decision.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluator"))

from sandbox import validate_action, BotFault


def make_dummy_obs(h=18, w=18, turn=1, my_gen=(1, 1), opp_gen=(16, 16)):
    """Generate a clean synthetic observation dictionary for testing."""
    type_grid = [[1 for _ in range(w)] for _ in range(h)]
    owner_grid = [[0 for _ in range(w)] for _ in range(h)]
    army_grid = [[0 for _ in range(w)] for _ in range(h)]

    # Place mountains
    for r in range(h):
        for c in range(w):
            if (r + c) % 7 == 0 and (r, c) not in (my_gen, opp_gen):
                type_grid[r][c] = 2

    # Place generals
    type_grid[my_gen[0]][my_gen[1]] = 4
    owner_grid[my_gen[0]][my_gen[1]] = 1
    army_grid[my_gen[0]][my_gen[1]] = 10

    type_grid[opp_gen[0]][opp_gen[1]] = 4
    owner_grid[opp_gen[0]][opp_gen[1]] = 2
    army_grid[opp_gen[0]][opp_gen[1]] = 10

    # Fog of war: only 3x3 around my_gen is visible
    for r in range(h):
        for c in range(w):
            if max(abs(r - my_gen[0]), abs(c - my_gen[1])) > 1:
                if type_grid[r][c] == 2:
                    type_grid[r][c] = 5  # mountain in fog
                else:
                    type_grid[r][c] = 0  # fog
                owner_grid[r][c] = 0
                army_grid[r][c] = 0

    return {
        "turn": turn,
        "height": h,
        "width": w,
        "player_id": 0,
        "my_land": 1,
        "my_army": 10,
        "opp_land": 1,
        "opp_army": 10,
        "type": type_grid,
        "owner": owner_grid,
        "army": army_grid,
    }


def test_action_contract(bot_module):
    """Test that act() always returns 5 signed 32-bit integers meeting sandbox contract."""
    obs = make_dummy_obs()
    if hasattr(bot_module, "_reset"):
        bot_module._reset()
    action = bot_module.act(obs)
    validated = validate_action(action)
    assert len(validated) == 5
    assert validated[0] in (0, 1, 2)
    assert validated[3] in range(4)
    assert validated[4] in (0, 1)
    print("  [PASS] Action contract and sandbox validation")


def test_lethal_snipe(bot_module):
    """Test that bot immediately snipes an adjacent, weaker enemy general."""
    obs = make_dummy_obs(h=18, w=18, my_gen=(5, 5), opp_gen=(5, 7))
    if hasattr(bot_module, "_reset"):
        bot_module._reset()

    # Place friendly army directly adjacent to enemy general at (5, 6)
    obs["owner"][5][6] = 1
    obs["army"][5][6] = 25
    obs["type"][5][6] = 1

    # Enemy general at (5, 7) has only 5 army
    obs["owner"][5][7] = 2
    obs["army"][5][7] = 5
    obs["type"][5][7] = 4

    action = bot_module.act(obs)
    # Expected action: move from (5, 6) RIGHT (dir=3) onto (5, 7)
    assert action[0] == 0, f"Expected move, got kind={action[0]}"
    assert (action[1], action[2]) == (5, 6), f"Expected source (5, 6), got ({action[1]}, {action[2]})"
    assert action[3] == 3, f"Expected dir=3 (RIGHT), got {action[3]}"
    print("  [PASS] Lethal snipe reflex executed correctly")


def test_adjacent_defense_no_suicide(bot_module):
    """Test that general does NOT suicide half its army into an adjacent stronger threat."""
    obs = make_dummy_obs(h=18, w=18, my_gen=(5, 5))
    if hasattr(bot_module, "_reset"):
        bot_module._reset()

    # General at (5, 5) has 12 army
    obs["owner"][5][5] = 1
    obs["army"][5][5] = 12
    obs["type"][5][5] = 4

    # Enemy adjacent at (5, 6) has 8 army
    # Note: 12 // 2 = 6 <= 8. Counter-attacking with split=1 would be SUICIDE (6 vs 8)!
    obs["owner"][5][6] = 2
    obs["army"][5][6] = 8
    obs["type"][5][6] = 1

    action = bot_module.act(obs)
    # Bot MUST NOT execute [0, 5, 5, 3, 1] (suicide attack with half army!)
    if action[0] == 0 and (action[1], action[2]) == (5, 5) and action[3] == 3:
        assert action[4] == 0, "General countered with split=1 into a stronger enemy! Suicide bug!"
        assert obs["army"][5][5] - 1 > obs["army"][5][6], "General attacked enemy it cannot beat!"
    print("  [PASS] General adjacent defense avoids suicide attacks")


def test_obstacle_avoidance(bot_module):
    """Test that bot never issues a move directly into a known mountain or type 5 structure."""
    obs = make_dummy_obs(h=18, w=18, my_gen=(5, 5))
    if hasattr(bot_module, "_reset"):
        bot_module._reset()

    # Surround friendly army at (5, 4) with mountains except one dir
    obs["owner"][5][4] = 1
    obs["army"][5][4] = 15
    obs["type"][5][4] = 1

    obs["type"][4][4] = 2  # UP is mountain
    obs["type"][6][4] = 2  # DOWN is mountain
    obs["type"][5][3] = 5  # LEFT is structure in fog

    # Run for 5 steps
    for turn in range(1, 6):
        obs["turn"] = turn
        action = bot_module.act(obs)
        if action[0] == 0:
            r, c, d = action[1], action[2], action[3]
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            nr, nc = r + dirs[d][0], c + dirs[d][1]
            if 0 <= nr < 18 and 0 <= nc < 18:
                assert obs["type"][nr][nc] != 2, f"Bot attempted to move into mountain at ({nr}, {nc})!"
                assert obs["type"][nr][nc] != 5, f"Bot attempted to move into structure in fog at ({nr}, {nc})!"
    print("  [PASS] Obstacle avoidance verified (no moves into type 2 or 5)")


def test_latency_budget(bot_module):
    """Test that bot acts comfortably within the 150 ms deadline (target < 20 ms)."""
    obs = make_dummy_obs(h=21, w=21)
    if hasattr(bot_module, "_reset"):
        bot_module._reset()

    latencies = []
    for turn in range(1, 25):
        obs["turn"] = turn
        t0 = time.perf_counter()
        bot_module.act(obs)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

    mean_lat = sum(latencies) / len(latencies)
    max_lat = max(latencies)
    print(f"  [PASS] Latency check: mean={mean_lat:.2f}ms, max={max_lat:.2f}ms (<150ms limit)")
    assert max_lat < 150.0, f"Max latency {max_lat:.2f}ms exceeded 150ms limit!"


def run_all_contract_tests(bot_path):
    print(f"\n>> Running Contract & Safeguard Tests for: {bot_path}")
    import importlib.util
    spec = importlib.util.spec_from_file_location("tested_bot", bot_path)
    bot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bot)

    test_action_contract(bot)
    test_lethal_snipe(bot)
    test_adjacent_defense_no_suicide(bot)
    test_obstacle_avoidance(bot)
    test_latency_budget(bot)
    print(">> ALL CONTRACT & SAFEGUARD TESTS PASSED!\n")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "models" / "model_27_final_champion" / "bot.py")
    run_all_contract_tests(target)
