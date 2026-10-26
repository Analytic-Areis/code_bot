#!/usr/bin/env python3
"""Unified Test Suite & Local Match Runner for all 22 Code Bot Models.

Features:
1. Automated execution of all 22 model tester suites.
2. Official evaluator static validation across all 22 bot.py files.
3. Pure Python 1v1 Match Simulator (runs head-to-head matches without Docker/JAX).
4. CLI options for batch verification or custom 1v1 matches.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

MODELS = [
    # 16 Game Theory Models from models.md
    ("model_01_bayesian_search", "Bayesian Target Acquisition"),
    ("model_02_pcst_gathering", "PCST Army Aggregation"),
    ("model_03_colonel_blotto", "Asymmetric Colonel Blotto"),
    ("model_04_stackelberg_expansion", "Stackelberg Opening Expansion"),
    ("model_05_minimax_castle", "Minimax Castle Facility Location"),
    ("model_06_two_ring_interdiction", "Two-Ring Network Interdiction"),
    ("model_07_war_of_attrition", "War of Attrition & 1200 Draw"),
    ("model_08_fog_signaling", "Fog-of-War Signaling Game"),
    ("model_09_mab_exploration", "UCB1 Multi-Armed Bandit"),
    ("model_10_nash_bargaining", "Nash Bargaining Resource Split"),
    ("model_11_differential_game", "Continuous Differential Game"),
    ("model_12_frontier_minimax", "Frontier Tactical Minimax"),
    ("model_13_opponent_classifier", "Bayesian Opponent Classifier"),
    ("model_14_pomdp_ghost_tracking", "POMDP Ghost Tracking"),
    ("model_15_fst_metagame", "Mealy FST Metagame Automaton"),
    ("model_16_info_revelation", "Value of Information Prober"),
    ("model_17_proposed_cascade", "Proposed Grandmaster Priority Cascade"),

    # 5 Models proposed & verified from '4 Never test against reference.txt'
    ("model_18_codebot_v0_known_good", "CodeBot v0 Known Good (Official Baseline)"),
    ("model_19_codebot_v1_probe_stage", "CodeBot v1 Probe & Stage (Dual Deathtouch)"),
    ("model_20_kg_rush", "KG Hyper-Aggressive Rusher"),
    ("model_21_kg_fortress", "KG Economic Fortress"),
    ("model_22_pinner", "The Pinner Exploit (Danger Lock)"),
    ("model_23_rl_champion", "RL Evolved Champion"),
    ("model_25_grandmaster_champion", "Grandmaster Tournament Champion"),
    ("model_26_apex_predator", "Apex Predator (IDS + Neural Net + Potential Fields)"),
    ("model_27_final_champion", "Final Champion (Beam Search + Calibrated Neural Head + DAG Funnel)"),
]

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def _model_path(folder, filename):
    p = ROOT / "models" / folder / filename
    if p.exists():
        return p
    return ROOT / folder / filename



def run_unit_tests():
    print("=" * 75)
    print(f"  RUNNING UNIT & BENCHMARK TESTS FOR ALL {len(MODELS)} MODELS")
    print("=" * 75)
    passed = 0
    failed = 0

    for folder, title in MODELS:
        tester_path = _model_path(folder, "tester.py")
        if not tester_path.exists():
            print(f"\n>> Skipping unit test for {folder} (uses official evaluator directly)")
            continue
        print(f"\n>> Running {folder} ({title})...")
        res = subprocess.run([sys.executable, str(tester_path)], capture_output=True, text=True)
        if res.returncode == 0:
            print(f"   [OK] {folder} PASSED")
            passed += 1
        else:
            print(f"   [FAIL] {folder} FAILED:")
            print(res.stderr or res.stdout)
            failed += 1

    print("\n" + "=" * 75)
    print(f"  UNIT TEST RESULTS: {passed} PASSED, {failed} FAILED (Total: {len(MODELS)})")
    print("=" * 75)
    return failed == 0


def run_evaluator_validation():
    print("\n" + "=" * 75)
    print("  RUNNING OFFICIAL EVALUATOR VALIDATION (evaluator/evaluate.py validate)")
    print("=" * 75)
    eval_script = ROOT / "evaluator" / "evaluate.py"
    all_bots = [str(_model_path(folder, "bot.py")) for folder, _ in MODELS]

    cmd = [sys.executable, str(eval_script), "validate"] + all_bots
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        lines = [l for l in res.stdout.strip().splitlines() if l]
        print(f">> Successfully validated {len(lines)} bot files:")
        for line in lines:
            try:
                data = json.loads(line)
                fname = Path(data["file"]).parent.name + "/bot.py"
                print(f"   [VALIDATED] {fname:<38} | {data['bytes']:>6} bytes | SHA: {data['sha256'][:12]}...")
            except Exception:
                print(f"   {line}")
        print(f">> ALL {len(MODELS)} BOTS PASSED OFFICIAL EVALUATOR VALIDATION!\n")
        return True
    else:
        print(">> EVALUATOR VALIDATION FAILED:")
        print(res.stderr or res.stdout)
        return False


def load_bot_module(bot_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, bot_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, "_reset"):
        mod._reset()
    return mod


class LocalGeneralsSim:
    def __init__(self, h=18, w=18, seed=42):
        self.h, self.w = h, w
        self.rng = random.Random(seed)
        self.turn = 1
        self.terrain = [[1 for _ in range(w)] for _ in range(h)]
        self.owner = [[0 for _ in range(w)] for _ in range(h)]
        self.army = [[0 for _ in range(w)] for _ in range(h)]

        # Spawn obstacles (mountains)
        for r in range(h):
            for c in range(w):
                if self.rng.random() < 0.15:
                    self.terrain[r][c] = 2  # Mountain

        # Generals (player 0 top-left, player 1 bottom-right)
        self.gen0 = (1, 1)
        self.gen1 = (h - 2, w - 2)
        self.terrain[self.gen0[0]][self.gen0[1]] = 4
        self.owner[self.gen0[0]][self.gen0[1]] = 1
        self.army[self.gen0[0]][self.gen0[1]] = 1

        self.terrain[self.gen1[0]][self.gen1[1]] = 4
        self.owner[self.gen1[0]][self.gen1[1]] = 2
        self.army[self.gen1[0]][self.gen1[1]] = 1

    def make_observation(self, player):
        my_code = 1 if player == 0 else 2
        opp_code = 2 if player == 0 else 1

        my_land = sum(row.count(my_code) for row in self.owner)
        opp_land = sum(row.count(opp_code) for row in self.owner)

        my_army = 0
        opp_army = 0
        for r in range(self.h):
            for c in range(self.w):
                if self.owner[r][c] == my_code:
                    my_army += self.army[r][c]
                elif self.owner[r][c] == opp_code:
                    opp_army += self.army[r][c]

        # Vision: 1-tile orthogonal/diagonal around owned cells
        type_obs = [[0 for _ in range(self.w)] for _ in range(self.h)]
        owner_obs = [[0 for _ in range(self.w)] for _ in range(self.h)]
        army_obs = [[0 for _ in range(self.w)] for _ in range(self.h)]

        for r in range(self.h):
            for c in range(self.w):
                if self.owner[r][c] == my_code:
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < self.h and 0 <= nc < self.w:
                                type_obs[nr][nc] = self.terrain[nr][nc]
                                raw_owner = self.owner[nr][nc]
                                if raw_owner == my_code:
                                    owner_obs[nr][nc] = 1
                                elif raw_owner == opp_code:
                                    owner_obs[nr][nc] = 2
                                else:
                                    owner_obs[nr][nc] = 0
                                army_obs[nr][nc] = self.army[nr][nc]

        # Mountains also visible as type 5 at tick 0 in fog
        for r in range(self.h):
            for c in range(self.w):
                if type_obs[r][c] == 0 and self.terrain[r][c] == 2:
                    type_obs[r][c] = 5

        return {
            "turn": self.turn,
            "height": self.h,
            "width": self.w,
            "player_id": player,
            "my_land": my_land,
            "my_army": my_army,
            "opp_land": opp_land,
            "opp_army": opp_army,
            "type": type_obs,
            "owner": owner_obs,
            "army": army_obs,
        }

    def step(self, act0, act1):
        if self.turn % 2 == 0:
            if self.owner[self.gen0[0]][self.gen0[1]] == 1:
                self.army[self.gen0[0]][self.gen0[1]] += 1
            if self.owner[self.gen1[0]][self.gen1[1]] == 2:
                self.army[self.gen1[0]][self.gen1[1]] += 1

        if self.turn % 50 == 0:
            for r in range(self.h):
                for c in range(self.w):
                    if self.owner[r][c] in (1, 2):
                        self.army[r][c] += 1

        moves = []
        for p_idx, act in enumerate([act0, act1]):
            if act and act[0] == 0:
                kind, r, c, direction, split = act
                if 0 <= direction < 4:
                    dr, dc = DIRS[direction]
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.h and 0 <= nc < self.w and self.terrain[nr][nc] != 2:
                        p_owner = 1 if p_idx == 0 else 2
                        if self.owner[r][c] == p_owner and self.army[r][c] >= 2:
                            amt = self.army[r][c] // 2 if split == 1 else self.army[r][c] - 1
                            moves.append((p_idx, r, c, nr, nc, amt))

        for p_idx, r, c, nr, nc, amt in moves:
            p_code = 1 if p_idx == 0 else 2
            opp_code = 2 if p_idx == 0 else 1

            if self.army[r][c] < amt + 1:
                continue
            self.army[r][c] -= amt

            opp_gen = self.gen1 if p_idx == 0 else self.gen0
            if self.turn >= 800 and (nr, nc) == opp_gen:
                self.owner[nr][nc] = p_code
                self.army[nr][nc] = amt
                return p_idx

            if self.owner[nr][nc] == p_code:
                self.army[nr][nc] += amt
            elif self.owner[nr][nc] == opp_code:
                if amt > self.army[nr][nc]:
                    self.owner[nr][nc] = p_code
                    self.army[nr][nc] = amt - self.army[nr][nc]
                elif amt == self.army[nr][nc]:
                    self.army[nr][nc] = 0
                else:
                    self.army[nr][nc] -= amt
            else:
                cost = 1
                if amt > cost:
                    self.owner[nr][nc] = p_code
                    self.army[nr][nc] = amt - cost
                else:
                    self.army[nr][nc] = 0

            if (nr, nc) == opp_gen and self.owner[nr][nc] == p_code:
                return p_idx

        self.turn += 1
        if self.turn >= 1200:
            return -1
        return None


def run_simulated_match(bot0_dir, bot1_dir, max_turns=600, seed=42):
    print(f"\n>> Simulating Match: {bot0_dir} (P0) vs {bot1_dir} (P1) [Seed={seed}]...")
    bot0_path = _model_path(bot0_dir, "bot.py")
    bot1_path = _model_path(bot1_dir, "bot.py")

    b0 = load_bot_module(bot0_path, f"b0_{random.randint(1000, 9999)}")
    b1 = load_bot_module(bot1_path, f"b1_{random.randint(1000, 9999)}")

    sim = LocalGeneralsSim(seed=seed)
    winner = None

    t_start = time.perf_counter()
    for turn in range(1, max_turns + 1):
        obs0 = sim.make_observation(0)
        obs1 = sim.make_observation(1)

        act0 = b0.act(obs0)
        act1 = b1.act(obs1)

        res = sim.step(act0, act1)
        if res is not None:
            winner = res
            break

    elapsed = time.perf_counter() - t_start
    if winner == 0:
        print(f"   WINNER: Player 0 ({bot0_dir}) in {sim.turn} turns! ({elapsed:.2f}s)")
    elif winner == 1:
        print(f"   WINNER: Player 1 ({bot1_dir}) in {sim.turn} turns! ({elapsed:.2f}s)")
    else:
        print(f"   DRAW: Reached turn {sim.turn} ({elapsed:.2f}s)")

    return winner


def main():
    p = argparse.ArgumentParser(description="Test and match runner for Code Bot models")
    p.add_argument("--test", action="store_true", help="Run all 22 model unit and benchmark tests")
    p.add_argument("--validate", action="store_true", help="Run official evaluator validation")
    p.add_argument("--match", nargs=2, metavar=("BOT0", "BOT1"), help="Run simulated match between two models")
    p.add_argument("--seed", type=int, default=42, help="Seed for simulated match")
    p.add_argument("--turns", type=int, default=600, help="Max turns for simulation")
    args = p.parse_args()

    if not any([args.test, args.validate, args.match]):
        t_ok = run_unit_tests()
        v_ok = run_evaluator_validation()
        print("\n>> Running Showcase Match: Model 18 (Known Good v0) vs Model 20 (KG Rush)...")
        run_simulated_match("model_18_codebot_v0_known_good", "model_20_kg_rush", max_turns=600, seed=42)
        return 0 if (t_ok and v_ok) else 1

    success = True
    if args.test:
        success = run_unit_tests() and success
    if args.validate:
        success = run_evaluator_validation() and success
    if args.match:
        run_simulated_match(args.match[0], args.match[1], max_turns=args.turns, seed=args.seed)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
