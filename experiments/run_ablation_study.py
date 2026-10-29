#!/usr/bin/env python3
"""Ablation Study for Model 28 Adaptive Champion.

Isolates:
A0: Full Model 28 Adaptive Champion
A1: Ablation - No Multi-Source DAG Frontline Funnel (reverts to M27 static consolidation)
A2: Ablation - No Threat-Horizon Pullback (reverts to M27 raw army gain)
A3: Ablation - No General Fortress Hold (reverts to M27 garrison rules)
A4: Ablation - No Live Bayesian Frontier Likelihood (static target only)

Evaluates on paired seeds against M25 and M23 (both P0 and P1).
"""
import copy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run_all_tests import load_bot_module, LocalGeneralsSim, _model_path
import models.model_28_adaptive_champion.bot as full_bot


def run_paired_evaluation(bot_mod, name, seeds=(1, 7, 42, 99, 123), opponents=("model_25_grandmaster_champion", "model_23_rl_champion")):
    wins = 0
    total = 0
    turns_list = []

    for s in seeds:
        for opp in opponents:
            opp_path = _model_path(opp, "bot.py")
            # Slot 0
            b_opp = load_bot_module(opp_path, "opp_0")
            if hasattr(bot_mod, "_reset"):
                bot_mod._reset()
            sim = LocalGeneralsSim(seed=s)
            w = None
            for t in range(1, 1000):
                o0 = sim.make_observation(0)
                o1 = sim.make_observation(1)
                a0 = bot_mod.act(o0)
                a1 = b_opp.act(o1)
                res = sim.step(a0, a1)
                if res is not None:
                    w = res
                    break
            if w == 0:
                wins += 1
            total += 1
            turns_list.append(sim.turn)

            # Slot 1
            b_opp = load_bot_module(opp_path, "opp_1")
            if hasattr(bot_mod, "_reset"):
                bot_mod._reset()
            sim = LocalGeneralsSim(seed=s)
            w = None
            for t in range(1, 1000):
                o0 = sim.make_observation(0)
                o1 = sim.make_observation(1)
                a0 = b_opp.act(o0)
                a1 = bot_mod.act(o1)
                res = sim.step(a0, a1)
                if res is not None:
                    w = res
                    break
            if w == 1:
                wins += 1
            total += 1
            turns_list.append(sim.turn)

    wr = (wins / total) * 100.0
    mean_turns = sum(turns_list) / len(turns_list)
    print(f"{name:<42} | WR: {wins:>2}/{total} ({wr:5.1f}%) | Avg Turns: {mean_turns:5.1f}")
    return wr, mean_turns


if __name__ == "__main__":
    print("=" * 75)
    print("  MODEL 28 ABLATION EXPERIMENTS ACROSS PAIRED TOURNAMENT SEEDS")
    print("=" * 75)

    # 1. Full Model 28
    m28_path = ROOT / "models" / "model_28_adaptive_champion" / "bot.py"
    m28 = load_bot_module(m28_path, "m28")
    run_paired_evaluation(m28, "A0: Full Model 28 Adaptive Champion")

    # 2. Ablation 1: No DAG Funnel
    p_no_dag = ROOT / "experiments" / "ablations" / "m28_no_dag_funnel.py"
    b_no_dag = load_bot_module(p_no_dag, "b_no_dag")
    run_paired_evaluation(b_no_dag, "A1: Without DAG Funnel & Mobilization")

    # 3. Ablation 2: No Threat Pullback
    p_no_pull = ROOT / "experiments" / "ablations" / "m28_no_threat_pullback.py"
    b_no_pull = load_bot_module(p_no_pull, "b_no_pull")
    run_paired_evaluation(b_no_pull, "A2: Without Threat-Horizon Pullback")

    # 4. Ablation 3: No Fortress Hold
    p_no_fort = ROOT / "experiments" / "ablations" / "m28_no_fortress_hold.py"
    b_no_fort = load_bot_module(p_no_fort, "b_no_fort")
    run_paired_evaluation(b_no_fort, "A3: Without General Fortress Hold")

    # 5. Ablation 4: No Bayesian Evidence Target
    p_no_bayes = ROOT / "experiments" / "ablations" / "m28_no_bayesian_evidence.py"
    b_no_bayes = load_bot_module(p_no_bayes, "b_no_bayes")
    run_paired_evaluation(b_no_bayes, "A4: Without Bayesian Frontier Target")

    # 6. Model 27 Champion (Original Baseline)
    m27_path = ROOT / "models" / "model_27_final_champion" / "bot.py"
    m27 = load_bot_module(m27_path, "m27")
    run_paired_evaluation(m27, "Original Baseline: Model 27 Final Champion")
    print("=" * 75)
