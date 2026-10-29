#!/usr/bin/env python3
"""Comprehensive Latency Profiling for Model 28 Adaptive Champion.

Measures turn latency percentiles (Mean, Median, P90, P95, P99, Max)
across diverse game stages, maps, and opponent styles.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run_all_tests import load_bot_module, LocalGeneralsSim, _model_path


def run_latency_profile():
    m28_path = ROOT / "models" / "model_28_adaptive_champion" / "bot.py"
    b0 = load_bot_module(m28_path, "m28_profile")

    opponents = [
        "model_25_grandmaster_champion",
        "model_23_rl_champion",
        "model_18_codebot_v0_known_good",
        "model_20_kg_rush",
    ]

    all_latencies = []
    stage_latencies = {"early (1-100)": [], "mid (101-400)": [], "late (401-800)": [], "deathtouch (801+)": []}

    for seed in [1, 7, 42, 99, 123]:
        for opp in opponents:
            opp_path = _model_path(opp, "bot.py")
            b_opp = load_bot_module(opp_path, "opp")
            b0._reset()
            sim = LocalGeneralsSim(seed=seed)

            for turn in range(1, 900):
                o0 = sim.make_observation(0)
                o1 = sim.make_observation(1)

                t0 = time.perf_counter()
                a0 = b0.act(o0)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                all_latencies.append(elapsed_ms)
                if turn <= 100:
                    stage_latencies["early (1-100)"].append(elapsed_ms)
                elif turn <= 400:
                    stage_latencies["mid (101-400)"].append(elapsed_ms)
                elif turn <= 800:
                    stage_latencies["late (401-800)"].append(elapsed_ms)
                else:
                    stage_latencies["deathtouch (801+)"].append(elapsed_ms)

                a1 = b_opp.act(o1)
                res = sim.step(a0, a1)
                if res is not None:
                    break

    all_latencies.sort()
    n = len(all_latencies)
    mean_lat = sum(all_latencies) / n
    median_lat = all_latencies[int(n * 0.50)]
    p90_lat = all_latencies[int(n * 0.90)]
    p95_lat = all_latencies[int(n * 0.95)]
    p99_lat = all_latencies[int(n * 0.99)]
    max_lat = all_latencies[-1]

    print("=" * 70)
    print("  MODEL 28 LATENCY BENCHMARK & PERCENTILES (Limit: 150.0 ms)")
    print("=" * 70)
    print(f"Total Decisions Profiled: {n:,}")
    print(f"Mean Latency:    {mean_lat:6.2f} ms")
    print(f"Median (P50):    {median_lat:6.2f} ms")
    print(f"P90 Latency:     {p90_lat:6.2f} ms")
    print(f"P95 Latency:     {p95_lat:6.2f} ms")
    print(f"P99 Latency:     {p99_lat:6.2f} ms")
    print(f"Max Latency:     {max_lat:6.2f} ms")
    print(f"Safety Margin:   {150.0 / max(0.01, max_lat):6.1f}x below forfeiture threshold")
    print("-" * 70)
    print("Latency by Match Phase:")
    for stage, lats in stage_latencies.items():
        if lats:
            lats.sort()
            s_mean = sum(lats) / len(lats)
            s_p95 = lats[int(len(lats) * 0.95)]
            s_max = lats[-1]
            print(f"  {stage:<18} (n={len(lats):>5}): mean={s_mean:5.2f}ms, p95={s_p95:5.2f}ms, max={s_max:5.2f}ms")
    print("=" * 70)


if __name__ == "__main__":
    run_latency_profile()
