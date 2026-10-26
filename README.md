# Code Bot — Generals.io Hackathon

> **Final Champion:** [`model_27_final_champion/bot.py`](./models/model_27_final_champion/bot.py) · **98.0% win rate** · 49 W / 0 L / 1 D in 50 games · 4.51 ms peak latency (limit 150 ms)
>
> **Author:** Lakshmi Sai Bhargav Vemparala — lakshmisaibhargav.v@students.iiit.ac.in

---

## At a Glance

| | |
|:---|:---|
| **Result** | 49 wins, 0 losses, 1 draw against 25 opposing bots (both player slots) |
| **Approach** | Hand-built heuristic core with ES-tuned weights, plus beam search and a small neural head that can only nudge scores |
| **Constraints met** | Standard library only, one 33.5 KB file, ~33× under the time limit |
| **Where to look** | [Final bot](./models/model_27_final_champion/bot.py) · [Results](#final-tournament-results) · [IEEE-style report](docs/IEEE_Report_Generals_IO_Bot.md) · [Dashboard](docs/dashboard.html) |

---

## What is Code Bot?

**Code Bot** is a 20-hour individual hackathon (Fuzzy Dynamics, Hyderabad, 2 October 2026). Each participant submits a single `.py` file that implements `act(observation)` — an autonomous bot for the **generals.bot** turn-based strategy game. Bots compete in a 1v1 round-robin tournament; the champion collects US\$500.

**Key constraints:**
- Python 3.12, standard library **only** (no numpy, torch, etc.)
- `act(obs)` must return `[kind, row, col, direction, split]` in ≤ 150 ms (immediate forfeit on timeout)
- File ≤ 1 MiB
- Game ends when one player captures the opponent's **general** tile (or Draw at Turn 1200)
- **Turn 800+ Deathtouch**: attacking the enemy general wins regardless of army size

---

## Development Journey

| Phase | Models | What changed | Win rate |
|:---:|:---|:---|:---:|
| 1 | 01–17 | 16 game-theoretic frameworks + 1 synthesis; used as sparring partners | 41.2% (avg.) |
| 2 | 23 | (1+λ)-ES self-play tunes the scoring weights | 85.4% |
| 3 | 25 | Defence, opponent adaptation and castles; beats all 23 earlier models | 97.9% |
| 4 | 26 | Pure-Python neural value net (overweighted, so it regressed) | 93.8% |
| 5 | 27 | Beam search + calibrated neural head on M25's core | **98.0%** |

---

## Phase 1 — 16 Game-Theoretic Baseline Models

The first 16 models each implement a distinct game-theoretic or decision-theoretic framework, serving as sparring partners and research ground for what strategies actually work.

| # | Model | Framework | Core Idea |
|:---:|:---|:---|:---|
| 01 | `bayesian_search` | Bayesian Search Games | Spawn-fairness prior + room-size filter ≥17 steps |
| 02 | `pcst_gathering` | Prize-Collecting Steiner Tree | Rooted inward army flow aggregation |
| 03 | `colonel_blotto` | Asymmetric Colonel Blotto | Attack when A > D/K across K fronts |
| 04 | `stackelberg_expansion` | Stackelberg Leadership | Commit first, exploit Mod-50 economy burst |
| 05 | `minimax_castle` | Minimax Facility Location | Optimal castle placement at d ≥ 7 |
| 06 | `two_ring_interdiction` | Network Interdiction | Two-ring blocking to intercept army flows |
| 07 | `war_of_attrition` | Biological WoA | Turtle, stall to Turn 1200 draw |
| 08 | `fog_signaling` | Costly Signaling | Border peeking to signal army strength |
| 09 | `mab_exploration` | UCB1 Multi-Armed Bandit | Explore fog vs exploit known tiles |
| 10 | `nash_bargaining` | Nash Bargaining | Dynamic offense/defense resource split |
| 11 | `differential_game` | Isaacs' Differential Game | Continuous gradient pursuit strategy |
| 12 | `frontier_minimax` | Alpha-Beta Minimax | Zero-sum frontier skirmish pruning |
| 13 | `opponent_classifier` | Bayesian Type Classification | Rush/Turtle/Boom opponent detection |
| 14 | `pomdp_ghost_tracking` | POMDP Ghost Reachability | Fog tracking via reachability diamonds |
| 15 | `fst_metagame` | Mealy FST Automaton | State machine eliminating ping-pong |
| 16 | `info_revelation` | Value of Information | VoI probing subgames |
| 17 | `proposed_cascade` | Priority Cascade Synthesis | Unified synthesis of all 16 |

**Outcome:** These bots were deliberately weak individually but together formed a comprehensive adversarial test suite. Each embodies one pure principle; none could beat the others consistently.

---

## Phase 2 — First RL Champion (Model 23)

With 16 sparring partners ready, **Model 23** was trained by **Evolution Strategy (1+λ)-ES self-play** — the bot plays against itself and all 16 game-theory models, mutating scalar utility weights until it finds the best-responding parameter set.

```
Model 23 Architecture:
  • 3-Layer Bayesian Target Tracking (Room-Fairness + Mirror Spawn Prior)
  • (1+λ)-ES evolved scalar weights: DIST_WEIGHT, ARMY_BONUS, CAPTURE_WEIGHT, ...
  • 2-step tactical lookahead
  • Turn 800 Deathtouch staging
```

**Result:** Model 23 beat all 16 game-theory models. However, it could occasionally be outmanoeuvred on specific map seeds by aggressive rush bots (Models 20–22).

---

## Phase 3 — Second RL Champion (Model 25 — Grandmaster)

**Model 25** was trained to beat **Model 23** and all 22 preceding models simultaneously. Additional architectural advances:

- **Real Danger Pullback Defense**: BFS from general, streams defenders home when `worst_excess > 0`
- **Adjacent-Enemy General Guard**: General never exposes itself when enemy is 1 step away
- **Dynamic Opponent Adaptation**: Real-time Rusher / Turtle / Boomer classifier adjusting 8 weights per-turn
- **Strategic Castle Economics**: `kind=2` build action at d ≥ 7, cost formula: `35 + Σ max(0, 14−2d)`
- **Garrison floor escalation**: `garrison_min = max(18, GARRISON_MIN)` after Turn 150

**ES-tuned weights (final evolved values):**

| Parameter | Value | Purpose |
|:---|---:|:---|
| `DIST_WEIGHT` | 24.068 | Approach bonus per BFS step closer to target |
| `ARMY_BONUS` | 0.403 | Stack concentration reward |
| `CAPTURE_WEIGHT` | 573.321 | Enemy tile capture score |
| `ATTACK_PENALTY` | −284.901 | Losing skirmish deterrent |
| `KILL_BONUS` | 11,000.0 | Direct general capture reward |
| `DEATHTOUCH_BONUS` | 13,832.318 | Turn 800+ forced win bonus |
| `SPREAD_MOD50_BONUS` | 15.198 | Mod-50 economy burst multiplier |

**Tournament Result:** **47/48 wins (97.9%)** across all 23 opposing models, 0 draws.

---

## Phase 4 — ML Integration (Model 26 — Apex Predator)

With the standard-library constraint, a **pure-Python 3-layer neural value network** was introduced:

```
Architecture:  16 inputs → 32 hidden (ReLU) → 16 hidden (ReLU) → 1 output (linear)
Training:      Online Temporal Difference (TD) backpropagation, lr = 0.008
Inference:     Every turn (< 0.1 ms per forward pass)
```

**16 input features:**
1. `Δd_target` — BFS distance delta toward enemy general  
2. `army_advantage` — tanh ratio of attacking vs defending army  
3. `enemy_proximity` — 1 / (1 + min dist to visible enemy)  
4. `defense_pull` — BFS distance delta toward own general  
5. `mod50_phase` — turn % 50 normalized  
6–8. `is_enemy`, `is_neutral`, `is_fog` — destination tile type  
9. `is_castle` — destination is castle tile  
10. `adjacent_threat` — max enemy army adjacent to destination  
11. `hidden_mass` — estimated unseen opponent army  
12. `expansion_rate` — opponent land growth rate  
13. `worst_excess` — general danger level  
14. `castle_count` — number of friendly structures  
15. `loop_penalty` — anti-ping-pong flag  
16. `turn_phase` — turn / 1200  

**Also added:** Iterative Deepening Search, Dynamic Potential Fields

**Challenge encountered:** The raw neural utility signal was too large relative to heuristic scores, occasionally overriding proven tactical rules. This caused the 3 extra losses vs Model 25.

**Tournament Result:** **45/48 wins (93.8%)** — strong but slightly behind M25 due to miscalibrated weighting.

---

## Phase 5 — Final Champion (Model 27)

Model 27 synthesizes all lessons: it keeps M25's proven heuristic core and adds **four carefully calibrated ML upgrades** that never override the hard tactical rules.

### Architecture

```
Tier 0 — Contract & History Guard
  └─ 5-int return contract, _reset() isolation, anti-ping-pong deque

Tier 1 — Hard Reflexes (< 0.1 ms)
  ├─ Lethal snipe: adjacent friendly with army > enemy general
  ├─ Adjacent defense: reinforce general from allies, then counter-attack
  └─ Real Danger Pullback: BFS-priority defender streaming

Tier 2 — Castle Economics
  └─ kind=2 build at d≥7, turns 60–350, up to 2 castles

Tier 3 — BFS Distance Map + Bayesian Prior
  ├─ Room-fairness candidate filter (d≥17, Δroom≤5)
  └─ Mirror spawn prior: P(spawn) ∝ 1/(1 + d_mirror^1.5)

Tier 4 — Full Move Enumeration + Heuristic Score
  ├─ M25 ES-tuned weights
  ├─ Dynamic Rusher / Turtle / Boomer adaptation
  └─ 2-step vulnerability check

Tier 5 — [NEW] Beam Search + 1-ply Opponent Simulation
  ├─ Sort top-10 candidates by heuristic score
  ├─ Forward-simulate board after each move
  ├─ Check if opponent can immediately recapture (−85 penalty)
  └─ Check if move enables lethal snipe next turn (+bonus)

Tier 6 — [NEW] Calibrated Neural Head (additive, ±8 clamped)
  ├─ 3-layer ReLU net (16→32→16→1)
  ├─ Additive only: final_score = heuristic + nn_bonus × 0.8
  ├─ Online TD learning (lr = 0.0003, very slow drift)
  └─ Clamped to never override heuristic decisions

Tier 7 — [NEW] Enhanced DAG Army Funnel
  ├─ Find best frontline tile (army≥3, min dist to target)
  └─ Score moves toward frontline: +30 + army×0.3

Tier 8 — [NEW] Deathtouch Column Routing (turn 720+)
  └─ Route largest attack stack toward enemy general via dist_map
```

### Why M27 beats M25

| Scenario | M25 behaviour | M27 behaviour |
|:---|:---|:---|
| Walking into enemy adjacency | 2-step check catches most | Beam search confirms with 1-ply sim |
| Scattered rear armies | Slow consolidation | DAG funnel channels to single spearhead |
| Approach to enemy general at 750 | Relies on dist_weight | Dedicated column routing at 720+ |
| Neural signal | None | Additive ±8 calibrated bonus |

---

## Final Tournament Results

```
==================================================================================
  model_27_final_champion vs ALL 25 MODELS (50 games total)
==================================================================================
vs Models 01–22 (22 opponents):   44 W / 0 L / 0 D  →  100.0%
vs model_23_rl_champion:           1 W / 0 L / 1 D  →  50% (draw on isolated map)
vs model_25_grandmaster_champion:  2 W / 0 L / 0 D  →  100%
vs model_26_apex_predator:         2 W / 0 L / 0 D  →  100%
==================================================================================
  TOTAL: 49 W  0 L  1 D  /  50 GAMES   →   WIN RATE: 98.0%
==================================================================================

  Model    | W  | L | D | WR%   | MaxLat
  ---------|----+---+---+-------+--------
  M27 ✅   | 49 | 0 | 1 | 98.0  |  4.51ms
  M25      | 47 | 1 | 0 | 97.9  | 11.55ms
  M26      | 45 | 3 | 0 | 93.8  |  4.65ms
```

---

## Performance & Empirical Visualizations

### Win rate across development phases
Model 26 regressed because its raw neural score overrode tactical rules; Model 27 fixed this by clamping the neural signal.

![Win Rate Progression](docs/figures/fig1_win_rate_progression.png)

### Peak latency vs the 150 ms rule
![Latency Analysis](docs/figures/fig2_latency_analysis.png)

### Model 27 results by opponent group
![Category Breakdown](docs/figures/fig3_category_breakdown.png)

### Model 27 decision pipeline
![Architecture Pipeline](docs/figures/fig4_architecture_pipeline.png)

---

## IEEE Research Paper & Interactive Dashboard

- **Official Research Report:** [📄 `docs/IEEE_Report_Generals_IO_Bot.md`](docs/IEEE_Report_Generals_IO_Bot.md)
  - Full formal formulation, game rules, mathematical proofs, ablation studies, and complete citations in IEEE conference paper format.
  - **Author:** Lakshmi Sai Bhargav Vemparala (`lakshmisaibhargav.v@students.iiit.ac.in`), IIIT Hyderabad.
- **Interactive Championship Dashboard:** [🌐 `docs/dashboard.html`](docs/dashboard.html)
  - Interactive multi-tab browser dashboard featuring real-time KPI metrics, match browser, and architecture deep dives.

---

## Model Directory Layout

```
code_bot/
├── models/
│   ├── model_01_bayesian_search/        Phase 1 — Game Theory
│   ├── model_02_pcst_gathering/
│   ├── ...
│   ├── model_16_info_revelation/
│   ├── model_17_proposed_cascade/       Phase 1 — Synthesis
│   ├── model_18_codebot_v0_known_good/  Phase 1 — Adversarial sparring
│   ├── model_19_codebot_v1_probe_stage/
│   ├── model_20_kg_rush/
│   ├── model_21_kg_fortress/
│   ├── model_22_pinner/
│   ├── model_23_rl_champion/            Phase 2 — First RL champion
│   │                                    (no model_24: numbering skips it)
│   ├── model_25_grandmaster_champion/   Phase 3 — Second RL champion (97.9%)
│   ├── model_26_apex_predator/          Phase 4 — ML exploration (93.8%)
│   └── model_27_final_champion/         Phase 5 — FINAL SUBMISSION (98.0%) ✅
├── docs/
│   ├── Code_Bot_Rules.pdf
│   └── Code_Bot_Slides.pdf
├── evaluator/
│   └── evaluate.py                      Official organizer validator
├── run_all_tests.py                     Full tournament runner
└── README.md
```

---

## Quick Start

```bash
# Validate the final submission bot
python3 evaluator/evaluate.py validate models/model_27_final_champion/bot.py

# Simulate a 1v1 match
python3 run_all_tests.py --match model_27_final_champion model_25_grandmaster_champion --seed 42 --turns 1000

# Validate all 26 bots with the evaluator
python3 run_all_tests.py --validate
```

---

## Submission

| Field | Value |
|:---|:---|
| **File** | `models/model_27_final_champion/bot.py` |
| **Size** | 33,557 bytes |
| **SHA-256** | `1383b8ebe7ab7998f45e2680b081e1d53f751db27cda500b238886f4a78c4b8a` |
| **Static check** | `passed` |
| **Win rate** | **98.0%** (49/50) |
| **Peak latency** | **4.51 ms** (33× under 150 ms limit) |

---

*Author: Lakshmi Sai Bhargav Vemparala — lakshmisaibhargav.v@students.iiit.ac.in*
