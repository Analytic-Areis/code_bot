# Model 20: KG Hyper-Aggressive Rusher (`kg_rush`)

## 1. Overview & Paradigm
- **Archetype**: Adversarial Sparring Model / Hyper-Aggressive Rusher
- **Mathematical Framework**: Low-Latency Decapitation Rush with Minimal Defensive Buffers
- **Bot Name**: `KGRush`
- **File**: `bot.py`
- **Test Suite**: `tester.py`
- **Source**: `4 Never test against reference.txt` (Line 1754)

## 2. Theoretical Principles
`kg_rush` is the hyper-aggressive mutation of the core engine designed to stress-test early defensive capabilities:
- **Immediate Opening Launch**: Rather than waiting for 13 units on the general, it streams troops outward as soon as army $\ge 6$.
- **Minimal Defensive Margins**: Ignores secondary territorial preservation, directing all momentum down the shortest topological path toward the opponent's candidate general quadrant.
- **Exploiting Low Early Garrisons**: Empirical testing in `4 Never test against reference.txt` proved that most hackathon bots (such as `Hunter`, `Expander`, and `Starter`) leave their home general with only 3–9 units between turns 50–200. A concentrated 15–20 unit rush arriving early scores rapid, uncounterable victories.

## 3. How to Test
Run the autonomous tester:
```bash
python3 tester.py
```
Or validate using the organizer evaluator:
```bash
python3 ../evaluator/evaluate.py validate bot.py
```
