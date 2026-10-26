# Model 05: Minimax Facility Location for Isolated Castles

## 1. Overview & Paradigm
- **Archetype**: Strategy II - Full Defensive
- **Mathematical Framework**: Discrete Minimax Facility Location
- **Bot Name**: `MinimaxCastleDefender`
- **File**: `bot.py`
- **Test Suite**: `tester.py`

## 2. Theoretical Principles
In Code Bot, dense castle layering is economically fatal due to the proximity surcharge:
$$\text{Cost} = 35 + \sum_{i \in \text{structures}} \max(0, 14 - 2 d_i)$$
At $d_i = 1$, a castle costs 47 army units (breakeven 94 ticks).
At $d_i \ge 7$, surcharge is 0, maintaining the baseline cost of 35 units (breakeven 70 ticks).

The Minimax Facility Location formulation seeks a coordinate $(x, y)$ that minimizes the maximum distance to the general's entrances while respecting the spatial constraint:
$$\min_{(x,y)} \max_{e \in E_{gen}} \text{dist}((x,y), e) \quad \text{s.t.} \quad |x - x_{gen}| + |y - y_{gen}| \ge 7$$

## 3. Algorithmic Implementation
- **Optimal Hub Selection**: Filters owned plain tiles with Manhattan distance $\ge 7$ from all existing structures (general and previous castles).
- **Economic Feasibility**: Only executes `kind=2` (build castle) when local tile army exceeds the calculated exact build cost.
- **Hub Reinforcement**: Castles generate $+1$ army every 2 ticks, funneling localized reinforcements directly to the home defense perimeter.

## 4. How to Test
Run the autonomous tester:
```bash
python3 tester.py
```
Or validate using the organizer evaluator:
```bash
python3 ../evaluator/evaluate.py validate bot.py
```
