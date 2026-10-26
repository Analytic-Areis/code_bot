# Model 02: Prize-Collecting Steiner Tree (PCST) for Army Gathering

## 1. Overview & Paradigm
- **Archetype**: Strategy I - Full Offensive
- **Mathematical Framework**: Rooted Prize-Collecting Steiner Tree (PCST) Heuristic
- **Bot Name**: `SteinerTreeAggregator`
- **File**: `bot.py`
- **Test Suite**: `tester.py`

## 2. Theoretical Principles
Army fragments scattered across wide territory waste strategic potential. In RTS grid games, units must be aggregated into a single lethal spear-tip before launching an assault.
We formalize this aggregation as a Rooted Prize-Collecting Steiner Tree problem:
- Graph $G = (V, E)$ over friendly territory.
- Edge latency cost $c_e = 1$ tick.
- Node prize $\pi_v = \max(0, \text{army}[v] - 1)$ (leaving 1 unit to retain territory ownership).
- Objective:
$$\min_T \left( \sum_{e \in T} c_e + \sum_{v \notin T} \pi_v \right)$$

## 3. Algorithmic Implementation
- **Root Identification**: Dynamically designates the maximum army stack as the spear-tip root.
- **Tree-Based Inward Routing**: Uses Breadth-First Search on owned territory to construct a shortest-path spanning tree rooted at the spear-tip.
- **Prize-Density Priority**: Armies on peripheral tree branches flow inward along tree edges with maximal $\frac{\pi_v}{\text{dist}(v, \text{root})}$.
- **Breakthrough Release**: Once the spear-tip accumulates overwhelming force ($\ge 25$ units or turn $\ge 800$), it stops gathering and strikes relentlessly into enemy territory.

## 4. How to Test
Run the autonomous tester:
```bash
python3 tester.py
```
Or validate using the organizer evaluator:
```bash
python3 ../evaluator/evaluate.py validate bot.py
```
