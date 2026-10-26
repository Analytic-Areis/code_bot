# Model 14: POMDP Ghost Tracking

## 1. Overview & Paradigm
- **Archetype**: Strategy IV - Adaptive
- **Mathematical Framework**: Partially Observable Markov Decision Process (POMDP) Reachability Sets
- **Bot Name**: `POMDPGhostTracker`
- **File**: `bot.py`
- **Test Suite**: `tester.py`

## 2. Theoretical Principles
When enemy stacks move out of direct vision, they do not disappear; they become invisible state vectors or "Ghosts":
- For each enemy stack $S \ge 4$ that vanishes into fog at $(x_0, y_0)$ at turn $t_0$, the agent initiates a reachability belief set.
- Because units move at most 1 cell per turn, the ghost's possible spatial support expands as a Manhattan diamond:
$$\mathcal{D}(t) = \{ (r, c) : |r - x_0| + |c - y_0| \le t - t_0 \}$$
- The worst-case arrival time $\eta$ to the friendly general $(g_r, g_c)$ is projected as:
$$\eta = \max(0, \text{dist}((x_0, y_0), (g_r, g_c)) - (t - t_0))$$
- If $\eta$ falls within a critical proximity horizon ($\le 4$ turns) and the ghost's army exceeds the general's defense, an emergency defensive recall is automatically triggered to intercept the unseen threat before impact.

## 3. Algorithmic Implementation
- **Diff Tracking**: Compares visible enemy coordinates between consecutive turns to catch vanishing armies.
- **Ghost Pruning**: Eliminates ghosts that exceed the horizon ($> 25$ turns) or are invalidated by direct sight.
- **Dynamic Response Override**: Forces inward gather flow and defensive positioning whenever an active ghost poses existential risk.

## 4. How to Test
Run the autonomous tester:
```bash
python3 tester.py
```
Or validate using the organizer evaluator:
```bash
python3 ../evaluator/evaluate.py validate bot.py
```
