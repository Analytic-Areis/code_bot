# Model 01: Bayesian Search Game for Target Acquisition

## 1. Overview & Paradigm
- **Archetype**: Strategy I - Full Offensive
- **Mathematical Framework**: Bayesian Search Theory / Optimal Search Game
- **Bot Name**: `BayesianSearchScout`
- **File**: `bot.py`
- **Test Suite**: `tester.py`

## 2. Theoretical Principles
In imperfect-information grid environments shrouded by fog of war, locating the enemy general at tick 0 is an unknown parameter. The Code Bot competition engine enforces a spawn-fairness filter:
1. The enemy general is placed at least 17 Breadth-First Search (BFS) steps away from the friendly general.
2. The number of passable tiles within a 7-step BFS radius ("room" metric) matches the friendly general within $\pm 5$.

This reduces the search space from $441$ tiles down to an initial candidate cluster of approximately 28 coordinates.

### Bayesian Belief Updating
We maintain a prior probability distribution $P(\omega)$ over all valid candidate coordinates $\Omega$.
As the agent expands, visible tiles revealed through the 1-tile observation radius are inspected. If a tile $(r, c) \in \Omega$ is revealed and does not contain the enemy general, it is eliminated from $\Omega$, and the posterior probability is updated via Bayes' rule:
$$P(\omega \mid O_t) = \frac{P(O_t \mid \omega) P(\omega)}{\sum_{\omega' \in \Omega} P(O_t \mid \omega') P(\omega')}$$

## 3. Algorithmic Implementation
- **Prior Generation**: At turn 0/1, BFS computes topological distance from friendly general. Tiles with distance $\ge 16$ form initial candidate space.
- **Scouting Vectors**: The agent finds the maximum probability candidate node $\arg\max_{\omega} P(\omega)$ and routes forward tendrils along the shortest BFS distance gradient.
- **Decapitation**: Once the enemy general is sighted, target acquisition locks in immediately with maximal offensive bonus, terminating in lethal capture or Turn 800 deathtouch.
- **Safety Margin**: 120 ms cutoff timer ensuring compliance with the strict 150 ms deadline.

## 4. How to Test
Run the autonomous tester:
```bash
python3 tester.py
```
Or validate using the organizer evaluator:
```bash
python3 ../evaluator/evaluate.py validate bot.py
```
