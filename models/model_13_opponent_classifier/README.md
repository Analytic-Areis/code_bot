# Model 13: Bayesian Extensive Form Game for Opponent Classification

## 1. Overview & Paradigm
- **Archetype**: Strategy IV - Adaptive
- **Mathematical Framework**: Incomplete Information Game with Bayesian Type Inference
- **Bot Name**: `BayesianOpponentClassifier`
- **File**: `bot.py`
- **Test Suite**: `tester.py`

## 2. Theoretical Principles
In competitive Generals.io, playstyles adhere to a non-transitive meta-dynamic across three fundamental types:
$$\Theta \in \{ \text{Rusher}, \text{Turtle}, \text{Greedy Expander} \}$$
- **Rusher** beats **Greedy Expander** (exploits defenseless home base).
- **Greedy Expander** beats **Turtle** (macro land bonus dwarfs turtle production).
- **Turtle** beats **Rusher** (absorbs rush on fortified garrison, rusher starves).

The agent maintains an online prior $P(\Theta)$ initialized uniformly at turn 0, and updates beliefs dynamically using scoreboard tokens:
1. **$\Delta A_{\text{opp}} \le -35$ with $\Delta L_{\text{opp}} = 0$**: Castle investment $\to P(\text{Turtle}) \uparrow$.
2. **Enemy stack $\ge 5$ sighted near home before tick 100**: Rusher detected $\to P(\text{Rusher}) \uparrow$.
3. **$\Delta L_{\text{opp}} \ge 2$ with low defensive garrison**: Greedy expansion $\to P(\text{Expander}) \uparrow$.

## 3. Algorithmic Counter-Strategies
- **Against Rusher ($P > 0.6$)**: Contracts units into home garrison, absorbs incoming rush, and executes a counter-sweep on the empty enemy base.
- **Against Turtle ($P > 0.6$)**: Claims 75%+ of the map's neutral territory, bypasses fortified choke points, and prepares a dual-entrance Deathtouch at tick 800.
- **Against Expander ($P > 0.6$)**: Aggregates a concentrated spear-tip force and pierces the diffuse, thinly protected frontier directly toward the enemy general.

## 4. How to Test
Run the autonomous tester:
```bash
python3 tester.py
```
Or validate using the organizer evaluator:
```bash
python3 ../evaluator/evaluate.py validate bot.py
```
