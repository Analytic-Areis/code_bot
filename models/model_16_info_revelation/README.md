# Model 16: Information Revelation in Repeated Subgames

## 1. Overview & Paradigm
- **Archetype**: Strategy IV - Adaptive
- **Mathematical Framework**: Information Economics & Value of Information (VoI) in Repeated Games
- **Bot Name**: `VoIProber`
- **File**: `bot.py`
- **Test Suite**: `tester.py`

## 2. Theoretical Principles
In repeated interactions at the fog boundary, acquiring information carries a unit cost (expending army to step forward):
- Capturing a neutral plain costs 1 army unit but reveals a $3 \times 3$ tile window of previously obscured terrain.
- When uncertainty regarding the enemy general or the distribution of opponent forces is high, the Value of Information (VoI) strictly exceeds the marginal cost of 1 army unit.
- The agent launches deliberate probing attacks into high-uncertainty fog cells to force the opponent to react.
- Any observable reaction by the opponent (stack movement, border reinforcement) immediately collapses Bayesian entropy, exposing their deployment and unlocking decisive counter-measures.

## 3. Algorithmic Implementation
- **Fog Revelation Metric**: Computes exact count of obscured tiles ($0$ or $5$) uncovered by candidate moves.
- **VoI Objective Optimization**: Scores moves based on visibility gain, frontier contact, and unit efficiency.
- **Information Saturation Shift**: Once the enemy general is sighted, VoI drops to zero and the agent executes an unyielding direct assault.

## 4. How to Test
Run the autonomous tester:
```bash
python3 tester.py
```
Or validate using the organizer evaluator:
```bash
python3 ../evaluator/evaluate.py validate bot.py
```
