# Model 08: Fog-of-War Signaling Game

## 1. Overview & Paradigm
- **Archetype**: Strategy II - Full Defensive
- **Mathematical Framework**: Signaling Games with Imperfect Information
- **Bot Name**: `FogSignaler`
- **File**: `bot.py`
- **Test Suite**: `tester.py`

## 2. Theoretical Principles
Under 1-tile visibility rules, players must infer opponent intentions from glimpse observations at the edge of fog:
- The defensive agent constructs a **costly signal of strength** by marching a large strike stack onto the frontier tile in full sight of the opponent's vision boundary.
- On the very next turn, the agent immediately withdraws the stack back into the fog.
- The opponent, updating their Bayesian belief distribution regarding attack probability, infers an imminent invasion and diverts resources toward home defense.
- This creates defensive paralysis in the opponent, costing the signaling player only 1 turn of local repositioning while halting enemy expansion across the map.

## 3. Algorithmic Implementation
- **Border Detection**: Identifies owned border tiles with high army count ($\ge 10$) adjacent to opponent territory or fog.
- **Two-Step Peek-and-Retreat**: State machine manages transition from state 0 (peeking forward) to state 1 (retreating backward into fog).
- **Macro Exploitation**: While the opponent turtle-defends in reaction to the signal, the agent quietly expands behind the fog shield.

## 4. How to Test
Run the autonomous tester:
```bash
python3 tester.py
```
Or validate using the organizer evaluator:
```bash
python3 ../evaluator/evaluate.py validate bot.py
```
