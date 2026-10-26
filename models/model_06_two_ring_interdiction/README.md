# Model 06: Network Interdiction on the Two-Ring Perimeter

## 1. Overview & Paradigm
- **Archetype**: Strategy II - Full Defensive
- **Mathematical Framework**: Network Interdiction & Move-Order Chasing Dynamics
- **Bot Name**: `TwoRingInterdictor`
- **File**: `bot.py`
- **Test Suite**: `tester.py`

## 2. Theoretical Principles
Under the Turn 800 Deathtouch mechanic, a single 2-unit enemy stack reaching an orthogonal neighbor of the general can execute an instant-kill attack. Traditional static turtling fails because multi-pronged 2-unit threats cannot all be blocked simultaneously.
The Two-Ring Interdiction architecture resolves this:
- **Ring 1 ($d=1$):** Direct orthogonal neighbors of the general. Garrisoned with defensive units to block vision and win numerical trades before tick 800.
- **Ring 2 ($d=2$):** Diagonal and radius-2 neighbors. Stationed with mobile reactive interceptors.
- **Move-Order Priority:** In the Code Bot engine, chasing onto a tile the opponent is vacating resolves *first*. If an enemy advances onto Ring 1, Ring 2 executes a chase move to strip the source tile down to $< 2$ units, rendering the deathtouch strike impossible to execute on the subsequent tick.

## 3. Algorithmic Implementation
- **Ring Geometry Identification**: Computes Ring 1 and Ring 2 coordinate sets relative to $(g_r, g_c)$.
- **Interdiction Override**: Checks if any enemy occupies Ring 1; immediately initiates an interdiction counter-strike.
- **Perimeter Ring Patrol**: Routes peripheral units into Ring 2 staging positions to maintain 360-degree coverage.

## 4. How to Test
Run the autonomous tester:
```bash
python3 tester.py
```
Or validate using the organizer evaluator:
```bash
python3 ../evaluator/evaluate.py validate bot.py
```
