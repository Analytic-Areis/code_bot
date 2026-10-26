# Model 03: Asymmetric Colonel Blotto Game for Breaching

## 1. Overview & Paradigm
- **Archetype**: Strategy I - Full Offensive
- **Mathematical Framework**: Asymmetric Sequential Colonel Blotto Game
- **Bot Name**: `BlottoBreacher`
- **File**: `bot.py`
- **Test Suite**: `tester.py`

## 2. Theoretical Principles
Breaching a fortified enemy general surrounded by $K \le 4$ orthogonal entrances presents an asymmetric game:
- The defending player must divide total garrison $D$ across all $K$ passable entrances, yielding approximately $D/K$ per entrance.
- Because the attacker holds consolidated army $A$ just inside the fog of war, the attacker possesses an informational advantage.
- The attacker directs the consolidated strike $A$ at the entrance with minimal defense $\min_k D_k$.
- Since $A > D/K$, localized numerical superiority is achieved, collapsing the perimeter before reinforcements can mobilize.

## 3. Algorithmic Implementation
- **Entrance Scanning**: When enemy general $(g_r, g_c)$ is revealed, identifies all passable orthogonal entrances.
- **Defense Profiling**: Evaluates visible defense on each entrance tile.
- **Concentrated Vectoring**: Routes the main offensive stack toward the weakest entrance tile.
- **Deathtouch / Decapitation**: If adjacent to the general with superior army or if turn $\ge 800$, triggers immediate strike onto $(g_r, g_c)$.

## 4. How to Test
Run the autonomous tester:
```bash
python3 tester.py
```
Or validate using the organizer evaluator:
```bash
python3 ../evaluator/evaluate.py validate bot.py
```
