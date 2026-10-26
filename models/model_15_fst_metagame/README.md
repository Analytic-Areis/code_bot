# Model 15: Finite State Transducers (FST) for Metagame Transitions

## 1. Overview & Paradigm
- **Archetype**: Strategy IV - Adaptive
- **Mathematical Framework**: Mealy-Type Finite State Transducer (FST)
- **Bot Name**: `FSTMetagameAutomaton`
- **File**: `bot.py`
- **Test Suite**: `tester.py`

## 2. Theoretical Principles
Heuristic bots frequently suffer from the "ping-pong bug," oscillating endlessly between offensive pathing and defensive retreat when evaluation scores hover near parity.
To eliminate this failure mode, the agent models macro gameplay as an $\mathcal{O}(1)$ Mealy Finite State Transducer with discrete operational states:
- **$S_0$: OPENING ($t \le 60$)**: Rapid outward tendril expansion, prioritizing unowned land capture before the first macro ticks.
- **$S_1$: STRIKE**: Triggered when a consolidated vanguard stack exceeds threshold and an assault vector / enemy general is available.
- **$S_2$: DEFEND**: Triggered unconditionally when a lethal threat appears within radius 4 of the general.
- **$S_3$: DEATHTOUCH STAGING ($t \ge 740$)**: Liquidates economic assets, transitioning all units into parallel 2-unit infiltration runners to exploit Turn 800 Deathtouch.

## 3. Algorithmic Implementation
- **Deterministic State Transition Function**: Evaluates inputs (turn, threat radius, vanguard size) to transition state in $\mathcal{O}(1)$ time.
- **Action Decoupling**: Each state maps to a specialized objective function, ensuring clear, decisive commitments on every tick.

## 4. How to Test
Run the autonomous tester:
```bash
python3 tester.py
```
Or validate using the organizer evaluator:
```bash
python3 ../evaluator/evaluate.py validate bot.py
```
