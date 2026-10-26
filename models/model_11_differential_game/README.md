# Model 11: Continuous Differential Games (Isaacs' Framework)

## 1. Overview & Paradigm
- **Archetype**: Strategy III - Balanced
- **Mathematical Framework**: Continuous Differential Pursuit-Evasion Games (Rufus Isaacs)
- **Bot Name**: `IsaacsDifferentialStrategist`
- **File**: `bot.py`
- **Test Suite**: `tester.py`

## 2. Theoretical Principles
Using the global scoreboard scalars (`my_army`, `opp_army`), the agent models macro-development as a continuous-time differential game:
- Computes empirical first derivatives $\dot{A}(t) = \frac{d A_{\text{my}}}{dt}$ and $\dot{E}(t) = \frac{d A_{\text{opp}}}{dt}$ over a rolling temporal window.
- **Regime 1: Macro Advantage ($\dot{A}(t) \ge \dot{E}(t)$)**: Delaying direct confrontation strictly increases the player's relative power advantage. The agent adopts an economic expansion stance.
- **Regime 2: Preemptive Strike ($\dot{E}(t) > \dot{A}(t)$)**: If the opponent out-produces the player (e.g. through neutral castle acquisition or faster expansion), the agent computes the optimal stopping time to launch an immediate, all-out assault before the army differential becomes insurmountable.

## 3. Algorithmic Implementation
- **Rolling Window Derivations**: Tracks rolling queues of the last 10 ticks for both players.
- **Derivative Thresholding**: Switches dynamic state from territorial growth to preemptive thrust based on $\dot{A}$ vs $\dot{E}$.
- **Turn 750 Phase Shift**: Automatically shifts into strike mode prior to Turn 800 deathtouch.

## 4. How to Test
Run the autonomous tester:
```bash
python3 tester.py
```
Or validate using the organizer evaluator:
```bash
python3 ../evaluator/evaluate.py validate bot.py
```
