# Model 07: The War of Attrition and the 1200-Tick Draw

## 1. Overview & Paradigm
- **Archetype**: Strategy II - Full Defensive
- **Mathematical Framework**: Biological War of Attrition & Asymmetric Combat Resolution
- **Bot Name**: `AttritionTurtle`
- **File**: `bot.py`
- **Test Suite**: `tester.py`

## 2. Theoretical Principles
The War of Attrition exploits the mathematical asymmetry of Code Bot combat:
- In simultaneous turn resolution, tied combat leaves the defender in possession of the contested tile with 0 units. Initiating an equal attack is mathematically negative EV.
- The organizer rules enforce a hard 1200-tick termination: any match reaching tick 1200 unresolved is declared a draw (awarding 0.5 points to both competitors in tournament standings).
- By maintaining an unbreachable defensive perimeter around the general and refusing disadvantageous offensive trades, the agent signals an infinite willingness to wait ($T \to 1200$).
- Opponents are forced into two bad options: accept a 0.5-point draw, or make mathematically ruinous overextended breaches that deplete their army, exposing them to an endgame counter-sweep.

## 3. Algorithmic Implementation
- **Radius-3 Fortification Zone**: Pulls all dispersed units into a compact perimeter within Manhattan distance $\le 3$ of the general.
- **Asymmetric Engagement Threshold**: Strictly refuses offensive trades unless friendly army exceeds $1.5\times$ the opponent's stack.
- **Pass Clock Burner**: Issues `[1, 0, 0, 0, 0]` when positions are optimal, safely advancing turns toward the 1200 ceiling.

## 4. How to Test
Run the autonomous tester:
```bash
python3 tester.py
```
Or validate using the organizer evaluator:
```bash
python3 ../evaluator/evaluate.py validate bot.py
```
