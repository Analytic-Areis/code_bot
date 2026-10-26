# Model 04: Stackelberg Leadership Competition for Early Expansion

## 1. Overview & Paradigm
- **Archetype**: Strategy I - Full Offensive
- **Mathematical Framework**: Stackelberg Leadership Game
- **Bot Name**: `StackelbergLeader`
- **File**: `bot.py`
- **Test Suite**: `tester.py`

## 2. Theoretical Principles
In the early game (turns 0-100), territorial expansion is a leadership game:
- The offensive agent acts as the **Stackelberg Leader** by committing unequivocally to an aggressive, forward-thrusting expansion vector.
- The opponent (**Follower**), observing the rapid approach of enemy units toward their sector, is forced into a reactive defensive posture.
- By pulling units back to guard their home general, the follower forfeits early neutral land captures.
- The leader exploits this concession by capturing maximal plain tiles right before the critical modulus-50 land reinforcement ticks ($t = 50, 100, 150$), locking in a compound economic growth advantage.

## 3. Algorithmic Implementation
- **Leadership Commitment**: Commits movement vectors straight outward into unowned territory.
- **Modulus-50 Acceleration**: Prioritizes capturing neutral plains during $t \pmod{50} \in [35, 49]$ to maximize the $+1$ army per cell bonus.
- **Offensive Follow-Through**: Uses the compound economic army advantage to overwhelm the defensive follower.

## 4. How to Test
Run the autonomous tester:
```bash
python3 tester.py
```
Or validate using the organizer evaluator:
```bash
python3 ../evaluator/evaluate.py validate bot.py
```
