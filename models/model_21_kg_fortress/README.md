# Model 21: KG Economic Fortress (`kg_fortress`)

## 1. Overview & Paradigm
- **Archetype**: Adversarial Sparring Model / Economic Fortress
- **Mathematical Framework**: High-Garrison Home Reserve with Minimax Loose Castles & Deathtouch Resilience
- **Bot Name**: `KGFortress`
- **File**: `bot.py`
- **Test Suite**: `tester.py`
- **Source**: `4 Never test against reference.txt` (Line 1755)

## 2. Theoretical Principles
`kg_fortress` is engineered specifically to challenge and test late-game endgame logic:
- **Home & Exposure Reserve**: Maintains a heavy garrison on the general at all times, fortifying all passable entrances to absorb enemy rushes without collapsing.
- **Loose Castle Infrastructure**: Constructs high-value economic castles at Manhattan distance $\ge 7$ from existing structures starting at tick 60, zeroing out proximity surcharges ($Cost = 35$) and compounding long-term troop production ($+0.5$ army/tick).
- **Deathtouch Endgame Test**: Defends general entrances via priority chase mechanics from turn 790 onward, reliably dragging matches into the post-800 phase to verify that attacking bots can execute true multi-pronged deathtouch infiltrations.

## 3. How to Test
Run the autonomous tester:
```bash
python3 tester.py
```
Or validate using the organizer evaluator:
```bash
python3 ../evaluator/evaluate.py validate bot.py
```
