# Model 12: Zero-Sum Extensive-Form Subgame with Perfect Information

## 1. Overview & Paradigm
- **Archetype**: Strategy III - Balanced
- **Mathematical Framework**: Extensive-Form Zero-Sum Game Tree with Minimax Pruning
- **Bot Name**: `FrontierMinimaxTactician`
- **File**: `bot.py`
- **Test Suite**: `tester.py`

## 2. Theoretical Principles
When territorial borders touch along the frontier, both players observe local stack sizes within their mutual 1-tile vision cones. This creates a local extensive-form subgame with perfect information.
Combat resolution rules dictate that:
- An attack that overpowers the defender captures the tile with remaining units: $A_{\text{rem}} = A_{\text{atk}} - 1 - A_{\text{def}}$.
- A tied attack ($A_{\text{atk}} - 1 = A_{\text{def}}$) leaves the defender in possession of the tile with 0 units. Tied attacking moves are strictly suboptimal.
- Defending choke points is structurally advantaged over initiating attacks.
The agent evaluates candidate frontier moves using a 2-ply Minimax algorithm, projecting the worst-case counter-move by the opponent to ensure every engaged trade yields positive expected troop differential.

## 3. Algorithmic Implementation
- **Contact Detection**: Scans for adjacent friendly and opponent occupied tiles.
- **Skirmish Minimax**: Computes net post-exchange delta for all possible collisions.
- **Pruning**: Completely suppresses attacks that result in tied or negative attrition trades, while instantly prioritizing decisive breaches and deathtouch vectors.

## 4. How to Test
Run the autonomous tester:
```bash
python3 tester.py
```
Or validate using the organizer evaluator:
```bash
python3 ../evaluator/evaluate.py validate bot.py
```
