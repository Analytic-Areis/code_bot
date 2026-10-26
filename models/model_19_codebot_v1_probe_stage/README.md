# Model 19: CodeBot v1 Probe & Stage (Deathtouch Staging & Probing)

## 1. Overview & Paradigm
- **Archetype**: Tournament Advanced Evolution
- **Mathematical Framework**: Deathtouch Staging, Active Probing, Half-Launch & Adaptive Castle Gating
- **Bot Name**: `CodeBotV1ProbeStage`
- **File**: `bot.py`
- **Test Suite**: `tester.py`
- **Source**: `4 Never test against reference.txt` (Milestone v1/v1b, lines 1588–1730)

## 2. Theoretical Principles & Advancements over v0
`CodeBot v1` directly resolves the specific edge cases discovered during 40-game benchmark traces against `Turtle` and `Hunter`:
1. **Deathtouch Dual-Entrance Staging ($t \ge 740$)**: In the endgame approaching Turn 800, pre-positions stacks of $\ge 2$ units onto at least two distinct orthogonal entrances around the enemy general. When Turn 800 strikes, the defender cannot block both entrances simultaneously.
2. **Deathtouch Entrance Fortification ($t \ge 760$)**: Splits armies to ensure all friendly general entrances are fortified by friendly units before the deathtouch deadline.
3. **The PROBE Rule (Never Pass)**: Instead of passing and idling when the main general stack cannot safely move out, the bot automatically paths the largest free external stack toward the candidate destination as a continuous probe, uncovering fog and maintaining tactical momentum.
4. **HALF_LAUNCH from General**: Allows the general to deploy half its garrison (`split=1`) if the remaining half exceeds the visible threat excess.
5. **Adaptive Castle Gate**: Builds castles only when $t \ge 300$ or when the enemy general garrison $\ge 40$, positioned strictly at Manhattan distance $\ge 7$.

## 3. How to Test
Run the autonomous tester:
```bash
python3 tester.py
```
Or validate using the organizer evaluator:
```bash
python3 ../evaluator/evaluate.py validate bot.py
```
