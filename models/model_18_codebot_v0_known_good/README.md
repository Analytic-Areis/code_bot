# Model 18: CodeBot v0 Known Good (Official Hackathon Baseline)

## 1. Overview & Paradigm
- **Archetype**: Tournament Known-Good Baseline
- **Mathematical Framework**: 8-Step Heuristic Priority Cascade with Exact 7-Step Room Filter
- **Bot Name**: `CodeBotV0KnownGood`
- **File**: `bot.py`
- **Test Suite**: `tester.py`
- **Source**: `4 Never test against reference.txt` (Milestone v0, lines 1125–1322)

## 2. Theoretical Principles
Documented directly in the hackathon development logs, `CodeBot v0` is the verified benchmark entry that achieved an 80%+ win rate against `Hunter` and sweeps `Starter` and `Expander`:
1. **Turn 0 Topological Knowledge**: All permanent mountains show as type 5 at tick 0 (even in fog).
2. **Spawn-Fairness Room Filter**: Computes $room(p) = \text{passable cells within 7 BFS steps of } p$. Filters candidates where $\text{dist} \ge 17$ and $|room(c) - room(\text{ours})| \le 5$, narrowing the candidate space to an average of 28 cells at tick 0.
3. **8-Priority Decision Cascade**:
   - **Priority 1: Win Now** — Strikes adjacent enemy general if superior or if $t \ge 800$.
   - **Priority 2: Deathtouch Defense ($t \ge 799$)** — Executes chase from an entrance-adjacent friendly tile to strip the attacker's source tile.
   - **Priority 3: Real Danger Defense** — Pulls army home along paths arriving before threat ETA: $\text{excess} = (A_{\text{threat}} - 1) - (A_{\text{gen}} + d/2)$.
   - **Priority 4: Intercept** — Attacks visible enemy stacks of $3+$ when $A_{\text{my}} - 1 > A_{\text{opp}}$.
   - **Priority 5: Opening ($t \le 50$)** — Waits for 13 army on general, then streams outward in thin lines toward the nearest candidate.
   - **Priority 6: Strike** — Marches main stack when $A \ge A_{\text{opp\_gen}} + 8$.
   - **Priority 7: Mod-50 Land Capture ($t \pmod{50} \ge 38$)** — Seizes neutral plains before the round boundary.
   - **Priority 8: Scored-Path Gather** — Pulls inward along high-density branches; crucially, **gathering never drains the general**, permanently eliminating the ping-pong bug.

## 3. How to Test
Run the autonomous tester:
```bash
python3 tester.py
```
Or validate using the organizer evaluator:
```bash
python3 ../evaluator/evaluate.py validate bot.py
```
