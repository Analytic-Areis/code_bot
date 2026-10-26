# Model 22: The Pinner Exploit (`pinner`)

## 1. Overview & Paradigm
- **Archetype**: Adversarial Exploitation Model / Threat-Psychology Pinner
- **Mathematical Framework**: Threat Gating Exploit via Perpetual Danger Induction
- **Bot Name**: `PinnerExploit`
- **File**: `bot.py`
- **Test Suite**: `tester.py`
- **Source**: `4 Never test against reference.txt` (Line 1756)

## 2. Theoretical Principles
`pinner` directly exploits standard heuristic danger evaluations programmed into competitor bots:
- Most advanced bots (including `Human.exe` derivatives and `v0`) compute danger as:
  $$\text{excess} = (A_{\text{threat}} - 1) - (A_{\text{gen}} + d/2)$$
  Whenever $\text{excess} > 0$ within a distance threshold ($d \le 8$), the defending bot freezes all outward offensive maneuvers and enters an infinite inward-gathering defensive loop.
- **The Exploit**: `pinner` marches a medium-to-large stack to an observation tile exactly Manhattan distance 3 away from the enemy general. It intentionally idles there in plain sight without committing to combat.
- **Economic Consequence**: The opponent remains perpetually terrified, holding all its armies uselessly on its home general. Meanwhile, `pinner` freely expands across the remaining 80% of the map, compound-harvesting the 50-tick land bonus until it either launches an overwhelming multi-hundred army sweep or slips in a 2-unit deathtouch strike at Turn 800.

## 3. How to Test
Run the autonomous tester:
```bash
python3 tester.py
```
Or validate using the organizer evaluator:
```bash
python3 ../evaluator/evaluate.py validate bot.py
```
