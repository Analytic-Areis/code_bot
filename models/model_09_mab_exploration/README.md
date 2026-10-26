# Model 09: Multi-Armed Bandit (MAB) for Fog Exploration

## 1. Overview & Paradigm
- **Archetype**: Strategy III - Balanced
- **Mathematical Framework**: Multi-Armed Bandit (UCB1) Exploration-Exploitation
- **Bot Name**: `UCB1FogBandit`
- **File**: `bot.py`
- **Test Suite**: `tester.py`

## 2. Theoretical Principles
Exploration of fogged tiles represents a classic exploration vs exploitation dilemma:
- Exploitation captures known, safe neutral tiles to accumulate land count.
- Exploration pushes into uncharted fog to locate the enemy general.
We partition the board into operational sectors (bandit arms) and apply the Upper Confidence Bound (UCB1) algorithm:
$$\text{Sector}^* = \arg\max_i \left( \mu_i + C \sqrt{\frac{\ln t}{N_i}} \right)$$
where:
- $\mu_i$ is the empirical reward of sector $i$ (fog cleared, enemy structures sighted).
- $N_i$ is the visit count for sector $i$.
- $C = \sqrt{2}$ balances exploration uncertainty against known value.
- $t$ is the current turn number.

## 3. Algorithmic Implementation
- **Sector Partitioning**: Dynamically segments the 2D grid into geographic quadrants/arms.
- **Arm Scoring & Selection**: Updates empirical rewards and visit counts per turn, directing the scout stack toward the sector with the maximal UCB1 score.
- **Target Shift**: As soon as the enemy general is uncovered, the bot bypasses exploration and routes directly for the win condition.

## 4. How to Test
Run the autonomous tester:
```bash
python3 tester.py
```
Or validate using the organizer evaluator:
```bash
python3 ../evaluator/evaluate.py validate bot.py
```
