"""Code Bot Participant: Model 13 - Bayesian Opponent Classifier
Bot: BayesianOpponentClassifier
Strategy: Real-Time Bayesian Classification of Opponent Strategy (Rusher/Turtle/Expander)
Sources: models.md (Game Theory Model 13)
AI assistance: Claude / Antigravity pair programming

Implements Bayesian Opponent Classification:
1. Maintains belief distribution over Theta in {Rusher, Turtle, Expander}.
2. Extracts real-time signals from scoreboard deltas and sightings.
3. Automatically transitions to the exact hard-counter strategy:
   - vs Rusher: Fortify general, absorb push, counter-strike.
   - vs Turtle: Macro-expand across map, stage endgame deathtouch.
   - vs Expander: Rapid concentrated spearhead piercing diffuse frontier.
"""
import time
from collections import deque

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_my_general = None
_beliefs = {"Rusher": 0.33, "Turtle": 0.33, "Expander": 0.34}
_opp_land_prev = 0
_opp_army_prev = 0


def _reset():
    global _my_general, _beliefs, _opp_land_prev, _opp_army_prev
    _my_general = None
    _beliefs = {"Rusher": 0.33, "Turtle": 0.33, "Expander": 0.34}
    _opp_land_prev = 0
    _opp_army_prev = 0


def _find_my_general(obs):
    owner, terrain = obs["owner"], obs["type"]
    for r in range(obs["height"]):
        for c in range(obs["width"]):
            if owner[r][c] == 1 and terrain[r][c] == 4:
                return (r, c)
    return None


def act(obs):
    t_start = time.perf_counter()
    global _my_general, _beliefs, _opp_land_prev, _opp_army_prev

    try:
        h, w = obs["height"], obs["width"]
        owner, army, terrain = obs["owner"], obs["army"], obs["type"]
        turn = obs["turn"]

        if _my_general is None or turn <= 1:
            _my_general = _find_my_general(obs)
            _opp_land_prev = obs["opp_land"]
            _opp_army_prev = obs["opp_army"]

        gr, gc = _my_general if _my_general else (h // 2, w // 2)

        # 1. Signal Extraction & Bayesian Belief Update
        delta_l = obs["opp_land"] - _opp_land_prev
        delta_a = obs["opp_army"] - _opp_army_prev
        _opp_land_prev = obs["opp_land"]
        _opp_army_prev = obs["opp_army"]

        # Signal 1: Early enemy units near home territory
        early_rush_sighted = False
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 2 and (abs(r - gr) + abs(c - gc) <= 6) and turn < 120:
                    early_rush_sighted = True
                    break

        if early_rush_sighted:
            _beliefs["Rusher"] += 0.4
            _beliefs["Expander"] -= 0.2
            _beliefs["Turtle"] -= 0.2

        # Signal 2: Rapid land expansion
        if delta_l >= 2 or (turn > 50 and obs["opp_land"] > obs["my_land"] + 5):
            _beliefs["Expander"] += 0.1
            _beliefs["Turtle"] -= 0.05
            _beliefs["Rusher"] -= 0.05

        # Signal 3: Sudden army drop (castle construction: -35 army with 0 delta land)
        if delta_a <= -30 and delta_l == 0:
            _beliefs["Turtle"] += 0.3
            _beliefs["Expander"] -= 0.15
            _beliefs["Rusher"] -= 0.15

        # Normalize probabilities
        for k in _beliefs:
            _beliefs[k] = max(0.05, _beliefs[k])
        total_p = sum(_beliefs.values())
        for k in _beliefs:
            _beliefs[k] /= total_p

        classified_type = max(_beliefs.items(), key=lambda x: x[1])[0]

        # 2. Counter-Strategy Execution
        best_move = None
        best_score = -1e9

        for r in range(h):
            for c in range(w):
                if time.perf_counter() - t_start > 0.120:
                    break
                if owner[r][c] == 1 and army[r][c] >= 2:
                    for d_idx, (dr, dc) in enumerate(DIRS):
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < h and 0 <= nc < w) or terrain[nr][nc] == 2:
                            continue

                        score = 0.0
                        d_gen = abs(nr - gr) + abs(nc - gc)

                        if classified_type == "Rusher":
                            # Defensive posture: keep troops near general
                            score += (10 - d_gen) * 5.0
                            if owner[nr][nc] == 2:
                                score += 80.0  # eliminate encroaching rusher
                        elif classified_type == "Turtle":
                            # Macro expansion: spread wide, avoid early turtle choke points
                            if owner[nr][nc] == 0:
                                score += 40.0
                            elif owner[nr][nc] == 1:
                                score += (d_gen) * 2.0  # push outward
                        else:  # "Expander"
                            # Spearhead assault: concentrate and pierce weak borders
                            if owner[nr][nc] == 2:
                                score += 60.0
                            elif owner[nr][nc] == 0:
                                score += 20.0

                        score += army[r][c] * 0.25
                        if score > best_score:
                            best_score = score
                            best_move = [0, int(r), int(c), int(d_idx), 0]

        if best_move:
            return best_move

        return [1, 0, 0, 0, 0]

    except Exception:
        return [1, 0, 0, 0, 0]
