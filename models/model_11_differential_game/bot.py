"""Code Bot Participant: Model 11 - Differential Game Solver
Bot: IsaacsDifferentialStrategist
Strategy: Continuous Differential Game Calculus for Optimal Engagement Timing
Sources: models.md (Game Theory Model 11)
AI assistance: Claude / Antigravity pair programming

Implements Continuous Differential Games (Isaacs' Framework):
1. Tracks rolling first derivatives of army production: dA/dt (friendly) vs dE/dt (enemy).
2. If dA/dt > dE/dt: Macro-delay policy (widen economic advantage, expand smoothly).
3. If dE/dt > dA/dt: Solves optimal stopping time to execute immediate preemptive assault.
4. Seamlessly adapts engagement stance based on dynamic scoreboard differential calculus.
"""
import time
from collections import deque

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_my_general = None
_army_history = deque(maxlen=10)
_opp_army_history = deque(maxlen=10)


def _reset():
    global _my_general, _army_history, _opp_army_history
    _my_general = None
    _army_history.clear()
    _opp_army_history.clear()


def _find_my_general(obs):
    owner, terrain = obs["owner"], obs["type"]
    for r in range(obs["height"]):
        for c in range(obs["width"]):
            if owner[r][c] == 1 and terrain[r][c] == 4:
                return (r, c)
    return None


def act(obs):
    t_start = time.perf_counter()
    global _my_general, _army_history, _opp_army_history

    try:
        h, w = obs["height"], obs["width"]
        owner, army, terrain = obs["owner"], obs["army"], obs["type"]

        if _my_general is None or obs["turn"] <= 1:
            _my_general = _find_my_general(obs)

        # 1. Update rolling scoreboard histories
        _army_history.append(obs["my_army"])
        _opp_army_history.append(obs["opp_army"])

        # Compute growth rate derivatives (over sliding window)
        if len(_army_history) >= 2:
            dA_dt = (_army_history[-1] - _army_history[0]) / (len(_army_history) - 1)
            dE_dt = (_opp_army_history[-1] - _opp_army_history[0]) / (len(_opp_army_history) - 1)
        else:
            dA_dt = 1.0
            dE_dt = 1.0

        # Mode determination from differential equation:
        # If dE_dt > dA_dt: PREEMPTIVE ASSAULT (stop them before they pull ahead)
        # If dA_dt >= dE_dt: MACRO DELAY (expand, stall, compound advantage)
        preemptive_strike = (dE_dt > dA_dt + 0.1) or (obs["turn"] >= 750)

        # 2. Check if enemy general is spotted
        opp_gen = None
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 2 and terrain[r][c] == 4:
                    opp_gen = (r, c)
                    break
            if opp_gen:
                break

        # 3. Action selection
        best_move = None
        best_score = -1e9

        dest_r = opp_gen[0] if opp_gen else (h - 1 - _my_general[0] if _my_general else h - 1)
        dest_c = opp_gen[1] if opp_gen else (w - 1 - _my_general[1] if _my_general else w - 1)

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
                        d_curr = abs(r - dest_r) + abs(c - dest_c)
                        d_next = abs(nr - dest_r) + abs(nc - dest_c)

                        if preemptive_strike:
                            # Highly aggressive pathing toward enemy territory
                            score += (d_curr - d_next) * 25.0
                            if owner[nr][nc] == 2:
                                score += 80.0
                            elif owner[nr][nc] == 0:
                                score += 15.0
                        else:
                            # Macro expansion policy
                            if owner[nr][nc] == 0:
                                score += 35.0  # land acquisition priority
                            elif owner[nr][nc] == 2:
                                score += 20.0
                            score += (d_curr - d_next) * 5.0

                        score += army[r][c] * 0.3
                        if score > best_score:
                            best_score = score
                            best_move = [0, int(r), int(c), int(d_idx), 0]

        if best_move:
            return best_move

        return [1, 0, 0, 0, 0]

    except Exception:
        return [1, 0, 0, 0, 0]
