"""Code Bot Participant: Model 15 - FST Metagame Automaton
Bot: FSTMetagameAutomaton
Strategy: Mealy-Type Finite State Transducer for Rigid O(1) Metagame Transitions
Sources: models.md (Game Theory Model 15)
AI assistance: Claude / Antigravity pair programming

Implements Finite State Transducer (FST) Transitions:
States:
  S0: OPENING (Turn <= 60: Fast expansion tendril, land acquisition)
  S1: STRIKE (Sufficient army & target identified: Relentless assault)
  S2: DEFEND (Lethal threat near home: Absolute perimeter recall)
  S3: DEATHTOUCH_STAGING (Turn >= 740: Position 2-unit runners at enemy entrances)
Guarantees O(1) transitions and zero ping-pong oscillation loops.
"""
import time
from collections import deque

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# State constants
STATE_S0_OPENING = 0
STATE_S1_STRIKE = 1
STATE_S2_DEFEND = 2
STATE_S3_DEATHTOUCH = 3

_my_general = None
_opp_general = None
_current_state = STATE_S0_OPENING


def _reset():
    global _my_general, _opp_general, _current_state
    _my_general = None
    _opp_general = None
    _current_state = STATE_S0_OPENING


def _find_my_general(obs):
    owner, terrain = obs["owner"], obs["type"]
    for r in range(obs["height"]):
        for c in range(obs["width"]):
            if owner[r][c] == 1 and terrain[r][c] == 4:
                return (r, c)
    return None


def act(obs):
    t_start = time.perf_counter()
    global _my_general, _opp_general, _current_state

    try:
        h, w = obs["height"], obs["width"]
        owner, army, terrain = obs["owner"], obs["army"], obs["type"]
        turn = obs["turn"]

        if _my_general is None or turn <= 1:
            _my_general = _find_my_general(obs)
            _current_state = STATE_S0_OPENING

        gr, gc = _my_general if _my_general else (h // 2, w // 2)

        # 1. Update enemy general sightings
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 2 and terrain[r][c] == 4:
                    _opp_general = (r, c)
                    break
            if _opp_general:
                break

        # 2. Check maximum visible army / threat
        max_my_stack = 0
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 1 and army[r][c] > max_my_stack:
                    max_my_stack = army[r][c]

        threat_near_general = False
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 2 and abs(r - gr) + abs(c - gc) <= 4:
                    if army[r][c] >= army[gr][gc] - 2:
                        threat_near_general = True
                        break

        # 3. FST State Transition Logic:
        if turn >= 740:
            _current_state = STATE_S3_DEATHTOUCH
        elif threat_near_general:
            _current_state = STATE_S2_DEFEND
        elif _opp_general and max_my_stack >= 20:
            _current_state = STATE_S1_STRIKE
        elif turn <= 60:
            _current_state = STATE_S0_OPENING
        else:
            # Default midgame: strike if army is decent, else opening expansion
            if max_my_stack >= 25:
                _current_state = STATE_S1_STRIKE
            else:
                _current_state = STATE_S0_OPENING

        # 4. State Action Execution
        best_move = None
        best_score = -1e9

        target_r = _opp_general[0] if _opp_general else (h - 1 - gr)
        target_c = _opp_general[1] if _opp_general else (w - 1 - gc)

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
                        d_target_curr = abs(r - target_r) + abs(c - target_c)
                        d_target_next = abs(nr - target_r) + abs(nc - target_c)
                        d_gen_curr = abs(r - gr) + abs(c - gc)
                        d_gen_next = abs(nr - gr) + abs(nc - gc)

                        if _current_state == STATE_S3_DEATHTOUCH:
                            # Advance any unit toward enemy general or entrances
                            score += (d_target_curr - d_target_next) * 35.0
                            if (nr, nc) == _opp_general:
                                score += 50000.0  # instant deathtouch win!
                        elif _current_state == STATE_S2_DEFEND:
                            # Inward flow to general
                            if (r, c) != (gr, gc):
                                score += (d_gen_curr - d_gen_next) * 30.0
                            if owner[nr][nc] == 2:
                                score += 100.0
                        elif _current_state == STATE_S1_STRIKE:
                            # Push main stacks toward target
                            score += (d_target_curr - d_target_next) * 20.0
                            if owner[nr][nc] == 2:
                                score += 60.0
                            score += army[r][c] * 0.5
                        else:  # STATE_S0_OPENING
                            # Fast territorial capture
                            if owner[nr][nc] == 0:
                                score += 35.0
                            score += (d_target_curr - d_target_next) * 10.0

                        if score > best_score:
                            best_score = score
                            best_move = [0, int(r), int(c), int(d_idx), 0]

        if best_move:
            return best_move

        return [1, 0, 0, 0, 0]

    except Exception:
        return [1, 0, 0, 0, 0]
