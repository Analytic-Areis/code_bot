"""Code Bot Participant: Model 3 - Asymmetric Colonel Blotto Breacher
Bot: BlottoBreacher
Strategy: Asymmetric Colonel Blotto Allocation for General Defense Penetration
Sources: models.md (Game Theory Model 3)
AI assistance: Claude / Antigravity pair programming

Implements Asymmetric Colonel Blotto Breaching:
1. Scouts for enemy general or estimates its location.
2. Stages a concentrated strike force A hidden 1-2 steps outside enemy vision.
3. Evaluates orthogonal entrances K (K <= 4) around enemy general.
4. Strikes the weakest entrance with A > D/K to breach localized defense.
"""
import time
from collections import deque

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_my_general = None
_opp_general = None


def _reset():
    global _my_general, _opp_general
    _my_general = None
    _opp_general = None


def _find_my_general(obs):
    owner, terrain = obs["owner"], obs["type"]
    for r in range(obs["height"]):
        for c in range(obs["width"]):
            if owner[r][c] == 1 and terrain[r][c] == 4:
                return (r, c)
    return None


def act(obs):
    t_start = time.perf_counter()
    global _my_general, _opp_general

    try:
        h, w = obs["height"], obs["width"]
        owner, army, terrain = obs["owner"], obs["army"], obs["type"]

        if _my_general is None or obs["turn"] <= 1:
            _my_general = _find_my_general(obs)

        # 1. Detect enemy general
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 2 and terrain[r][c] == 4:
                    _opp_general = (r, c)
                    break
            if _opp_general:
                break

        # 2. If enemy general is located, execute Asymmetric Colonel Blotto
        if _opp_general:
            gr, gc = _opp_general
            entrances = []
            for d_idx, (dr, dc) in enumerate(DIRS):
                nr, nc = gr + dr, gc + dc
                if 0 <= nr < h and 0 <= nc < w and terrain[nr][nc] != 2:
                    # Entrance defense: known army if visible, else estimate
                    ent_defense = army[nr][nc] if owner[nr][nc] == 2 else 0
                    entrances.append((ent_defense, nr, nc, d_idx))

            # Find weakest entrance
            if entrances:
                entrances.sort(key=lambda x: x[0])
                target_def, tr, tc, ent_dir = entrances[0]

                # If we have a friendly stack adjacent to enemy general, strike general directly!
                for d_idx, (dr, dc) in enumerate(DIRS):
                    fr, fc = gr + dr, gc + dc
                    if 0 <= fr < h and 0 <= fc < w and owner[fr][fc] == 1 and army[fr][fc] >= 2:
                        rev_dir = d_idx ^ 1
                        # If deathtouch active (turn >= 800) or we outnumber general
                        if obs["turn"] >= 800 or army[fr][fc] - 1 > army[gr][gc]:
                            return [0, int(fr), int(fc), int(rev_dir), 0]

                # If we own an entrance, push from that entrance or move our main army to it
                best_move = None
                max_stack = 0
                for r in range(h):
                    for c in range(w):
                        if owner[r][c] == 1 and army[r][c] > max_stack:
                            max_stack = army[r][c]
                            # Move toward target entrance (tr, tc)
                            for d_idx, (dr, dc) in enumerate(DIRS):
                                nr, nc = r + dr, c + dc
                                if 0 <= nr < h and 0 <= nc < w and terrain[nr][nc] != 2:
                                    if abs(nr - tr) + abs(nc - tc) < abs(r - tr) + abs(c - tc):
                                        best_move = [0, int(r), int(c), int(d_idx), 0]

                if best_move and max_stack >= 2:
                    return best_move

        # 3. Exploration / expansion when enemy general not directly breached
        # Move largest available stack outward toward opposite corner
        dest_r = h - 1 - _my_general[0] if _my_general else h - 1
        dest_c = w - 1 - _my_general[1] if _my_general else w - 1

        best_score = -1e9
        best_act = None

        for r in range(h):
            for c in range(w):
                if owner[r][c] == 1 and army[r][c] >= 2:
                    for d_idx, (dr, dc) in enumerate(DIRS):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and terrain[nr][nc] != 2:
                            score = army[r][c]
                            if owner[nr][nc] != 1:
                                score += 20
                            # Heading toward enemy quadrant
                            d_curr = abs(r - dest_r) + abs(c - dest_c)
                            d_next = abs(nr - dest_r) + abs(nc - dest_c)
                            score += (d_curr - d_next) * 15

                            if score > best_score:
                                best_score = score
                                best_act = [0, int(r), int(c), int(d_idx), 0]

        if best_act:
            return best_act

        return [1, 0, 0, 0, 0]

    except Exception:
        return [1, 0, 0, 0, 0]
