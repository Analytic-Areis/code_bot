"""Code Bot Participant: Model 22 - The Pinner Exploit
Bot: PinnerExploit
Strategy: Danger-Logic Exploit - Parks Threat 3 Cells Away to Freeze Opponent Defense
Sources: 4 Never test against reference.txt (line 1756)
AI assistance: Claude & Gemini pair programming

Implements pinner:
1. Locates enemy general.
2. Marches a sizeable stack to park 3 cells away, visible but intentionally not attacking.
3. Exploits opponent's danger/threat heuristic, keeping opponent frozen in home defense.
4. Freely captures remaining neutral territory across the map while opponent is paralyzed.
5. Commits to decapitation strike when overwhelming or at turn 800 via deathtouch.
"""
import time
from collections import deque

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_my_general = None
_opp_general = None
_pin_location = None


def _reset():
    global _my_general, _opp_general, _pin_location
    _my_general = None
    _opp_general = None
    _pin_location = None


def _find_my_general(obs):
    owner, terrain = obs["owner"], obs["type"]
    for r in range(obs["height"]):
        for c in range(obs["width"]):
            if owner[r][c] == 1 and terrain[r][c] == 4:
                return (r, c)
    return None


def act(obs):
    t_start = time.perf_counter()
    global _my_general, _opp_general, _pin_location

    try:
        h, w = obs["height"], obs["width"]
        owner, army, terrain = obs["owner"], obs["army"], obs["type"]
        turn = obs["turn"]

        if _my_general is None or turn <= 1:
            _my_general = _find_my_general(obs)

        gr, gc = _my_general if _my_general else (h // 2, w // 2)

        # 1. Spot enemy general
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 2 and terrain[r][c] == 4:
                    _opp_general = (r, c)
                    break
            if _opp_general:
                break

        # 2. Deathtouch or decisive victory check
        if _opp_general:
            eg_r, eg_c = _opp_general
            for d_idx, (dr, dc) in enumerate(DIRS):
                fr, fc = eg_r + dr, eg_c + dc
                if 0 <= fr < h and 0 <= fc < w and owner[fr][fc] == 1 and army[fr][fc] >= 2:
                    rev_dir = d_idx ^ 1
                    # Strike if turn >= 800 or crushing advantage (> 2x)
                    if turn >= 800 or army[fr][fc] - 1 > army[eg_r][eg_c] * 1.5:
                        return [0, int(fr), int(fc), int(rev_dir), 0]

        # 3. Pinning Strategy Logic:
        # If enemy general is located, select a pinning post exactly Manhattan distance 3 away
        if _opp_general:
            eg_r, eg_c = _opp_general
            if not _pin_location:
                # Find passable tile at distance 3
                for dr in range(-3, 4):
                    for dc in range(-3, 4):
                        if abs(dr) + abs(dc) == 3:
                            pr, pc = eg_r + dr, eg_c + dc
                            if 0 <= pr < h and 0 <= pc < w and terrain[pr][pc] != 2:
                                _pin_location = (pr, pc)
                                break
                    if _pin_location:
                        break

            # If pin location is chosen, route a vanguard stack there to maintain the pin
            if _pin_location:
                pr, pc = _pin_location
                # If we don't own pin location or army on it is small, send troops to it
                if owner[pr][pc] != 1 or army[pr][pc] < 15:
                    best_move = None
                    best_dist = 999
                    for r in range(h):
                        for c in range(w):
                            if owner[r][c] == 1 and army[r][c] >= 2 and (r, c) != _pin_location:
                                for d_idx, (dr, dc) in enumerate(DIRS):
                                    nr, nc = r + dr, c + dc
                                    if 0 <= nr < h and 0 <= nc < w and terrain[nr][nc] != 2:
                                        d = abs(nr - pr) + abs(nc - pc)
                                        if d < best_dist:
                                            best_dist = d
                                            best_move = [0, int(r), int(c), int(d_idx), 0]
                    if best_move:
                        return best_move

        # 4. While enemy is pinned, freely expand across the rest of the map!
        best_act = None
        best_score = -1e9

        for r in range(h):
            for c in range(w):
                if time.perf_counter() - t_start > 0.120:
                    break
                # Do not move the pinning stack away once it is in place!
                if _pin_location and (r, c) == _pin_location and turn < 800:
                    continue

                if owner[r][c] == 1 and army[r][c] >= 2:
                    for d_idx, (dr, dc) in enumerate(DIRS):
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < h and 0 <= nc < w) or terrain[nr][nc] == 2:
                            continue

                        score = 0.0
                        if owner[nr][nc] == 0:
                            score += 35.0  # free macro land grab
                        elif owner[nr][nc] == 2:
                            score += 20.0

                        score += army[r][c] * 0.2
                        if score > best_score:
                            best_score = score
                            best_act = [0, int(r), int(c), int(d_idx), 0]

        if best_act:
            return best_act

        return [1, 0, 0, 0, 0]

    except Exception:
        return [1, 0, 0, 0, 0]
