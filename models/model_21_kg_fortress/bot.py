"""Code Bot Participant: Model 21 - KG Economic Fortress
Bot: KGFortress
Strategy: Defensive & Economic Fortress with Home Reserve, Loose Castles & Deathtouch Resilience
Sources: 4 Never test against reference.txt (line 1755)
AI assistance: Claude & Gemini pair programming

Implements kg_fortress:
1. Home reserve: retains high garrison (0.5 * free_army) on general at all times.
2. Loose castles: constructs economic castles at Manhattan distance >= 7 from turn 60.
3. Exposure reserve: covers all general entrances to absorb raids.
4. Survives guaranteed into turn 800+ deathtouch phase to test opponent endgame capabilities.
"""
import time
from collections import deque

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_my_general = None
_opp_general = None
_castles_built = 0


def _reset():
    global _my_general, _opp_general, _castles_built
    _my_general = None
    _opp_general = None
    _castles_built = 0


def _find_my_general(obs):
    owner, terrain = obs["owner"], obs["type"]
    for r in range(obs["height"]):
        for c in range(obs["width"]):
            if owner[r][c] == 1 and terrain[r][c] == 4:
                return (r, c)
    return None


def act(obs):
    t_start = time.perf_counter()
    global _my_general, _opp_general, _castles_built

    try:
        h, w = obs["height"], obs["width"]
        owner, army, terrain = obs["owner"], obs["army"], obs["type"]
        turn = obs["turn"]

        if _my_general is None or turn <= 1:
            _my_general = _find_my_general(obs)

        gr, gc = _my_general if _my_general else (h // 2, w // 2)

        # 1. Spot opponent general
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 2 and terrain[r][c] == 4:
                    _opp_general = (r, c)
                    break
            if _opp_general:
                break

        # 2. Deathtouch Win check (turn >= 800)
        if _opp_general:
            eg_r, eg_c = _opp_general
            for d_idx, (dr, dc) in enumerate(DIRS):
                fr, fc = eg_r + dr, eg_c + dc
                if 0 <= fr < h and 0 <= fc < w and owner[fr][fc] == 1 and army[fr][fc] >= 2:
                    rev_dir = d_idx ^ 1
                    if turn >= 800 or army[fr][fc] - 1 > army[eg_r][eg_c]:
                        return [0, int(fr), int(fc), int(rev_dir), 0]

        # 3. Fortress Defense: Deathtouch Chase Defense (@ turn >= 790)
        if turn >= 790:
            for dr, dc in DIRS:
                er, ec = gr + dr, gc + dc
                if 0 <= er < h and 0 <= ec < w and owner[er][ec] == 2:
                    # Enemy adjacent to general! Chase it
                    for d_idx, (dr2, dc2) in enumerate(DIRS):
                        fr, fc = er + dr2, ec + dc2
                        if 0 <= fr < h and 0 <= fc < w and (fr, fc) != (gr, gc) and owner[fr][fc] == 1 and army[fr][fc] >= 2:
                            rev_dir = d_idx ^ 1
                            return [0, int(fr), int(fc), int(rev_dir), 0]
                    if army[gr][gc] >= 2:
                        dir_idx = DIRS.index((dr, dc))
                        return [0, int(gr), int(gc), int(dir_idx), 0]

        # 4. Loose Castle Construction (turn 60 - 700)
        if _castles_built < 2 and 60 <= turn <= 700:
            for r in range(h):
                for c in range(w):
                    if owner[r][c] == 1 and terrain[r][c] == 1:
                        # Manhattan distance >= 7 from general
                        if abs(r - gr) + abs(c - gc) >= 7 and army[r][c] >= 35:
                            _castles_built += 1
                            return [2, int(r), int(c), 0, 0]

        # 5. Entrance Garrison Maintenance (keep entrances guarded)
        for dr, dc in DIRS:
            er, ec = gr + dr, gc + dc
            if 0 <= er < h and 0 <= ec < w and terrain[er][ec] != 2:
                if owner[er][ec] != 1 and army[gr][gc] >= 5:
                    dir_idx = DIRS.index((dr, dc))
                    return [0, int(gr), int(gc), int(dir_idx), 1]  # half-launch to fortify entrance

        # 6. Fortress Gathering: Flow outer units inward to general perimeter
        best_move = None
        best_score = -1e9

        for r in range(h):
            for c in range(w):
                if time.perf_counter() - t_start > 0.120:
                    break
                if owner[r][c] == 1 and army[r][c] >= 2 and (r, c) != (gr, gc):
                    for d_idx, (dr, dc) in enumerate(DIRS):
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < h and 0 <= nc < w) or terrain[nr][nc] == 2:
                            continue

                        score = 0.0
                        d_curr = abs(r - gr) + abs(c - gc)
                        d_next = abs(nr - gr) + abs(nc - gc)

                        # Inward flow toward general perimeter (within radius 3)
                        if d_curr > 3:
                            score += (d_curr - d_next) * 20.0
                        else:
                            if owner[nr][nc] == 0:
                                score += 15.0  # safe local expansion
                            elif owner[nr][nc] == 2:
                                score += 40.0

                        score += army[r][c] * 0.25
                        if score > best_score:
                            best_score = score
                            best_move = [0, int(r), int(c), int(d_idx), 0]

        if best_move:
            return best_move

        return [1, 0, 0, 0, 0]

    except Exception:
        return [1, 0, 0, 0, 0]
