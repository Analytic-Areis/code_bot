"""Code Bot Participant: Model 20 - KG Hyper-Aggressive Rusher
Bot: KGRush
Strategy: Hyper-Aggressive Early Rusher with Fast Opening & Minimal Defensive Margins
Sources: 4 Never test against reference.txt (line 1754)
AI assistance: Claude & Gemini pair programming

Implements kg_rush:
1. Fast opening: moves out immediately as soon as army >= 6 (no waiting for 13).
2. Minimal margins: commits all available troops to rapid scouting tendrils.
3. Decapitation rush: marches straight down shortest path toward candidate general cluster.
4. Capitalizes on early opponents who leave their generals undefended (3-9 army).
"""
import time
from collections import deque

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_my_general = None
_opp_general = None
_target_candidate = None


def _reset():
    global _my_general, _opp_general, _target_candidate
    _my_general = None
    _opp_general = None
    _target_candidate = None


def _find_my_general(obs):
    owner, terrain = obs["owner"], obs["type"]
    for r in range(obs["height"]):
        for c in range(obs["width"]):
            if owner[r][c] == 1 and terrain[r][c] == 4:
                return (r, c)
    return None


def act(obs):
    t_start = time.perf_counter()
    global _my_general, _opp_general, _target_candidate

    try:
        h, w = obs["height"], obs["width"]
        owner, army, terrain = obs["owner"], obs["army"], obs["type"]
        turn = obs["turn"]

        if _my_general is None or turn <= 1:
            _my_general = _find_my_general(obs)
            if _my_general:
                # Opposite quadrant as initial rush vector
                gr, gc = _my_general
                _target_candidate = (h - 1 - gr, w - 1 - gc)

        gr, gc = _my_general if _my_general else (h // 2, w // 2)

        # 1. Spot opponent general
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 2 and terrain[r][c] == 4:
                    _opp_general = (r, c)
                    break
            if _opp_general:
                break

        target = _opp_general if _opp_general else _target_candidate
        if not target:
            target = (h - 1, w - 1)

        # 2. Lethal win check
        if _opp_general:
            eg_r, eg_c = _opp_general
            for d_idx, (dr, dc) in enumerate(DIRS):
                fr, fc = eg_r + dr, eg_c + dc
                if 0 <= fr < h and 0 <= fc < w and owner[fr][fc] == 1 and army[fr][fc] >= 2:
                    rev_dir = d_idx ^ 1
                    if turn >= 800 or army[fr][fc] - 1 > army[eg_r][eg_c]:
                        return [0, int(fr), int(fc), int(rev_dir), 0]

        # 3. Hyper-Aggressive Early Opening (Turn <= 40, opens at army >= 6)
        if turn <= 40 and army[gr][gc] >= 6:
            # Thrust straight toward target
            best_dir = None
            best_d = 999
            for d_idx, (dr, dc) in enumerate(DIRS):
                nr, nc = gr + dr, gc + dc
                if 0 <= nr < h and 0 <= nc < w and terrain[nr][nc] != 2:
                    d = abs(nr - target[0]) + abs(nc - target[1])
                    if d < best_d:
                        best_d = d
                        best_dir = d_idx
            if best_dir is not None:
                return [0, int(gr), int(gc), int(best_dir), 0]

        # 4. March Forward: Find largest stack and push toward target
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
                        d_curr = abs(r - target[0]) + abs(c - target[1])
                        d_next = abs(nr - target[0]) + abs(nc - target[1])

                        # Aggression gradient
                        score += (d_curr - d_next) * 25.0
                        if owner[nr][nc] == 2:
                            score += 80.0  # attack enemy units
                        elif owner[nr][nc] == 0:
                            score += 30.0

                        score += army[r][c] * 0.4
                        if score > best_score:
                            best_score = score
                            best_move = [0, int(r), int(c), int(d_idx), 0]

        if best_move:
            return best_move

        return [1, 0, 0, 0, 0]

    except Exception:
        return [1, 0, 0, 0, 0]
