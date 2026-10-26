"""Code Bot Participant: Model 18 - CodeBot v0 Known Good
Bot: CodeBotV0KnownGood
Strategy: Hackathon Verified v0 Engine - Priority List with Exact Room-Fairness & Safe Gather
Sources: 4 Never test against reference.txt (lines 1125-1322)
AI assistance: Claude & Gemini pair programming

Priority list:
1. Win now (beat enemy general or deathtouch turn >= 800)
2. Deathtouch defense (turn >= 799 chase from entrance neighbor)
3. Real danger defense (pullback army arriving before threat ETA, never drain general)
4. Intercept (destroy visible 3+ enemy stacks where we win the trade)
5. Opening (wait for 13 army, stream in thin lines toward candidates)
6. Strike (march main stack >= enemy_general + 8 along shortest path)
7. Mod-50 land capture (turn % 50 >= 38)
8. Scored-path gather into main stack
"""
import time
from collections import deque

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT

_my_general = None
_opp_general = None
_mountains = set()
_candidates = set()
_initialized = False


def _reset():
    global _my_general, _opp_general, _mountains, _candidates, _initialized
    _my_general = None
    _opp_general = None
    _mountains = set()
    _candidates = set()
    _initialized = False


def _find_my_general(obs):
    owner, terrain = obs["owner"], obs["type"]
    for r in range(obs["height"]):
        for c in range(obs["width"]):
            if owner[r][c] == 1 and terrain[r][c] == 4:
                return (r, c)
    return None


def _bfs_room(h, w, mountains, start_r, start_c, max_depth=7):
    # Passable cells reachable within 7 BFS steps (excluding start)
    visited = { (start_r, start_c): 0 }
    q = deque([(start_r, start_c)])
    count = 0
    while q:
        r, c = q.popleft()
        d = visited[(r, c)]
        if d < max_depth:
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in mountains and (nr, nc) not in visited:
                    visited[(nr, nc)] = d + 1
                    count += 1
                    q.append((nr, nc))
    return count


def _bfs_distance(h, w, mountains, start_r, start_c):
    visited = { (start_r, start_c): 0 }
    q = deque([(start_r, start_c)])
    while q:
        r, c = q.popleft()
        d = visited[(r, c)]
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in mountains and (nr, nc) not in visited:
                visited[(nr, nc)] = d + 1
                q.append((nr, nc))
    return visited


def _init_candidates(obs):
    global _my_general, _mountains, _candidates, _initialized
    h, w = obs["height"], obs["width"]
    terrain = obs["type"]

    # At turn 0/1: all mountains are type 2, or type 5 (fogged mountains)
    _mountains = set()
    for r in range(h):
        for c in range(w):
            if terrain[r][c] in (2, 5):
                _mountains.add((r, c))

    gr, gc = _my_general
    my_room = _bfs_room(h, w, _mountains, gr, gc, max_depth=7)
    dists = _bfs_distance(h, w, _mountains, gr, gc)

    _candidates = set()
    for (r, c), d in dists.items():
        if d >= 17:
            cand_room = _bfs_room(h, w, _mountains, r, c, max_depth=7)
            if abs(cand_room - my_room) <= 5:
                _candidates.add((r, c))

    if not _candidates:
        # Fallback if maze/tight board
        for (r, c), d in dists.items():
            if d >= 14:
                _candidates.add((r, c))

    _initialized = True


def act(obs):
    t_start = time.perf_counter()
    global _my_general, _opp_general, _mountains, _candidates, _initialized

    try:
        h, w = obs["height"], obs["width"]
        owner, army, terrain = obs["owner"], obs["army"], obs["type"]
        turn = obs["turn"]

        if _my_general is None or turn <= 1 or not _initialized:
            _my_general = _find_my_general(obs)
            if _my_general:
                _init_candidates(obs)

        gr, gc = _my_general if _my_general else (h // 2, w // 2)

        # Update opponent general sightings
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 2 and terrain[r][c] == 4:
                    _opp_general = (r, c)
                    break
            if _opp_general:
                break

        # Remove seen candidates
        if not _opp_general and _candidates:
            for (cr, cc) in list(_candidates):
                if terrain[cr][cc] != 0:  # revealed
                    if not (owner[cr][cc] == 2 and terrain[cr][cc] == 4):
                        _candidates.discard((cr, cc))

        # ------------------------------------------------------------------
        # 1. WIN NOW: stack next to enemy general that can conquer it
        # ------------------------------------------------------------------
        if _opp_general:
            eg_r, eg_c = _opp_general
            for d_idx, (dr, dc) in enumerate(DIRS):
                fr, fc = eg_r + dr, eg_c + dc
                if 0 <= fr < h and 0 <= fc < w and owner[fr][fc] == 1 and army[fr][fc] >= 2:
                    rev_dir = d_idx ^ 1
                    if turn >= 800 or army[fr][fc] - 1 > army[eg_r][eg_c]:
                        return [0, int(fr), int(fc), int(rev_dir), 0]

        # ------------------------------------------------------------------
        # 2. DEATHTOUCH DEFENSE (turn >= 799): chase enemy stack off general entrance
        # ------------------------------------------------------------------
        if turn >= 799:
            for dr, dc in DIRS:
                er, ec = gr + dr, gc + dc
                if 0 <= er < h and 0 <= ec < w and owner[er][ec] == 2 and army[er][ec] >= 2:
                    # Enemy is on general's entrance! Chase from adjacent non-general tile
                    for d_idx, (dr2, dc2) in enumerate(DIRS):
                        fr, fc = er + dr2, ec + dc2
                        if 0 <= fr < h and 0 <= fc < w and (fr, fc) != (gr, gc):
                            if owner[fr][fc] == 1 and army[fr][fc] >= 2:
                                rev_dir = d_idx ^ 1
                                return [0, int(fr), int(fc), int(rev_dir), 0]
                    # If no other tile can chase, attack from general
                    if army[gr][gc] >= 2:
                        dir_idx = DIRS.index((dr, dc))
                        return [0, int(gr), int(gc), int(dir_idx), 0]

        # ------------------------------------------------------------------
        # 3. REAL DANGER DEFENSE: calculate excess and ETA
        # ------------------------------------------------------------------
        # BFS distances from general
        dist_from_gen = _bfs_distance(h, w, _mountains, gr, gc)
        worst_excess = 0
        threat_eta = None
        threat_pos = None

        for r in range(h):
            for c in range(w):
                if owner[r][c] == 2:
                    d = dist_from_gen.get((r, c), 999)
                    if 0 < d <= 10:
                        # General grows +1 army every 2 ticks
                        gen_growth = d // 2
                        excess = (army[r][c] - 1) - (army[gr][gc] + gen_growth)
                        if excess > worst_excess:
                            worst_excess = excess
                            threat_eta = d
                            threat_pos = (r, c)

        if worst_excess > 0 and threat_pos:
            # Danger detected! Pull army home arriving before ETA
            best_def_move = None
            best_gain = -999
            for r in range(h):
                for c in range(w):
                    if owner[r][c] == 1 and army[r][c] >= 2 and (r, c) != (gr, gc):
                        for d_idx, (dr, dc) in enumerate(DIRS):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in _mountains:
                                d_curr = dist_from_gen.get((r, c), 999)
                                d_next = dist_from_gen.get((nr, nc), 999)
                                if d_next < d_curr:
                                    gain = army[r][c]
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_def_move = [0, int(r), int(c), int(d_idx), 0]
            if best_def_move:
                return best_def_move

        # ------------------------------------------------------------------
        # 4. INTERCEPT: destroy visible 3+ enemy stacks when we win the trade
        # ------------------------------------------------------------------
        best_intercept = None
        best_int_val = 0
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 2 and army[r][c] >= 3:
                    for d_idx, (dr, dc) in enumerate(DIRS):
                        fr, fc = r + dr, c + dc
                        if 0 <= fr < h and 0 <= fc < w and owner[fr][fc] == 1 and (fr, fc) != (gr, gc):
                            if army[fr][fc] - 1 > army[r][c]:
                                val = army[r][c]
                                if val > best_int_val:
                                    best_int_val = val
                                    rev_dir = d_idx ^ 1
                                    best_intercept = [0, int(fr), int(fc), int(rev_dir), 0]
        if best_intercept:
            return best_intercept

        # ------------------------------------------------------------------
        # 5. OPENING (turns 0-50): wait for 13 on general, stream in thin lines
        # ------------------------------------------------------------------
        target_dest = _opp_general
        if not target_dest and _candidates:
            # Nearest candidate
            target_dest = min(_candidates, key=lambda cand: abs(cand[0] - gr) + abs(cand[1] - gc))
        if not target_dest:
            target_dest = (h - 1 - gr, w - 1 - gc)

        if turn <= 50:
            # If general has >= 13 army, push outward in thin line
            if army[gr][gc] >= 13:
                for d_idx, (dr, dc) in enumerate(DIRS):
                    nr, nc = gr + dr, gc + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in _mountains:
                        if abs(nr - target_dest[0]) + abs(nc - target_dest[1]) < abs(gr - target_dest[0]) + abs(gc - target_dest[1]):
                            return [0, int(gr), int(gc), int(d_idx), 0]

            # Extend outer tip of the line
            for r in range(h):
                for c in range(w):
                    if owner[r][c] == 1 and army[r][c] >= 2 and (r, c) != (gr, gc):
                        for d_idx, (dr, dc) in enumerate(DIRS):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in _mountains and owner[nr][nc] == 0:
                                return [0, int(r), int(c), int(d_idx), 0]

        # ------------------------------------------------------------------
        # 6. STRIKE: march main stack >= enemy_general + 8
        # ------------------------------------------------------------------
        max_stack = 0
        best_stack = None
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 1 and army[r][c] > max_stack:
                    max_stack = army[r][c]
                    best_stack = (r, c)

        opp_gen_army = army[_opp_general[0]][_opp_general[1]] if _opp_general else 15
        if best_stack and (max_stack >= opp_gen_army + 8 or max_stack >= 25 or turn >= 750):
            sr, sc = best_stack
            # Do not march general directly if danger is non-zero
            if (sr, sc) != (gr, gc) or worst_excess == 0:
                best_d = 999
                best_dir = None
                for d_idx, (dr, dc) in enumerate(DIRS):
                    nr, nc = sr + dr, sc + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in _mountains:
                        d = abs(nr - target_dest[0]) + abs(nc - target_dest[1])
                        if d < best_d:
                            best_d = d
                            best_dir = d_idx
                if best_dir is not None and army[sr][sc] >= 2:
                    return [0, int(sr), int(sc), int(best_dir), 0]

        # ------------------------------------------------------------------
        # 7. MOD-50 LAND CAPTURE: turn % 50 >= 38
        # ------------------------------------------------------------------
        if turn % 50 >= 38:
            for r in range(h):
                for c in range(w):
                    if owner[r][c] == 1 and army[r][c] >= 2:
                        for d_idx, (dr, dc) in enumerate(DIRS):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in _mountains and owner[nr][nc] == 0:
                                return [0, int(r), int(c), int(d_idx), 0]

        # ------------------------------------------------------------------
        # 8. SCORED-PATH GATHER into main stack (never drain general)
        # ------------------------------------------------------------------
        if best_stack:
            bs_r, bs_c = best_stack
            best_gather = None
            best_density = -1e9
            for r in range(h):
                for c in range(w):
                    if owner[r][c] == 1 and army[r][c] >= 2 and (r, c) != best_stack:
                        # Safety rule: gather never drains general!
                        if (r, c) == (gr, gc):
                            continue
                        for d_idx, (dr, dc) in enumerate(DIRS):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in _mountains:
                                d_curr = abs(r - bs_r) + abs(c - bs_c)
                                d_next = abs(nr - bs_r) + abs(nc - bs_c)
                                if d_next < d_curr:
                                    density = (army[r][c] - 1) / max(1, d_curr)
                                    if density > best_density:
                                        best_density = density
                                        best_gather = [0, int(r), int(c), int(d_idx), 0]
            if best_gather:
                return best_gather

        return [1, 0, 0, 0, 0]

    except Exception:
        return [1, 0, 0, 0, 0]
