"""Code Bot Participant: Model 19 - CodeBot v1 Probe & Stage
Bot: CodeBotV1ProbeStage
Strategy: Hackathon v1 Evolution - Deathtouch Dual Staging, Probing & Adaptive Castles
Sources: 4 Never test against reference.txt (lines 1588-1730)
AI assistance: Claude & Gemini pair programming

Key upgrades over v0:
1. Deathtouch Staging @740: stages 2+ stacks on 2 enemy entrances.
2. Deathtouch Fortification @760: secures our entrances before tick 800.
3. PROBE (never pass): marches biggest free stack toward target instead of idling.
4. HALF_LAUNCH: general sends half if remainder covers visible threat.
5. Adaptive Castle Gate: builds castles only when turn >= 300 or enemy general seen >= 40.
"""
import time
from collections import deque

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

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
                if terrain[cr][cc] != 0:
                    if not (owner[cr][cc] == 2 and terrain[cr][cc] == 4):
                        _candidates.discard((cr, cc))

        # ------------------------------------------------------------------
        # 1. WIN NOW: direct decapitation or deathtouch (turn >= 800)
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
        # 2. DEATHTOUCH DUAL-ENTRANCE STAGING (@ turn >= 740)
        # ------------------------------------------------------------------
        if _opp_general and 740 <= turn < 800:
            eg_r, eg_c = _opp_general
            opp_entrances = []
            for d_idx, (dr, dc) in enumerate(DIRS):
                nr, nc = eg_r + dr, eg_c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in _mountains:
                    opp_entrances.append((nr, nc))

            # Find friendly units adjacent to opponent entrances
            for er, ec in opp_entrances:
                if owner[er][ec] != 1:  # entrance not yet occupied by us
                    # Move adjacent unit onto it
                    for d_idx, (dr, dc) in enumerate(DIRS):
                        fr, fc = er + dr, ec + dc
                        if 0 <= fr < h and 0 <= fc < w and owner[fr][fc] == 1 and army[fr][fc] >= 2:
                            rev_dir = d_idx ^ 1
                            return [0, int(fr), int(fc), int(rev_dir), 0]

        # ------------------------------------------------------------------
        # 3. ENTRANCE FORTIFICATION (@ turn >= 760)
        # ------------------------------------------------------------------
        if turn >= 760:
            for dr, dc in DIRS:
                er, ec = gr + dr, gc + dc
                if 0 <= er < h and 0 <= ec < w and (er, ec) not in _mountains:
                    if owner[er][ec] != 1 and army[gr][gc] >= 4:
                        d_idx = DIRS.index((dr, dc))
                        return [0, int(gr), int(gc), int(d_idx), 1]  # half-split to fortify entrance

        # ------------------------------------------------------------------
        # 4. DEATHTOUCH CHASE DEFENSE (@ turn >= 799)
        # ------------------------------------------------------------------
        if turn >= 799:
            for dr, dc in DIRS:
                er, ec = gr + dr, gc + dc
                if 0 <= er < h and 0 <= ec < w and owner[er][ec] == 2 and army[er][ec] >= 2:
                    for d_idx, (dr2, dc2) in enumerate(DIRS):
                        fr, fc = er + dr2, ec + dc2
                        if 0 <= fr < h and 0 <= fc < w and (fr, fc) != (gr, gc):
                            if owner[fr][fc] == 1 and army[fr][fc] >= 2:
                                rev_dir = d_idx ^ 1
                                return [0, int(fr), int(fc), int(rev_dir), 0]
                    if army[gr][gc] >= 2:
                        dir_idx = DIRS.index((dr, dc))
                        return [0, int(gr), int(gc), int(dir_idx), 0]

        # ------------------------------------------------------------------
        # 5. DANGER & INTERCEPT
        # ------------------------------------------------------------------
        dist_from_gen = _bfs_distance(h, w, _mountains, gr, gc)
        worst_excess = 0
        threat_pos = None
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 2:
                    d = dist_from_gen.get((r, c), 999)
                    if 0 < d <= 10:
                        gen_growth = d // 2
                        excess = (army[r][c] - 1) - (army[gr][gc] + gen_growth)
                        if excess > worst_excess:
                            worst_excess = excess
                            threat_pos = (r, c)

        if worst_excess > 0 and threat_pos:
            best_def = None
            best_gain = -999
            for r in range(h):
                for c in range(w):
                    if owner[r][c] == 1 and army[r][c] >= 2 and (r, c) != (gr, gc):
                        for d_idx, (dr, dc) in enumerate(DIRS):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in _mountains:
                                if dist_from_gen.get((nr, nc), 999) < dist_from_gen.get((r, c), 999):
                                    if army[r][c] > best_gain:
                                        best_gain = army[r][c]
                                        best_def = [0, int(r), int(c), int(d_idx), 0]
            if best_def:
                return best_def

        # Intercept visible 3+
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 2 and army[r][c] >= 3:
                    for d_idx, (dr, dc) in enumerate(DIRS):
                        fr, fc = r + dr, c + dc
                        if 0 <= fr < h and 0 <= fc < w and owner[fr][fc] == 1 and (fr, fc) != (gr, gc):
                            if army[fr][fc] - 1 > army[r][c]:
                                rev_dir = d_idx ^ 1
                                return [0, int(fr), int(fc), int(rev_dir), 0]

        # ------------------------------------------------------------------
        # 6. ADAPTIVE CASTLE GATE: turn >= 300 or opp general army >= 40
        # ------------------------------------------------------------------
        opp_g_army = army[_opp_general[0]][_opp_general[1]] if _opp_general else 0
        if turn >= 300 or opp_g_army >= 40:
            for r in range(h):
                for c in range(w):
                    if owner[r][c] == 1 and terrain[r][c] == 1 and abs(r - gr) + abs(c - gc) >= 7:
                        if army[r][c] >= 35:
                            return [2, int(r), int(c), 0, 0]

        # ------------------------------------------------------------------
        # 7. TARGET SELECTION & STRIKE / HALF-LAUNCH
        # ------------------------------------------------------------------
        target_dest = _opp_general
        if not target_dest and _candidates:
            target_dest = min(_candidates, key=lambda cand: abs(cand[0] - gr) + abs(cand[1] - gc))
        if not target_dest:
            target_dest = (h - 1 - gr, w - 1 - gc)

        # Opening (turns 0-50)
        if turn <= 50 and army[gr][gc] >= 13:
            for d_idx, (dr, dc) in enumerate(DIRS):
                nr, nc = gr + dr, gc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in _mountains:
                    if abs(nr - target_dest[0]) + abs(nc - target_dest[1]) < abs(gr - target_dest[0]) + abs(gc - target_dest[1]):
                        return [0, int(gr), int(gc), int(d_idx), 0]

        # Main strike / Half-launch
        max_stack = 0
        best_stack = None
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 1 and army[r][c] > max_stack:
                    max_stack = army[r][c]
                    best_stack = (r, c)

        if best_stack and (max_stack >= opp_g_army + 8 or max_stack >= 25 or turn >= 740):
            sr, sc = best_stack
            split = 0
            if (sr, sc) == (gr, gc):
                # Half-launch check: send half if remainder covers visible threat
                if army[gr][gc] // 2 >= worst_excess + 5:
                    split = 1
                else:
                    best_stack = None  # keep general guarded

            if best_stack:
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
                    return [0, int(sr), int(sc), int(best_dir), split]

        # ------------------------------------------------------------------
        # 8. PROBE (NEVER PASS: march biggest free stack toward target)
        # ------------------------------------------------------------------
        probe_stack = None
        probe_max = 0
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 1 and army[r][c] > probe_max and (r, c) != (gr, gc):
                    probe_max = army[r][c]
                    probe_stack = (r, c)

        if probe_stack and probe_max >= 2:
            pr, pc = probe_stack
            best_d = 999
            best_dir = None
            for d_idx, (dr, dc) in enumerate(DIRS):
                nr, nc = pr + dr, pc + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in _mountains:
                    d = abs(nr - target_dest[0]) + abs(nc - target_dest[1])
                    if d < best_d:
                        best_d = d
                        best_dir = d_idx
            if best_dir is not None:
                return [0, int(pr), int(pc), int(best_dir), 0]

        return [1, 0, 0, 0, 0]

    except Exception:
        return [1, 0, 0, 0, 0]
