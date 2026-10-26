"""Code Bot Participant: Model 2 - PCST Army Gathering
Bot: SteinerTreeAggregator
Strategy: Prize-Collecting Steiner Tree Heuristic for Optimal Army Aggregation
Sources: models.md (Game Theory Model 2)
AI assistance: Claude / Antigravity pair programming

Implements Prize-Collecting Steiner Tree (PCST) Aggregation:
1. Treats owned tiles as prize nodes with prize = army - 1.
2. Identifies or maintains the designated frontline spearhead / root.
3. Rallies peripheral armies inward toward the root along highest-density tree paths.
4. Once aggregated, spearhead advances relentlessly into enemy territory.
"""
import time
from collections import deque

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_my_general = None
_spearhead = None


def _reset():
    global _my_general, _spearhead
    _my_general = None
    _spearhead = None


def _find_my_general(obs):
    owner, terrain = obs["owner"], obs["type"]
    for r in range(obs["height"]):
        for c in range(obs["width"]):
            if owner[r][c] == 1 and terrain[r][c] == 4:
                return (r, c)
    return None


def act(obs):
    t_start = time.perf_counter()
    global _my_general, _spearhead

    try:
        h, w = obs["height"], obs["width"]
        owner, army, terrain = obs["owner"], obs["army"], obs["type"]

        if _my_general is None or obs["turn"] <= 1:
            _my_general = _find_my_general(obs)

        # 1. Identify current spearhead: owned tile with highest army
        max_army = 0
        curr_root = _my_general
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 1 and army[r][c] > max_army:
                    max_army = army[r][c]
                    curr_root = (r, c)

        _spearhead = curr_root

        # 2. Check if enemy general is spotted
        opp_gen = None
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 2 and terrain[r][c] == 4:
                    opp_gen = (r, c)
                    break
            if opp_gen:
                break

        # 3. Decision phase:
        # If spearhead is sufficiently large (>= 20) or turn >= 800, spearhead pushes forward!
        # Otherwise, peripheral troops gather toward spearhead (PCST inward flow).
        if _spearhead and (army[_spearhead[0]][_spearhead[1]] >= 25 or obs["turn"] >= 800):
            # Attack vector from spearhead
            sr, sc = _spearhead
            target = opp_gen
            if not target:
                # Target nearest enemy tile or nearest unexplored fog
                min_d = 999
                for r in range(h):
                    for c in range(w):
                        if owner[r][c] == 2 or (owner[r][c] == 0 and terrain[r][c] in (0, 1)):
                            d = abs(r - sr) + abs(c - sc)
                            if d < min_d:
                                min_d = d
                                target = (r, c)

            if target:
                # Step toward target
                tr, tc = target
                best_dir = None
                best_d = 999
                for d_idx, (dr, dc) in enumerate(DIRS):
                    nr, nc = sr + dr, sc + dc
                    if 0 <= nr < h and 0 <= nc < w and terrain[nr][nc] != 2:
                        d = abs(nr - tr) + abs(nc - tc)
                        if d < best_d:
                            best_d = d
                            best_dir = d_idx
                if best_dir is not None and army[sr][sc] >= 2:
                    return [0, int(sr), int(sc), int(best_dir), 0]

        # 4. PCST Gathering: Flow peripheral armies inward toward _spearhead
        if _spearhead:
            sr, sc = _spearhead
            # BFS distance tree rooted at _spearhead over owned territory
            dist = { (sr, sc): 0 }
            parent = {}
            q = deque([(sr, sc)])
            while q:
                r, c = q.popleft()
                for d_idx, (dr, dc) in enumerate(DIRS):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and owner[nr][nc] == 1 and (nr, nc) not in dist:
                        dist[(nr, nc)] = dist[(r, c)] + 1
                        # Reverse direction points from neighbor toward (r, c)
                        rev_dir = (d_idx ^ 1)  # 0<->1, 2<->3
                        parent[(nr, nc)] = (r, c, rev_dir)
                        q.append((nr, nc))

            # Pick highest prize node away from root to flow along tree
            best_node = None
            max_prize_density = 0.0
            for (r, c), (pr, pc, direction) in parent.items():
                if army[r][c] >= 2:
                    prize = army[r][c] - 1
                    d = dist[(r, c)]
                    density = prize / max(1, d)
                    if density > max_prize_density:
                        max_prize_density = density
                        best_node = (r, c, direction)

            if best_node:
                r, c, direction = best_node
                return [0, int(r), int(c), int(direction), 0]

        # 5. Default expander move if no gather available
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 1 and army[r][c] >= 2:
                    for direction, (dr, dc) in enumerate(DIRS):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and terrain[nr][nc] != 2:
                            if owner[nr][nc] != 1:
                                return [0, int(r), int(c), int(direction), 0]

        return [1, 0, 0, 0, 0]

    except Exception:
        return [1, 0, 0, 0, 0]
