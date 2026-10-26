"""Code Bot Participant: Model 5 - Minimax Castle Facility Location
Bot: MinimaxCastleDefender
Strategy: Minimax Facility Location for Economically Optimal Castle Infrastructure
Sources: models.md (Game Theory Model 5)
AI assistance: Claude / Antigravity pair programming

Implements Minimax Facility Location for Castles:
1. Avoids the proximity surcharge Cost = 35 + sum(max(0, 14 - 2*d_i)) by strictly building at d >= 7.
2. Minimizes maximum distance from castle to general entrances for rapid reinforcement.
3. Issues kind=2 build actions only when economically sound.
4. Fortifies the perimeter with steady castle production (+1 army / 2 ticks).
"""
import time

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_my_general = None
_castles_built = 0


def _reset():
    global _my_general, _castles_built
    _my_general = None
    _castles_built = 0


def _find_my_general(obs):
    owner, terrain = obs["owner"], obs["type"]
    for r in range(obs["height"]):
        for c in range(obs["width"]):
            if owner[r][c] == 1 and terrain[r][c] == 4:
                return (r, c)
    return None


def _calculate_build_cost(r, c, structures):
    cost = 35
    for sr, sc in structures:
        d = abs(r - sr) + abs(c - sc)
        cost += max(0, 14 - 2 * d)
    return cost


def act(obs):
    t_start = time.perf_counter()
    global _my_general, _castles_built

    try:
        h, w = obs["height"], obs["width"]
        owner, army, terrain = obs["owner"], obs["army"], obs["type"]

        if _my_general is None or obs["turn"] <= 1:
            _my_general = _find_my_general(obs)

        # 1. Identify all structures (general and existing castles)
        structures = []
        if _my_general:
            structures.append(_my_general)
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 1 and terrain[r][c] == 3:
                    structures.append((r, c))

        # 2. Castle placement via Minimax Facility Location
        # Limit total castles (e.g. 2 max) to prevent economic overextension
        if len(structures) < 3 and obs["turn"] > 60 and obs["turn"] < 700:
            best_candidate = None
            min_max_dist = 999
            best_cost = 999

            for r in range(h):
                for c in range(w):
                    # Must be owned plain tile
                    if owner[r][c] == 1 and terrain[r][c] == 1:
                        # Must satisfy Manhattan distance >= 7 from all existing structures
                        min_dist_to_struct = min(abs(r - sr) + abs(c - sc) for sr, sc in structures)
                        if min_dist_to_struct >= 7:
                            # Evaluate minimax distance to general
                            dist_to_gen = abs(r - _my_general[0]) + abs(c - _my_general[1]) if _my_general else 7
                            cost = _calculate_build_cost(r, c, structures)
                            if dist_to_gen < min_max_dist:
                                min_max_dist = dist_to_gen
                                best_candidate = (r, c)
                                best_cost = cost

            # If we found an optimal facility location and have enough army on that tile:
            if best_candidate and army[best_candidate[0]][best_candidate[1]] >= best_cost:
                _castles_built += 1
                # Build castle: kind=2, r, c
                return [2, int(best_candidate[0]), int(best_candidate[1]), 0, 0]

        # 3. Defensive perimeter and gathering around general
        gr, gc = _my_general if _my_general else (h // 2, w // 2)

        # Emergency defense: if enemy within 2 steps of general
        for dr, dc in DIRS:
            nr, nc = gr + dr, gc + dc
            if 0 <= nr < h and 0 <= nc < w:
                if owner[nr][nc] == 2:  # enemy adjacent to general!
                    # Attack enemy from general or adjacent guard
                    if army[gr][gc] >= 2:
                        dir_idx = DIRS.index((dr, dc))
                        return [0, int(gr), int(gc), int(dir_idx), 0]

        # Inward gathering toward general or perimeter fortification
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
                        d_curr = abs(r - gr) + abs(c - gc)
                        d_next = abs(nr - gr) + abs(nc - gc)

                        if d_curr > 4:
                            # Far troops move inward toward general defense zone
                            score += (d_curr - d_next) * 15.0
                        else:
                            # Close troops expand or reinforce
                            if owner[nr][nc] == 0:
                                score += 20.0
                            elif owner[nr][nc] == 2:
                                score += 40.0

                        score += army[r][c] * 0.5
                        if score > best_score:
                            best_score = score
                            best_move = [0, int(r), int(c), int(d_idx), 0]

        if best_move:
            return best_move

        return [1, 0, 0, 0, 0]

    except Exception:
        return [1, 0, 0, 0, 0]
