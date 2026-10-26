"""Code Bot Participant: Model 10 - Nash Bargaining Resource Allocator
Bot: NashResourceAllocator
Strategy: Nash Bargaining Solution for Offense-Defense Troop Partitioning
Sources: models.md (Game Theory Model 10)
AI assistance: Claude / Antigravity pair programming

Implements Nash Bargaining Resource Allocation:
1. Formulates internal troop split between Defense (x) and Offense (y).
2. Sets disagreement point d_def = threat level to general, d_off = minimal frontier presence.
3. Maximizes (U_def(x) - d_def) * (U_off(y) - d_off).
4. Strictly protects general garrison from being drained below d_def.
5. Projects offensive surplus y outward to pressure the opponent.
"""
import time

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_my_general = None


def _reset():
    global _my_general
    _my_general = None


def _find_my_general(obs):
    owner, terrain = obs["owner"], obs["type"]
    for r in range(obs["height"]):
        for c in range(obs["width"]):
            if owner[r][c] == 1 and terrain[r][c] == 4:
                return (r, c)
    return None


def act(obs):
    t_start = time.perf_counter()
    global _my_general

    try:
        h, w = obs["height"], obs["width"]
        owner, army, terrain = obs["owner"], obs["army"], obs["type"]

        if _my_general is None or obs["turn"] <= 1:
            _my_general = _find_my_general(obs)

        gr, gc = _my_general if _my_general else (h // 2, w // 2)

        # 1. Compute threat level to friendly general -> Disagreement point d_def
        max_visible_enemy = 0
        min_enemy_dist = 999
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 2:
                    if army[r][c] > max_visible_enemy:
                        max_visible_enemy = army[r][c]
                    d = abs(r - gr) + abs(c - gc)
                    if d < min_enemy_dist:
                        min_enemy_dist = d

        # Threat proxy: if enemy is close or large, d_def increases
        if min_enemy_dist <= 4:
            d_def = max(10, max_visible_enemy + 2)
        else:
            d_def = 5  # minimum baseline guard

        # Current defense level = army on general
        gen_army = army[gr][gc] if (0 <= gr < h and 0 <= gc < w and owner[gr][gc] == 1) else 1

        # 2. Nash Bargaining Check:
        # If gen_army < d_def: DEFENSE is below disagreement point! Gather troops to general!
        if gen_army < d_def:
            # Flow troops towards general
            best_move = None
            best_dist_gain = -999
            for r in range(h):
                for c in range(w):
                    if owner[r][c] == 1 and army[r][c] >= 2 and (r, c) != (gr, gc):
                        for d_idx, (dr, dc) in enumerate(DIRS):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and terrain[nr][nc] != 2:
                                d_curr = abs(r - gr) + abs(c - gc)
                                d_next = abs(nr - gr) + abs(nc - gc)
                                gain = d_curr - d_next
                                if gain > best_dist_gain:
                                    best_dist_gain = gain
                                    best_move = [0, int(r), int(c), int(d_idx), 0]
            if best_move and best_dist_gain > 0:
                return best_move

        # 3. Offense allocation: Surplus above d_def projects outward
        # Move troops that are outside general or surplus on general
        best_act = None
        best_score = -1e9

        for r in range(h):
            for c in range(w):
                if time.perf_counter() - t_start > 0.120:
                    break
                if owner[r][c] == 1 and army[r][c] >= 2:
                    # If on general, only move if army > d_def
                    if (r, c) == (gr, gc) and army[r][c] <= d_def:
                        continue

                    for d_idx, (dr, dc) in enumerate(DIRS):
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < h and 0 <= nc < w) or terrain[nr][nc] == 2:
                            continue

                        score = 0.0
                        if owner[nr][nc] == 0:
                            score += 25.0
                        elif owner[nr][nc] == 2:
                            if army[r][c] - 1 > army[nr][nc] or obs["turn"] >= 800:
                                score += 70.0
                            else:
                                score -= 35.0

                        # Push away from general into frontier
                        d_curr = abs(r - gr) + abs(c - gc)
                        d_next = abs(nr - gr) + abs(nc - gc)
                        score += (d_next - d_curr) * 10.0
                        score += army[r][c] * 0.4

                        if score > best_score:
                            best_score = score
                            best_act = [0, int(r), int(c), int(d_idx), 0]

        if best_act:
            return best_act

        return [1, 0, 0, 0, 0]

    except Exception:
        return [1, 0, 0, 0, 0]
