"""Code Bot Participant: Model 6 - Two-Ring Network Interdiction
Bot: TwoRingInterdictor
Strategy: Two-Ring Network Interdiction & Source-Stripping Defense
Sources: models.md (Game Theory Model 6)
AI assistance: Claude / Antigravity pair programming

Implements Two-Ring Network Interdiction:
1. Ring 1 (Orthogonal general neighbors): Heavy garrison perimeters.
2. Ring 2 (Radius-2 neighbors): Mobile reactive interceptors.
3. Chasing Priority: Intercepts and strips enemy source tiles when they step into Ring 1.
4. Neutralizes Turn 800 Deathtouch by ensuring enemy never holds >= 2 units adjacent to general.
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

        # 1. Ring 1 classification (Manhattan dist == 1)
        ring1 = []
        for dr, dc in DIRS:
            nr, nc = gr + dr, gc + dc
            if 0 <= nr < h and 0 <= nc < w and terrain[nr][nc] != 2:
                ring1.append((nr, nc))

        # 2. Ring 2 classification (Manhattan dist == 2)
        ring2 = []
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if abs(dr) + abs(dc) == 2:
                    nr, nc = gr + dr, gc + dc
                    if 0 <= nr < h and 0 <= nc < w and terrain[nr][nc] != 2:
                        ring2.append((nr, nc))

        # 3. INTERDICTION CHECK: Is enemy currently in Ring 1?
        # If so, immediate priority: clear Ring 1 or strip its source from Ring 2 / general
        for r1_r, r1_c in ring1:
            if owner[r1_r][r1_c] == 2:
                # Enemy is on Ring 1! Lethal deathtouch threat if turn >= 800.
                # Attack it immediately from general or Ring 2
                if army[gr][gc] >= 2 and (obs["turn"] < 800 or army[gr][gc] - 1 > army[r1_r][r1_c]):
                    dir_idx = DIRS.index((r1_r - gr, r1_c - gc))
                    return [0, int(gr), int(gc), int(dir_idx), 0]

                # Strike from Ring 2 into Ring 1
                for r2_r, r2_c in ring2:
                    if owner[r2_r][r2_c] == 1 and army[r2_r][r2_c] >= 2:
                        for d_idx, (dr, dc) in enumerate(DIRS):
                            if r2_r + dr == r1_r and r2_c + dc == r1_c:
                                return [0, int(r2_r), int(r2_c), int(d_idx), 0]

        # 4. Ring 1 Garrison Maintenance:
        # Keep friendly garrisons on Ring 1 tiles
        for r1_r, r1_c in ring1:
            if owner[r1_r][r1_c] != 1 and army[gr][gc] >= 3:
                # Reinforce Ring 1 from general
                dir_idx = DIRS.index((r1_r - gr, r1_c - gc))
                return [0, int(gr), int(gc), int(dir_idx), 1]  # split to keep general guarded

        # 5. Ring 2 Mobile Patrol & Gathering:
        # Move outer troops into Ring 2 positions
        best_move = None
        best_score = -1e9
        for r in range(h):
            for c in range(w):
                if time.perf_counter() - t_start > 0.120:
                    break
                if owner[r][c] == 1 and army[r][c] >= 2:
                    # Do not drain general or Ring 1 unless necessary
                    if (r, c) == (gr, gc) or (r, c) in ring1:
                        continue

                    for d_idx, (dr, dc) in enumerate(DIRS):
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < h and 0 <= nc < w) or terrain[nr][nc] == 2:
                            continue

                        score = 0.0
                        d_target = min(abs(nr - r2_r) + abs(nc - r2_c) for r2_r, r2_c in ring2) if ring2 else 0
                        d_curr = min(abs(r - r2_r) + abs(c - r2_c) for r2_r, r2_c in ring2) if ring2 else 0
                        score += (d_curr - d_target) * 15.0

                        if owner[nr][nc] == 0:
                            score += 10.0
                        elif owner[nr][nc] == 2:
                            score += 30.0

                        if score > best_score:
                            best_score = score
                            best_move = [0, int(r), int(c), int(d_idx), 0]

        if best_move:
            return best_move

        # Fallback: pass or harmless local step
        return [1, 0, 0, 0, 0]

    except Exception:
        return [1, 0, 0, 0, 0]
