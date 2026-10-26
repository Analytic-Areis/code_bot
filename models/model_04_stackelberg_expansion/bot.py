"""Code Bot Participant: Model 4 - Stackelberg Expansion Leader
Bot: StackelbergLeader
Strategy: Stackelberg Leadership Commitment for Opening Macro Dominance
Sources: models.md (Game Theory Model 4)
AI assistance: Claude / Antigravity pair programming

Implements Stackelberg Leadership Expansion:
1. Commits early (Turns 0-100) to rapid, outward-thrusting expansion tendrils.
2. Forces the opponent to react defensively, abandoning their own geometric expansion.
3. Maximizes land count L(t) evaluated precisely at turn 50, 100, 150.
4. Transitions into aggressive frontline pressure using the accumulated army advantage.
"""
import time

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_my_general = None
_expansion_vector = None


def _reset():
    global _my_general, _expansion_vector
    _my_general = None
    _expansion_vector = None


def _find_my_general(obs):
    owner, terrain = obs["owner"], obs["type"]
    for r in range(obs["height"]):
        for c in range(obs["width"]):
            if owner[r][c] == 1 and terrain[r][c] == 4:
                return (r, c)
    return None


def act(obs):
    t_start = time.perf_counter()
    global _my_general, _expansion_vector

    try:
        h, w = obs["height"], obs["width"]
        owner, army, terrain = obs["owner"], obs["army"], obs["type"]

        if _my_general is None or obs["turn"] <= 1:
            _my_general = _find_my_general(obs)
            if _my_general:
                # Primary expansion direction toward map center or opposite quadrant
                target_r = h - 1 if _my_general[0] < h // 2 else 0
                target_c = w - 1 if _my_general[1] < w // 2 else 0
                _expansion_vector = (target_r, target_c)

        # 1. Check if enemy general is spotted
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 2 and terrain[r][c] == 4:
                    _expansion_vector = (r, c)
                    break

        # 2. Stackelberg expansion rule:
        # Move forward into unowned plains, maximizing delta L before mod 50 ticks
        tr, tc = _expansion_vector if _expansion_vector else (h // 2, w // 2)

        best_move = None
        best_score = -1e9

        for r in range(h):
            for c in range(w):
                if time.perf_counter() - t_start > 0.120:
                    break
                if owner[r][c] != 1 or army[r][c] < 2:
                    continue

                for d_idx, (dr, dc) in enumerate(DIRS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if terrain[nr][nc] == 2:  # mountain
                        continue

                    score = 0.0
                    # Stackelberg commitment: prefer moving into unowned tiles
                    if owner[nr][nc] == 0:
                        score += 50.0  # huge bonus for capturing neutral plain
                    elif owner[nr][nc] == 2:
                        # Combat
                        if army[r][c] - 1 > army[nr][nc] or obs["turn"] >= 800:
                            score += 80.0
                        else:
                            score -= 40.0
                    else:
                        # Moving into friendly territory (internal flow)
                        score += 5.0

                    # Distance improvement toward leadership vector
                    d_curr = abs(r - tr) + abs(c - tc)
                    d_next = abs(nr - tr) + abs(nc - tc)
                    score += (d_curr - d_next) * 12.0

                    # Extra incentive if close to mod 50 tick (Turns 35-49, 85-99, etc.)
                    turn_mod = obs["turn"] % 50
                    if turn_mod >= 35 and owner[nr][nc] == 0:
                        score += 30.0

                    if score > best_score:
                        best_score = score
                        best_move = [0, int(r), int(c), int(d_idx), 0]

        if best_move:
            return best_move

        return [1, 0, 0, 0, 0]

    except Exception:
        return [1, 0, 0, 0, 0]
