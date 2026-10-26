"""Code Bot Participant: Model 8 - Fog Signaling Game
Bot: FogSignaler
Strategy: Costly Signaling of Strength via Fog-Border Peeking
Sources: models.md (Game Theory Model 8)
AI assistance: Claude / Antigravity pair programming

Implements Fog-of-War Signaling Game:
1. Gathers a visible vanguard stack.
2. Identifies the border between friendly vision and opponent territory / fog.
3. Steps the stack onto the visible border to send a strong signal of impending invasion.
4. Retreats back into fog on the next turn, freezing opponent expansion in panic.
5. Capitalizes on the opponent's defensive paralysis to expand safely behind the fog screen.
"""
import time

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_my_general = None
_signaling_state = 0  # 0: gather/idle, 1: stepped_forward (needs retreat)
_last_step = None


def _reset():
    global _my_general, _signaling_state, _last_step
    _my_general = None
    _signaling_state = 0
    _last_step = None


def _find_my_general(obs):
    owner, terrain = obs["owner"], obs["type"]
    for r in range(obs["height"]):
        for c in range(obs["width"]):
            if owner[r][c] == 1 and terrain[r][c] == 4:
                return (r, c)
    return None


def act(obs):
    t_start = time.perf_counter()
    global _my_general, _signaling_state, _last_step

    try:
        h, w = obs["height"], obs["width"]
        owner, army, terrain = obs["owner"], obs["army"], obs["type"]

        if _my_general is None or obs["turn"] <= 1:
            _my_general = _find_my_general(obs)

        # 1. State 1: Retreat signal (step back into fog/cover)
        if _signaling_state == 1 and _last_step:
            src_r, src_c, orig_r, orig_c = _last_step
            # If our army is still at src_r, src_c, retreat back to orig_r, orig_c
            if 0 <= src_r < h and 0 <= src_c < w and owner[src_r][src_c] == 1 and army[src_r][src_c] >= 2:
                for d_idx, (dr, dc) in enumerate(DIRS):
                    if src_r + dr == orig_r and src_c + dc == orig_c:
                        _signaling_state = 0
                        _last_step = None
                        return [0, int(src_r), int(src_c), int(d_idx), 0]
            _signaling_state = 0
            _last_step = None

        # 2. State 0: Look for signaling opportunity or expansion
        # Find border tiles adjacent to opponent vision or neutral fog
        border_candidates = []
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 1 and army[r][c] >= 10:
                    for d_idx, (dr, dc) in enumerate(DIRS):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and terrain[nr][nc] != 2:
                            # If adjacent to opponent territory or structure in fog
                            if owner[nr][nc] == 2 or terrain[nr][nc] in (0, 5):
                                border_candidates.append((army[r][c], r, c, d_idx, nr, nc))

        # If a large stack is on border and turn is ripe for signaling (e.g. every 20 turns)
        if border_candidates and obs["turn"] % 20 < 5:
            border_candidates.sort(key=lambda x: x[0], reverse=True)
            stack, r, c, d_idx, nr, nc = border_candidates[0]
            _signaling_state = 1
            _last_step = (nr, nc, r, c)
            return [0, int(r), int(c), int(d_idx), 0]

        # 3. Default defensive development & steady expansion
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
                        if owner[nr][nc] == 0:
                            score += 25.0
                        elif owner[nr][nc] == 2:
                            if army[r][c] - 1 > army[nr][nc] or obs["turn"] >= 800:
                                score += 60.0
                            else:
                                score -= 30.0

                        score += army[r][c] * 0.3
                        if score > best_score:
                            best_score = score
                            best_move = [0, int(r), int(c), int(d_idx), 0]

        if best_move:
            return best_move

        return [1, 0, 0, 0, 0]

    except Exception:
        return [1, 0, 0, 0, 0]
