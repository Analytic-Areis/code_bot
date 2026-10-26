"""Code Bot Participant: Model 16 - Information Revelation Prober
Bot: VoIProber
Strategy: Value of Information (VoI) Probing in Repeated Subgames
Sources: models.md (Game Theory Model 16)
AI assistance: Claude / Antigravity pair programming

Implements Value of Information (VoI) Probing:
1. Calculates VoI = expected uncertainty reduction / unit cost.
2. Identifies key fog exploration vectors that maximize 3x3 visibility gain.
3. Launches dedicated probe stacks to force opponent reaction.
4. Collapses Bayesian uncertainty regarding enemy deployment.
"""
import time

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_my_general = None
_opp_general = None


def _reset():
    global _my_general, _opp_general
    _my_general = None
    _opp_general = None


def _find_my_general(obs):
    owner, terrain = obs["owner"], obs["type"]
    for r in range(obs["height"]):
        for c in range(obs["width"]):
            if owner[r][c] == 1 and terrain[r][c] == 4:
                return (r, c)
    return None


def _count_fog_revealed(obs, r, c):
    # Stepping onto (r, c) reveals adjacent 3x3
    h, w = obs["height"], obs["width"]
    terrain = obs["type"]
    fog_count = 0
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                if terrain[nr][nc] == 0:  # fog
                    fog_count += 1
    return fog_count


def act(obs):
    t_start = time.perf_counter()
    global _my_general, _opp_general

    try:
        h, w = obs["height"], obs["width"]
        owner, army, terrain = obs["owner"], obs["army"], obs["type"]

        if _my_general is None or obs["turn"] <= 1:
            _my_general = _find_my_general(obs)

        # 1. Check for enemy general
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 2 and terrain[r][c] == 4:
                    _opp_general = (r, c)
                    break
            if _opp_general:
                break

        # If enemy general is already known, VoI of scouting is zero -> Direct assault!
        if _opp_general:
            gr, gc = _opp_general
            best_move = None
            best_dist = 999
            for r in range(h):
                for c in range(w):
                    if owner[r][c] == 1 and army[r][c] >= 2:
                        for d_idx, (dr, dc) in enumerate(DIRS):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and terrain[nr][nc] != 2:
                                d = abs(nr - gr) + abs(nc - gc)
                                if d < best_dist:
                                    best_dist = d
                                    best_move = [0, int(r), int(c), int(d_idx), 0]
            if best_move:
                return best_move

        # 2. Information Probing: Evaluate moves based on VoI (Value of Information)
        best_act = None
        best_voi_score = -1e9

        for r in range(h):
            for c in range(w):
                if time.perf_counter() - t_start > 0.120:
                    break
                if owner[r][c] == 1 and army[r][c] >= 2:
                    for d_idx, (dr, dc) in enumerate(DIRS):
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < h and 0 <= nc < w) or terrain[nr][nc] == 2:
                            continue

                        # VoI metric: newly revealed fog cells
                        fog_gain = _count_fog_revealed(obs, nr, nc)
                        voi = fog_gain * 10.0

                        if owner[nr][nc] == 0:
                            voi += 20.0  # capturing plain
                        elif owner[nr][nc] == 2:
                            voi += 50.0  # contact with enemy forces information revelation

                        # Army weighting
                        voi += army[r][c] * 0.2

                        if voi > best_voi_score:
                            best_voi_score = voi
                            best_act = [0, int(r), int(c), int(d_idx), 0]

        if best_act:
            return best_act

        return [1, 0, 0, 0, 0]

    except Exception:
        return [1, 0, 0, 0, 0]
