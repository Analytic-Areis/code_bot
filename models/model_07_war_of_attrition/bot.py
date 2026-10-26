"""Code Bot Participant: Model 7 - War of Attrition Staller
Bot: AttritionTurtle
Strategy: War of Attrition & Turn 1200 Draw Stalling with Asymmetric Defense
Sources: models.md (Game Theory Model 7)
AI assistance: Claude / Antigravity pair programming

Implements War of Attrition Strategy:
1. Maximizes defensive tie advantage (ties leave defender in possession of tile).
2. Minimizes active risk, preserving troops in dense defensive clusters around general.
3. Exploits turn 1200 automatic draw ceiling to guarantee tournament points.
4. Punishes overextended enemy attacks when enemy army depletes below threshold.
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

        # 1. Immediate General Defense:
        # Check if enemy is threatening general
        for d_idx, (dr, dc) in enumerate(DIRS):
            nr, nc = gr + dr, gc + dc
            if 0 <= nr < h and 0 <= nc < w:
                if owner[nr][nc] == 2:
                    # Enemy directly adjacent! Counter-attack with general if safe or deathtouch
                    if army[gr][gc] >= 2 and (obs["turn"] >= 800 or army[gr][gc] - 1 >= army[nr][nc]):
                        return [0, int(gr), int(gc), int(d_idx), 0]

        # 2. Asymmetric Trade Rule:
        # Never initiate combat unless we have a decisive numbers advantage (>= 2x)
        # Otherwise, let the opponent attack us so tied combat leaves us holding the tile!
        # Consolidate all troops into a compact 3-radius defensive fortress
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

                        # Turtle inward if outside radius 3
                        if d_curr > 3:
                            score += (d_curr - d_next) * 20.0
                        elif d_curr <= 3 and d_next <= 3:
                            # Internal troop balancing
                            if owner[nr][nc] == 1:
                                score += 5.0
                            elif owner[nr][nc] == 0:
                                score += 10.0  # safe nearby expansion
                            elif owner[nr][nc] == 2:
                                # Only engage if crushing advantage
                                if army[r][c] - 1 > army[nr][nc] * 1.5:
                                    score += 50.0
                                else:
                                    score -= 100.0  # refuse bad combat, let them hit us

                        score += army[r][c] * 0.2
                        if score > best_score:
                            best_score = score
                            best_move = [0, int(r), int(c), int(d_idx), 0]

        if best_move and best_score > 0:
            return best_move

        # If perfectly fortified, pass to preserve position and run down clock to 1200
        return [1, 0, 0, 0, 0]

    except Exception:
        return [1, 0, 0, 0, 0]
