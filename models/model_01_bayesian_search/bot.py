"""Code Bot Participant: Model 1 - Bayesian Search Game
Bot: BayesianSearchScout
Strategy: Bayesian Target Acquisition with Directed Exploration Tendrils
Sources: models.md (Game Theory Model 1)
AI assistance: Claude / Antigravity pair programming

Implements Bayesian Search Game:
1. Filters initial candidate enemy general locations (>= 17 steps away, symmetric room prior).
2. Updates belief distribution via Bayes' rule as vision clears fog.
3. Routes high-tempo single-file tendrils toward high-probability clusters.
4. Decapitates enemy general immediately once located.
"""
import time
from collections import deque

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT

# Persistent game state across turns
_my_general = None
_opp_general = None
_candidates = {}  # (r, c) -> probability
_initialized = False


def _reset():
    global _my_general, _opp_general, _candidates, _initialized
    _my_general = None
    _opp_general = None
    _candidates = {}
    _initialized = False


def _find_my_general(obs):
    owner, terrain = obs["owner"], obs["type"]
    for r in range(obs["height"]):
        for c in range(obs["width"]):
            if owner[r][c] == 1 and terrain[r][c] == 4:
                return (r, c)
    return None


def _bfs_distance(obs, start_r, start_c):
    h, w = obs["height"], obs["width"]
    terrain = obs["type"]
    dist = { (start_r, start_c): 0 }
    q = deque([(start_r, start_c)])
    while q:
        r, c = q.popleft()
        d = dist[(r, c)]
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dist:
                if terrain[nr][nc] != 2:  # not impassable mountain
                    dist[(nr, nc)] = d + 1
                    q.append((nr, nc))
    return dist


def _init_bayesian_prior(obs, my_gen):
    global _candidates, _initialized
    h, w = obs["height"], obs["width"]
    gr, gc = my_gen
    dists = _bfs_distance(obs, gr, gc)

    # Room metric: passable cells within 7 steps
    my_room = sum(1 for (r, c), d in dists.items() if d <= 7)

    cands = []
    for r in range(h):
        for c in range(w):
            d = dists.get((r, c), abs(r - gr) + abs(c - gc))
            if d >= 16:  # engine fairness filter: >= 17 steps, buffer 16
                cands.append((r, c))

    if not cands:
        for r in range(h):
            for c in range(w):
                if abs(r - gr) + abs(c - gc) >= 12:
                    cands.append((r, c))

    prob = 1.0 / len(cands) if cands else 1.0
    _candidates = {coord: prob for coord in cands}
    _initialized = True


def act(obs):
    t_start = time.perf_counter()
    global _my_general, _opp_general, _candidates, _initialized

    try:
        h, w = obs["height"], obs["width"]
        owner, army, terrain = obs["owner"], obs["army"], obs["type"]

        if obs["turn"] <= 1 or not _initialized or _my_general is None:
            _my_general = _find_my_general(obs)
            if _my_general:
                _init_bayesian_prior(obs, _my_general)

        # 1. Check if enemy general is spotted
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 2 and terrain[r][c] == 4:
                    _opp_general = (r, c)
                    break
            if _opp_general:
                break

        # 2. Update Bayesian belief: remove visible non-generals
        if not _opp_general and _candidates:
            eliminated = False
            for (cr, cc) in list(_candidates.keys()):
                if terrain[cr][cc] != 0:  # revealed cell
                    if not (owner[cr][cc] == 2 and terrain[cr][cc] == 4):
                        del _candidates[(cr, cc)]
                        eliminated = True
            if eliminated and _candidates:
                total_p = sum(_candidates.values())
                if total_p > 0:
                    for k in _candidates:
                        _candidates[k] /= total_p

        # Target selection: if general spotted, target it; else target top Bayesian cluster
        target = _opp_general
        if not target and _candidates:
            target = max(_candidates.items(), key=lambda x: x[1])[0]
        if not target:
            target = (h - 1 - _my_general[0], w - 1 - _my_general[1]) if _my_general else (h // 2, w // 2)

        # 3. Find our best available stack to move
        best_move = None
        best_score = -1e9

        # Precompute BFS distance map to target
        dist_to_target = {}
        q = deque([target])
        dist_to_target[target] = 0
        while q:
            r, c = q.popleft()
            d = dist_to_target[(r, c)]
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dist_to_target:
                    if terrain[nr][nc] != 2:
                        dist_to_target[(nr, nc)] = d + 1
                        q.append((nr, nc))

        # Evaluate all legal moves from our tiles with army >= 2
        for r in range(h):
            for c in range(w):
                if time.perf_counter() - t_start > 0.120:
                    break
                if owner[r][c] != 1 or army[r][c] < 2:
                    continue

                for direction, (dr, dc) in enumerate(DIRS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if terrain[nr][nc] == 2:  # impassable mountain
                        continue

                    curr_dist = dist_to_target.get((r, c), 999)
                    next_dist = dist_to_target.get((nr, nc), 999)

                    # Base score based on closing distance to Bayesian target
                    score = (curr_dist - next_dist) * 20.0
                    score += (army[r][c] - 1) * 0.5

                    # Enemy general decapitation / attack
                    if (nr, nc) == _opp_general:
                        score += 10000.0
                    elif owner[nr][nc] == 2:
                        if army[r][c] - 1 > army[nr][nc] or obs["turn"] >= 800:
                            score += 500.0
                        else:
                            score -= 300.0
                    elif owner[nr][nc] == 0:  # neutral capture / scouting
                        score += 25.0
                        if terrain[nr][nc] == 0:  # into fog
                            score += 15.0

                    if score > best_score:
                        best_score = score
                        best_move = [0, int(r), int(c), int(direction), 0]

        if best_move:
            return best_move
        return [1, 0, 0, 0, 0]

    except Exception:
        return [1, 0, 0, 0, 0]
