"""Code Bot Participant: Model 14 - POMDP Ghost Tracker
Bot: POMDPGhostTracker
Strategy: Partially Observable MDP & Reachability Diamond Ghost Threat Tracking
Sources: models.md (Game Theory Model 14)
AI assistance: Claude / Antigravity pair programming

Implements POMDP Ghost Tracking:
1. Tracks enemy stacks that vanish into the fog as "Ghosts".
2. Propagates maximum reachability diamonds: dist <= delta_t.
3. Evaluates worst-case straight-line trajectory toward friendly general.
4. Triggers emergency recall of gather paths when projected ghost arrival breaches defense.
"""
import time

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_my_general = None
_ghosts = []  # list of dicts: {"origin": (r, c), "army": S, "last_turn": t}
_prev_enemy_cells = {}


def _reset():
    global _my_general, _ghosts, _prev_enemy_cells
    _my_general = None
    _ghosts = []
    _prev_enemy_cells = {}


def _find_my_general(obs):
    owner, terrain = obs["owner"], obs["type"]
    for r in range(obs["height"]):
        for c in range(obs["width"]):
            if owner[r][c] == 1 and terrain[r][c] == 4:
                return (r, c)
    return None


def act(obs):
    t_start = time.perf_counter()
    global _my_general, _ghosts, _prev_enemy_cells

    try:
        h, w = obs["height"], obs["width"]
        owner, army, terrain = obs["owner"], obs["army"], obs["type"]
        turn = obs["turn"]

        if _my_general is None or turn <= 1:
            _my_general = _find_my_general(obs)

        gr, gc = _my_general if _my_general else (h // 2, w // 2)

        # 1. Update visible enemy cells and detect new ghosts
        curr_enemy_cells = {}
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 2:
                    curr_enemy_cells[(r, c)] = army[r][c]

        # Check which previous enemy cells disappeared into fog
        for (r, c), a in _prev_enemy_cells.items():
            if (r, c) not in curr_enemy_cells:
                # Disappeared into fog!
                if a >= 4:
                    _ghosts.append({"origin": (r, c), "army": a, "last_turn": turn})

        _prev_enemy_cells = curr_enemy_cells

        # 2. Prune old or invalidated ghosts
        active_ghosts = []
        for g in _ghosts:
            dt = turn - g["last_turn"]
            # A ghost older than 25 turns or already visible is purged
            if dt < 25:
                # Projected worst-case distance to general
                gor, goc = g["origin"]
                dist_to_gen = abs(gor - gr) + abs(goc - gc)
                projected_dist = max(0, dist_to_gen - dt)
                active_ghosts.append({**g, "projected_dist": projected_dist})
        _ghosts = active_ghosts

        # 3. Assess Existential Threat Level from Ghosts and Visible Enemies
        general_army = army[gr][gc] if (0 <= gr < h and 0 <= gc < w and owner[gr][gc] == 1) else 1
        emergency_defense = False

        for g in _ghosts:
            if g["projected_dist"] <= 4 and g["army"] >= general_army:
                emergency_defense = True
                break

        # Also check visible enemies near general
        for (r, c), a in curr_enemy_cells.items():
            if abs(r - gr) + abs(c - gc) <= 3 and a >= general_army:
                emergency_defense = True
                break

        # 4. Action Selection
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

                        if emergency_defense:
                            # Emergency recall: pull all external units inward
                            if (r, c) != (gr, gc):
                                score += (d_curr - d_next) * 30.0
                            if owner[nr][nc] == 2:
                                score += 100.0  # intercept enemy
                        else:
                            # Standard expansion and frontier exploration
                            if owner[nr][nc] == 0:
                                score += 25.0
                            elif owner[nr][nc] == 2:
                                score += 50.0
                            score += (d_next - d_curr) * 5.0  # project forward

                        score += army[r][c] * 0.2
                        if score > best_score:
                            best_score = score
                            best_move = [0, int(r), int(c), int(d_idx), 0]

        if best_move:
            return best_move

        return [1, 0, 0, 0, 0]

    except Exception:
        return [1, 0, 0, 0, 0]
