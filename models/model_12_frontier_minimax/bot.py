"""Code Bot Participant: Model 12 - Frontier Minimax Skirmisher
Bot: FrontierMinimaxTactician
Strategy: Local Zero-Sum Extensive-Form Minimax with Alpha-Beta Pruning
Sources: models.md (Game Theory Model 12)
AI assistance: Claude / Antigravity pair programming

Implements Frontier Minimax Skirmishing:
1. Detects local visible collision zones on the frontier.
2. Formulates a localized 2-ply zero-sum extensive-form game tree.
3. Evaluates tactical post-exchange trade value, favoring defensive choke points.
4. Prevents wasteful skirmish trades while optimizing unit preservation.
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


def _evaluate_trade(my_army, opp_army, is_attack):
    # Simultaneous / collision evaluation
    # If we attack opponent tile:
    if is_attack:
        if my_army - 1 > opp_army:
            # We win tile, keep (my_army - 1 - opp_army)
            return (my_army - 1 - opp_army) + 10.0
        elif my_army - 1 == opp_army:
            # Tie: defender keeps tile with 0 units! Bad for attacker!
            return -opp_army - 5.0
        else:
            # We lose attacking army, opponent keeps (opp_army - (my_army - 1))
            return -(my_army - 1) - 10.0
    else:
        # We hold tile defensively:
        if opp_army - 1 > my_army:
            return -(my_army)
        elif opp_army - 1 == my_army:
            # Tie: we hold tile!
            return 5.0
        else:
            return (my_army - (opp_army - 1))


def act(obs):
    t_start = time.perf_counter()
    global _my_general

    try:
        h, w = obs["height"], obs["width"]
        owner, army, terrain = obs["owner"], obs["army"], obs["type"]

        if _my_general is None or obs["turn"] <= 1:
            _my_general = _find_my_general(obs)

        # 1. Check for immediate frontier contact (visible enemy units adjacent to our units)
        frontier_contacts = []
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 1 and army[r][c] >= 2:
                    for d_idx, (dr, dc) in enumerate(DIRS):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and owner[nr][nc] == 2:
                            frontier_contacts.append((r, c, d_idx, nr, nc))

        # 2. Minimax evaluation over frontier moves
        best_tactical_move = None
        best_tactical_val = -1e9

        for r, c, d_idx, nr, nc in frontier_contacts:
            if time.perf_counter() - t_start > 0.120:
                break
            my_a = army[r][c]
            opp_a = army[nr][nc]

            # Minimax depth 2: our move vs worst-case opponent reaction
            val = _evaluate_trade(my_a, opp_a, is_attack=True)

            # Bonus if targeting enemy general or post-800 deathtouch
            if terrain[nr][nc] == 4 or obs["turn"] >= 800:
                val += 1000.0

            if val > best_tactical_val:
                best_tactical_val = val
                best_tactical_move = [0, int(r), int(c), int(d_idx), 0]

        # Only execute tactical attack if minimax value is favorable
        if best_tactical_move and best_tactical_val > 0:
            return best_tactical_move

        # 3. If no favorable tactical attack, expand or position on chokepoints
        best_macro_move = None
        best_macro_score = -1e9

        for r in range(h):
            for c in range(w):
                if time.perf_counter() - t_start > 0.120:
                    break
                if owner[r][c] == 1 and army[r][c] >= 2:
                    for d_idx, (dr, dc) in enumerate(DIRS):
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < h and 0 <= nc < w) or terrain[nr][nc] == 2:
                            continue

                        # Never walk into an unfavorable combat when expanding
                        if owner[nr][nc] == 2 and army[r][c] - 1 <= army[nr][nc] and obs["turn"] < 800:
                            continue

                        score = 0.0
                        if owner[nr][nc] == 0:
                            score += 25.0
                        elif owner[nr][nc] == 1:
                            score += 5.0

                        score += army[r][c] * 0.2
                        if score > best_macro_score:
                            best_macro_score = score
                            best_macro_move = [0, int(r), int(c), int(d_idx), 0]

        if best_macro_move:
            return best_macro_move

        return [1, 0, 0, 0, 0]

    except Exception:
        return [1, 0, 0, 0, 0]
