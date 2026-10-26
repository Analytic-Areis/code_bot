"""Code Bot Participant: Model 17 - Proposed Unified Priority Cascade
Bot: GrandmasterPriorityCascade
Strategy: Heuristic Priority Cascade with Mod-50 Sync, Scoreboard Calculus & Turn 800 Singularity
Sources: models.md (Section: Strategic Domination in Imperfect Information Grid Games)
AI assistance: Claude / Antigravity pair programming

Implements the Proposed Grandmaster Priority Cascade:
Priority 1: Existential General Defense & Deathtouch Blocking
Priority 2: Lethal Strike / Deathtouch Singularity Execution (Turn >= 800)
Priority 3: Fog of War Breach & Target Acquisition
Priority 4: Macroeconomic Consolidation / Gathering (t % 50 < 38)
Priority 5: Frontier Expansion / Land Maximization (t % 50 >= 38)
Strictly adheres to 150ms deadline with 120ms safety guard.
"""
import time
from collections import deque

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_my_general = None
_opp_general = None
_prev_opp_land = 0
_prev_opp_army = 0


def _reset():
    global _my_general, _opp_general, _prev_opp_land, _prev_opp_army
    _my_general = None
    _opp_general = None
    _prev_opp_land = 0
    _prev_opp_army = 0


def _find_my_general(obs):
    owner, terrain = obs["owner"], obs["type"]
    for r in range(obs["height"]):
        for c in range(obs["width"]):
            if owner[r][c] == 1 and terrain[r][c] == 4:
                return (r, c)
    return None


def act(obs):
    t_start = time.perf_counter()
    global _my_general, _opp_general, _prev_opp_land, _prev_opp_army

    try:
        h, w = obs["height"], obs["width"]
        owner, army, terrain = obs["owner"], obs["army"], obs["type"]
        turn = obs["turn"]

        if _my_general is None or turn <= 1:
            _my_general = _find_my_general(obs)
            _prev_opp_land = obs["opp_land"]
            _prev_opp_army = obs["opp_army"]

        gr, gc = _my_general if _my_general else (h // 2, w // 2)

        # 0. Track global scoreboard differential
        delta_l_opp = obs["opp_land"] - _prev_opp_land
        delta_a_opp = obs["opp_army"] - _prev_opp_army
        _prev_opp_land = obs["opp_land"]
        _prev_opp_army = obs["opp_army"]

        # Check for enemy general
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 2 and terrain[r][c] == 4:
                    _opp_general = (r, c)
                    break
            if _opp_general:
                break

        # ============================================================
        # PRIORITY 1: EXISTENTIAL GENERAL DEFENSE & DEATHTOUCH BLOCKING
        # ============================================================
        # Check direct threats to general (within distance 2)
        urgent_threats = []
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                if abs(dr) + abs(dc) <= 2 and (dr != 0 or dc != 0):
                    nr, nc = gr + dr, gc + dc
                    if 0 <= nr < h and 0 <= nc < w and owner[nr][nc] == 2:
                        urgent_threats.append((nr, nc, army[nr][nc]))

        if urgent_threats:
            urgent_threats.sort(key=lambda x: (abs(x[0] - gr) + abs(x[1] - gc), -x[2]))
            tr, tc, tr_army = urgent_threats[0]

            # If threat is directly adjacent (distance 1):
            if abs(tr - gr) + abs(tc - gc) == 1:
                # Attack from general if we can eliminate it or post-800
                if army[gr][gc] >= 2 and (turn >= 800 or army[gr][gc] - 1 > tr_army):
                    d_idx = DIRS.index((tr - gr, tc - gc))
                    return [0, int(gr), int(gc), int(d_idx), 0]

            # Intercept threat using nearest friendly army
            for r in range(h):
                for c in range(w):
                    if owner[r][c] == 1 and army[r][c] >= 2 and (r, c) != (gr, gc):
                        for d_idx, (dr, dc) in enumerate(DIRS):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if (nr, nc) == (tr, tc):
                                    return [0, int(r), int(c), int(d_idx), 0]
                                if abs(nr - tr) + abs(nc - tc) < abs(r - tr) + abs(c - tc):
                                    return [0, int(r), int(c), int(d_idx), 0]

        # ============================================================
        # PRIORITY 2: LETHAL STRIKE / DEATHTOUCH SINGULARITY (Turn >= 800)
        # ============================================================
        if _opp_general:
            otr, otc = _opp_general
            # Check if any friendly unit is adjacent to enemy general
            for d_idx, (dr, dc) in enumerate(DIRS):
                fr, fc = otr + dr, otc + dc
                if 0 <= fr < h and 0 <= fc < w and owner[fr][fc] == 1 and army[fr][fc] >= 2:
                    rev_dir = d_idx ^ 1
                    # Lethal strike condition: turn >= 800 or outnumber garrison
                    if turn >= 800 or army[fr][fc] - 1 > army[otr][otc]:
                        return [0, int(fr), int(fc), int(rev_dir), 0]

            # Path largest available stack directly to enemy general
            max_stack = 0
            best_stack_pos = None
            for r in range(h):
                for c in range(w):
                    if owner[r][c] == 1 and army[r][c] > max_stack:
                        max_stack = army[r][c]
                        best_stack_pos = (r, c)

            if best_stack_pos and (turn >= 750 or max_stack > army[otr][otc] + 10):
                sr, sc = best_stack_pos
                best_dir = None
                best_dist = 999
                for d_idx, (dr, dc) in enumerate(DIRS):
                    nr, nc = sr + dr, sc + dc
                    if 0 <= nr < h and 0 <= nc < w and terrain[nr][nc] != 2:
                        d = abs(nr - otr) + abs(nc - otc)
                        if d < best_dist:
                            best_dist = d
                            best_dir = d_idx
                if best_dir is not None and army[sr][sc] >= 2:
                    return [0, int(sr), int(sc), int(best_dir), 0]

        # ============================================================
        # PRIORITY 3: FOG OF WAR BREACH & TARGET ACQUISITION
        # ============================================================
        target_dest = _opp_general if _opp_general else (h - 1 - gr, w - 1 - gc)

        # ============================================================
        # PRIORITY 4 & 5: MACRO CYCLE (Gathering vs Expansion)
        # ============================================================
        mod_turn = turn % 50
        is_expansion_phase = (mod_turn >= 38)  # Prior to mod-50: expand aggressively

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
                        d_target_curr = abs(r - target_dest[0]) + abs(c - target_dest[1])
                        d_target_next = abs(nr - target_dest[0]) + abs(nc - target_dest[1])

                        if is_expansion_phase:
                            # PRIORITY 5: Frontier Expansion
                            if owner[nr][nc] == 0:
                                score += 50.0  # huge bonus for capturing neutral plain
                            elif owner[nr][nc] == 2:
                                if army[r][c] - 1 > army[nr][nc] or turn >= 800:
                                    score += 70.0
                                else:
                                    score -= 40.0
                            score += (d_target_curr - d_target_next) * 10.0
                        else:
                            # PRIORITY 4: Macroeconomic Consolidation (Gathering inward/forward)
                            if owner[nr][nc] == 1:
                                score += 15.0  # consolidating stacks
                            elif owner[nr][nc] == 0:
                                score += 20.0
                            score += (d_target_curr - d_target_next) * 20.0

                        score += army[r][c] * 0.4
                        if score > best_score:
                            best_score = score
                            best_move = [0, int(r), int(c), int(d_idx), 0]

        if best_move:
            return best_move

        return [1, 0, 0, 0, 0]

    except Exception:
        return [1, 0, 0, 0, 0]
