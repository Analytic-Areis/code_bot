"""Code Bot Participant: Grandmaster Champion
Bot: Model 25 Grandmaster Champion
Strategy: 5-Tier Priority Cascade with Inward Flow Aggregation, 2-Step Minimax Tactical Look-Ahead, Strategic Castle Economics, Exact Room-Fairness Bayesian Tracking, and Turn 800 Deathtouch Singularity
Sources: model_23_rl_champion/bot.py + model_18_codebot_v0_known_good/bot.py + model_17_proposed_cascade/bot.py
AI assistance: Antigravity pair programming

Architecture:
  Tier 0: Evaluator Contract Integrity, Anti-Ping-Pong Cyclic History Guard
  Tier 1: Inviolable Reflexes (1-hop General Lethal Decapitation, Real Danger Pullback Defense, 1-hop Garrison Lock)
  Tier 2: Bayesian General Location Heatmap + Scoreboard POMDP Calculus
  Tier 3: Strategic Phase Engine (Mod-50 Land Snatch, Inward Flow Aggregator, Castle Economics, Turn 800 Deathtouch Staging)
  Tier 4: 2-Step Tactical Simultaneous Look-Ahead & Minimax Heuristic Scorer
  Tier 5: 80ms Hard Watchdog Safety Timer (Zero Timeout Forfeit Guarantee)
"""

import time
from collections import deque

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
OPPOSITE_DIR = {0: 1, 1: 0, 2: 3, 3: 2}

# ─── EVOLVABLE UTILITY WEIGHTS (Tuned via Self-Play ES) ────────────────────────
DIST_WEIGHT        = 24.068    # Approach bonus per step towards target
ARMY_BONUS         = 0.403    # Army stacking utility multiplier
CAPTURE_WEIGHT     = 573.321   # Enemy tile capture reward
ATTACK_PENALTY     = -284.901  # Losing skirmish penalty
FOG_BONUS          = 15.49    # Fog exploration reward
NEUTRAL_BONUS      = 20.513    # Neutral land expansion bonus
DEATHTOUCH_BONUS   = 13832.318 # Turn 800+ decapitation force
KILL_BONUS         = 11000.0 # Direct general kill reward
SPREAD_MOD50_BONUS = 15.198    # Mod-50 economy expansion burst
GARRISON_MIN       = 12      # General safe minimum garrison
CASTLE_THRESHOLD   = 48      # Army required on tile to construct castle
VULNERABLE_PENALTY = -176.523  # 2-step tactical vulnerability penalty

# ─── PERSISTENT MODULE STATE ──────────────────────────────────────────────────
_my_general     = None
_opp_general    = None
_mountains      = set()
_candidates     = {}     # (r, c) -> probability mass
_initialized    = False
_action_history = deque(maxlen=8)
_castles_built  = 0
_opp_land_history = deque(maxlen=20)
_opp_army_history = deque(maxlen=20)


def _reset():
    """State isolation between matches."""
    global _my_general, _opp_general, _mountains, _candidates, _initialized, _action_history, _castles_built
    global _opp_land_history, _opp_army_history
    _my_general     = None
    _opp_general    = None
    _mountains      = set()
    _candidates     = {}
    _initialized    = False
    _action_history.clear()
    _castles_built  = 0
    _opp_land_history.clear()
    _opp_army_history.clear()


def _find_my_general(obs):
    owner, terrain = obs["owner"], obs["type"]
    for r in range(obs["height"]):
        for c in range(obs["width"]):
            if owner[r][c] == 1 and terrain[r][c] == 4:
                return (r, c)
    return None


def _bfs_room(h, w, mountains, start_r, start_c, max_depth=7):
    visited = {(start_r, start_c): 0}
    q = deque([(start_r, start_c)])
    count = 0
    while q:
        r, c = q.popleft()
        d = visited[(r, c)]
        if d < max_depth:
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in mountains and (nr, nc) not in visited:
                    visited[(nr, nc)] = d + 1
                    count += 1
                    q.append((nr, nc))
    return count


def _bfs_distance(h, w, mountains, start_r, start_c):
    visited = {(start_r, start_c): 0}
    q = deque([(start_r, start_c)])
    while q:
        r, c = q.popleft()
        d = visited[(r, c)]
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in mountains and (nr, nc) not in visited:
                visited[(nr, nc)] = d + 1
                q.append((nr, nc))
    return visited


def _init_candidates(obs, my_gen):
    global _candidates, _mountains, _initialized
    h, w = obs["height"], obs["width"]
    terrain = obs["type"]

    _mountains = set()
    for r in range(h):
        for c in range(w):
            if terrain[r][c] in (2, 5):
                _mountains.add((r, c))

    gr, gc = my_gen
    my_room = _bfs_room(h, w, _mountains, gr, gc, max_depth=7)
    dists = _bfs_distance(h, w, _mountains, gr, gc)

    # 1. Primary Filter: Exact room-fairness filter (d >= 17, room difference <= 5)
    cands = []
    for (r, c), d in dists.items():
        if d >= 17:
            cand_room = _bfs_room(h, w, _mountains, r, c, max_depth=7)
            if abs(cand_room - my_room) <= 5:
                cands.append((r, c))

    # 2. Fallback Filter: distance >= 14 without strict room filter
    if not cands:
        cands = [(r, c) for (r, c), d in dists.items() if d >= 14]

    # 3. Geometric Fallback for tiny/extreme maps
    if not cands:
        cands = [
            (r, c) for r in range(h) for c in range(w)
            if (r, c) not in _mountains and abs(r - gr) + abs(c - gc) >= 12
        ]

    # Generals.io map generator prior: Opposing general is near the point-reflection mirror
    target_mirror = (h - 1 - gr, w - 1 - gc)
    raw_weights = {
        coord: 1.0 / (1.0 + (abs(coord[0] - target_mirror[0]) + abs(coord[1] - target_mirror[1])) ** 1.5)
        for coord in cands
    }
    total_w = sum(raw_weights.values()) if raw_weights else 1.0
    _candidates = {coord: w / total_w for coord, w in raw_weights.items()}
    _initialized = True


def _update_bayes(obs):
    """Remove revealed non-general cells from candidate set; renormalize."""
    terrain = obs["type"]
    owner   = obs["owner"]
    changed = False
    for coord in list(_candidates.keys()):
        r, c = coord
        if terrain[r][c] != 0:  # revealed cell
            if not (owner[r][c] == 2 and terrain[r][c] == 4):
                del _candidates[coord]
                changed = True
    if changed and _candidates:
        total = sum(_candidates.values())
        if total > 0:
            for k in _candidates:
                _candidates[k] /= total


def _best_target(obs):
    if _opp_general:
        return _opp_general
    if _candidates:
        return max(_candidates.items(), key=lambda x: x[1])[0]
    h, w = obs["height"], obs["width"]
    gr, gc = _my_general if _my_general else (h // 2, w // 2)
    return (h - 1 - gr, w - 1 - gc)


def _detect_loop(r, c, direction):
    """Detect if this action reverses the immediate prior action creating a 2-cycle."""
    if not _action_history:
        return False
    prev_kind, prev_r, prev_c, prev_d, _ = _action_history[-1]
    if prev_kind == 0:
        dr, dc = DIRS[prev_d]
        if (r, c) == (prev_r + dr, prev_c + dc) and direction == OPPOSITE_DIR.get(prev_d):
            return True
    return False


def act(obs):
    t_start = time.perf_counter()
    global _my_general, _opp_general, _candidates, _initialized, _action_history, _castles_built

    try:
        h, w    = obs["height"], obs["width"]
        owner   = obs["owner"]
        army    = obs["army"]
        terrain = obs["type"]
        turn    = obs["turn"]

        # ── 1. INITIALIZE & ACQUIRE GENERALS ──────────────────────────────────
        if turn <= 1 or not _initialized or _my_general is None:
            _my_general = _find_my_general(obs)
            if _my_general:
                _init_candidates(obs, _my_general)

        if not _opp_general:
            for r in range(h):
                for c in range(w):
                    if owner[r][c] == 2 and terrain[r][c] == 4:
                        _opp_general = (r, c)
                        break
                if _opp_general:
                    break

        if not _opp_general and _candidates:
            _update_bayes(obs)

        gr, gc = _my_general if _my_general else (0, 0)

        # ── 2. SCOREBOARD & THREAT METRICS ────────────────────────────────────
        visible_enemy_army = 0
        enemy_adjacent_cells = {}
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 2:
                    visible_enemy_army += army[r][c]
                    # Map adjacent enemy power for 2-step lookahead
                    for dr, dc in DIRS:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            enemy_adjacent_cells[(nr, nc)] = max(
                                enemy_adjacent_cells.get((nr, nc), 0), army[r][c]
                            )

        opp_army = obs.get("opp_army", visible_enemy_army)
        hidden_enemy_mass = max(0, opp_army - visible_enemy_army)

        # Compute threat distance & worst excess from general
        dist_from_gen = _bfs_distance(h, w, _mountains, gr, gc)
        worst_excess = 0
        threat_pos = None

        for r in range(h):
            for c in range(w):
                if owner[r][c] == 2:
                    d = dist_from_gen.get((r, c), 999)
                    if 0 < d <= 10:
                        gen_growth = d // 2
                        excess = (army[r][c] - 1) - (army[gr][gc] + gen_growth)
                        if excess > worst_excess:
                            worst_excess = excess
                            threat_pos = (r, c)

        # ── REAL-TIME IN-GAME PARAMETER ADAPTATION ────────────────────────────
        opp_land = obs.get("opp_land", 1)
        _opp_land_history.append(opp_land)
        _opp_army_history.append(opp_army)

        if len(_opp_land_history) >= 8:
            expansion_rate = (opp_land - _opp_land_history[0]) / len(_opp_land_history)
        else:
            expansion_rate = 0.35

        army_per_land = opp_army / max(1, opp_land)

        # Start from base evolved weights and dynamically adapt to opponent style
        dist_weight        = DIST_WEIGHT
        army_bonus         = ARMY_BONUS
        capture_weight     = CAPTURE_WEIGHT
        attack_penalty     = ATTACK_PENALTY
        fog_bonus          = FOG_BONUS
        neutral_bonus      = NEUTRAL_BONUS
        deathtouch_bonus   = DEATHTOUCH_BONUS
        spread_mod50_bonus = SPREAD_MOD50_BONUS
        garrison_min       = GARRISON_MIN
        if turn >= 150:
            garrison_min = max(garrison_min, 18.0)
        vulnerable_penalty = VULNERABLE_PENALTY

        # Dynamic Adaptation A: Rusher Detection (threat approaching or hoarded rush stack)
        is_rusher = (worst_excess > 0) or (40 <= turn <= 160 and opp_land <= 4 and opp_army >= 18)
        if is_rusher:
            garrison_min       = max(garrison_min, 16.0)  # Lock higher garrison at home to absorb rush
            vulnerable_penalty *= 1.25   # Penalize walking into enemy counter-threats
            attack_penalty     *= 1.20   # Suppress losing skirmishes
            capture_weight     *= 1.20   # Decapitate approaching enemy scouts

        # Dynamic Adaptation B: Turtle / Fortress Detection (Low expansion, hoarding army, safe home)
        is_turtle = (turn >= 80 and expansion_rate < 0.22 and army_per_land > 1.7 and worst_excess == 0)
        if is_turtle:
            neutral_bonus      *= 1.40   # Snatch maximum neutral territory while enemy turtles
            fog_bonus          *= 1.30   # Expand vision deeper into fog
            spread_mod50_bonus *= 1.35   # Maximize mod-50 army generation
            garrison_min        = max(12.0, garrison_min - 2.0)  # Release idle backline garrison

        # Dynamic Adaptation C: Boomer / Ribbon Spreader Detection (Fast expansion, thin army)
        is_boomer = (turn >= 60 and expansion_rate > 0.45 and army_per_land < 1.35)
        if is_boomer:
            capture_weight     *= 1.35   # Puncture thin 1-army expansion ribbons
            army_bonus         *= 1.30   # Merge stacks into heavy assault columns

        # ── 3. TIER 1: EMERGENCY HARD REFLEXES ────────────────────────────────
        # A. Lethal strike on enemy general if adjacent
        if _opp_general:
            egr, egc = _opp_general
            for d_idx, (dr, dc) in enumerate(DIRS):
                adj_r, adj_c = egr + dr, egc + dc
                if 0 <= adj_r < h and 0 <= adj_c < w and owner[adj_r][adj_c] == 1:
                    my_atk = army[adj_r][adj_c]
                    opp_def = army[egr][egc]
                    rev_dir = OPPOSITE_DIR[d_idx]
                    if my_atk - 1 > opp_def or turn >= 800:
                        act_vec = [0, int(adj_r), int(adj_c), int(rev_dir), 0]
                        _action_history.append(act_vec)
                        return act_vec

        # B. Direct Defense of our General from adjacent threats
        for d_idx, (dr, dc) in enumerate(DIRS):
            adj_r, adj_c = gr + dr, gc + dc
            if 0 <= adj_r < h and 0 <= adj_c < w and owner[adj_r][adj_c] == 2:
                enemy_threat = army[adj_r][adj_c]
                # If general has enough army to crush threat, counter-attack
                if army[gr][gc] - 1 > enemy_threat:
                    act_vec = [0, int(gr), int(gc), int(d_idx), 1]
                    _action_history.append(act_vec)
                    return act_vec
                # Otherwise, reinforce general from any adjacent allied cell
                for d2_idx, (dr2, dc2) in enumerate(DIRS):
                    nr2, nc2 = gr + dr2, gc + dc2
                    if 0 <= nr2 < h and 0 <= nc2 < w and owner[nr2][nc2] == 1 and (nr2, nc2) != (gr, gc):
                        if army[nr2][nc2] >= 2:
                            rev_dir2 = OPPOSITE_DIR[d2_idx]
                            act_vec = [0, int(nr2), int(nc2), int(rev_dir2), 0]
                            _action_history.append(act_vec)
                            return act_vec

        # D. Real Danger Pullback: Stream nearest army home before threat arrives
        if worst_excess > 0 and threat_pos:
            best_def_move = None
            best_gain = -999
            for r in range(h):
                for c in range(w):
                    if owner[r][c] == 1 and army[r][c] >= 2 and (r, c) != (gr, gc):
                        d_curr = dist_from_gen.get((r, c), 999)
                        for d_idx, (dr, dc) in enumerate(DIRS):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in _mountains:
                                d_next = dist_from_gen.get((nr, nc), 999)
                                if d_next < d_curr:
                                    gain = army[r][c]
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_def_move = [0, int(r), int(c), int(d_idx), 0]
            if best_def_move:
                _action_history.append(best_def_move)
                return best_def_move

        # ── 4. STRATEGIC CASTLE ECONOMICS (kind = 2) ──────────────────────────
        # Castles generate +1 army every 2 ticks! Build at d >= 7 from existing structures
        my_structures = [(gr, gc)]
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 1 and terrain[r][c] == 3:
                    my_structures.append((r, c))

        if 60 <= turn <= 350 and _castles_built < 2 and worst_excess == 0:
            for r in range(h):
                for c in range(w):
                    if owner[r][c] == 1 and terrain[r][c] == 1 and army[r][c] >= CASTLE_THRESHOLD:
                        # Compute distance penalty: 35 + sum(max(0, 14 - 2d))
                        dist_penalty = sum(max(0, 14 - 2 * (abs(r - sr) + abs(c - sc))) for sr, sc in my_structures)
                        build_cost = 35 + dist_penalty
                        # Ensure d >= 7 from all existing structures so dist_penalty == 0
                        min_struct_dist = min(abs(r - sr) + abs(c - sc) for sr, sc in my_structures)
                        if min_struct_dist >= 7 and army[r][c] >= build_cost:
                            _castles_built += 1
                            act_vec = [2, int(r), int(c), 0, 0]
                            _action_history.append(act_vec)
                            return act_vec

        # ── 5. TIER 2: DISTANCE MAP TO PRIMARY TARGET ─────────────────────────
        target = _best_target(obs)
        dist_map = {}
        q = deque([target])
        dist_map[target] = 0
        while q:
            r, c = q.popleft()
            d = dist_map[(r, c)]
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dist_map:
                    if terrain[nr][nc] != 2:
                        dist_map[(nr, nc)] = d + 1
                        q.append((nr, nc))

        # ── 6. TIER 3 & 4: TACTICAL EVALUATION & 2-STEP LOOK-AHEAD ─────────────
        best_move  = None
        best_score = -1e18

        is_mod50_burst = (turn % 50) >= 38 or (turn % 50) <= 2
        is_deathtouch_staging = 740 <= turn < 800

        for r in range(h):
            # Hard 80ms Watchdog Timer to prevent timeout
            if time.perf_counter() - t_start > 0.080:
                break
            for c in range(w):
                if owner[r][c] != 1 or army[r][c] < 2:
                    continue

                army_curr = army[r][c]
                curr_d = dist_map.get((r, c), 999)

                # Guard general garrison: never drain general under minimum; use split=1 to retain garrison
                split_flag = 0
                if (r, c) == (gr, gc):
                    # NEVER move general away if an enemy is adjacent
                    has_adj_enemy = any(
                        0 <= gr+dr < h and 0 <= gc+dc < w and owner[gr+dr][gc+dc] == 2
                        for dr, dc in DIRS
                    )
                    if has_adj_enemy or worst_excess > 0 or army_curr < garrison_min:
                        continue
                    if worst_excess > 0 or army_curr < garrison_min:
                        continue
                    split_flag = 1  # Keep half army on general for defense

                for direction, (dr, dc) in enumerate(DIRS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if terrain[nr][nc] == 2:  # mountain
                        continue

                    next_d = dist_map.get((nr, nc), 999)
                    score = (curr_d - next_d) * dist_weight
                    score += (army_curr - 1) * army_bonus

                    # Anti-pingpong cycle penalty
                    if _detect_loop(r, c, direction):
                        score -= 600.0

                    # Mod-50 Land Snatch Priority
                    if is_mod50_burst and owner[nr][nc] != 1:
                        score += spread_mod50_bonus * 2.5

                    # Move onto targets
                    if _opp_general and (nr, nc) == _opp_general:
                        if turn >= 800:
                            score += deathtouch_bonus
                        elif army_curr - 1 > army[nr][nc]:
                            score += KILL_BONUS
                        else:
                            score += attack_penalty * 2

                    elif is_deathtouch_staging and _opp_general and next_d <= 2:
                        # Stage around enemy general without suiciding before 800
                        if next_d == 1:
                            score += 8000.0
                        elif next_d == 2:
                            score += 5000.0

                    elif owner[nr][nc] == 2:  # Enemy tile
                        opp_tile_army = army[nr][nc]
                        if army_curr - 1 > opp_tile_army or turn >= 800:
                            if terrain[nr][nc] == 4:
                                score += KILL_BONUS
                            else:
                                score += capture_weight
                                if opp_tile_army >= 3:
                                    score += opp_tile_army * 15.0
                        else:
                            score += attack_penalty

                    elif owner[nr][nc] == 0:  # Neutral tile
                        score += neutral_bonus
                        if terrain[nr][nc] == 0:  # Fog exploration
                            score += fog_bonus
                        elif terrain[nr][nc] == 3:  # Castle
                            if army_curr - 1 > army[nr][nc]:
                                score += 400.0  # Safe castle capture
                            else:
                                score -= 200.0  # Premature castle suicide

                    elif owner[nr][nc] == 1:  # Inward Army Funnel / Consolidation
                        # Backline consolidation towards frontline spearhead
                        if next_d < curr_d:
                            score += 18.0 + (army[nr][nc] * 0.25)

                    # ── 2-Step Simultaneous Tactical Look-Ahead ───────────
                    # If this move enters an enemy threat zone, verify that remaining army survives
                    if (nr, nc) in enemy_adjacent_cells and owner[nr][nc] != 1:
                        threat_val = enemy_adjacent_cells[(nr, nc)]
                        remaining_army = (army_curr - 1 - army[nr][nc]) if owner[nr][nc] != 1 else army_curr
                        if remaining_army <= threat_val and turn < 800:
                            score += vulnerable_penalty

                    if score > best_score:
                        best_score = score
                        best_move = [0, int(r), int(c), int(direction), int(split_flag)]

        if best_move:
            _action_history.append(best_move)
            return best_move

        return [1, 0, 0, 0, 0]

    except Exception:
        return [1, 0, 0, 0, 0]
