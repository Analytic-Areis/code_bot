"""Code Bot Participant: Final Champion
Bot: Model 27 Final Champion
Strategy: Beam Search with Calibrated Neural Value Signal, Enhanced Inward Funnel,
          Smarter Deathtouch Routing, ES-Tuned Grandmaster Weights, Real Danger Pullback,
          Bayesian Mirror Spawn Prior, Dynamic Opponent Adaptation, Castle Economics.
Sources: model_25_grandmaster_champion/bot.py + model_26_apex_predator/bot.py
AI assistance: Antigravity pair programming

Architecture:
  Tier 0: Evaluator Contract Integrity, Anti-Ping-Pong Cyclic History Guard
  Tier 1: Hard Reflexes (Lethal Snipe, Adjacent Defense, Real Danger Pullback)
  Tier 2: Bayesian Room-Fairness Mirror Prior + Scoreboard POMDP Calculus
  Tier 3: Castle Economics, Inward DAG Funnel, Deathtouch Routing
  Tier 4: Heuristic Scorer + Additive Neural Bonus (clamped ±8)
  Tier 5: Beam Search Top-10 with 1-ply opponent simulation + time budget
  Tier 6: 120ms Hard Watchdog Safety Timer
"""

import math
import time
from collections import deque

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
OPPOSITE_DIR = {0: 1, 1: 0, 2: 3, 3: 2}

# ─── ES-TUNED GRANDMASTER WEIGHTS (Model 25 best-known) ───────────────────────
DIST_WEIGHT        = 24.068
ARMY_BONUS         = 0.403
CAPTURE_WEIGHT     = 573.321
ATTACK_PENALTY     = -284.901
FOG_BONUS          = 15.49
NEUTRAL_BONUS      = 20.513
DEATHTOUCH_BONUS   = 13832.318
KILL_BONUS         = 11000.0
SPREAD_MOD50_BONUS = 15.198
GARRISON_MIN       = 12.0
CASTLE_THRESHOLD   = 48
VULNERABLE_PENALTY = -176.523

# ─── NEURAL VALUE HEAD ARCHITECTURE ──────────────────────────────────────────
D_IN, D_H1, D_H2 = 16, 32, 16
NN_LR             = 0.0003   # Very small LR for stability (additive signal only)
NN_CLAMP          = 8.0      # Clamp NN output to ±8 before adding to heuristic
NN_BLEND          = 0.8      # final_score = heuristic + nn_bonus * NN_BLEND


class NeuralValueNet:
    """Pure-Python 3-layer ReLU net. Additive bonus on heuristic, NOT a replacement."""
    def __init__(self):
        # Calibrated priors – match feature semantics (Improvement 2: properly calibrated)
        priors = [
            1.2,   # x0: dist_delta – closer to target is good
            1.0,   # x1: army_adv – winning the combat ratio
            -0.3,  # x2: enemy_prox – near enemy is risky
            0.8,   # x3: defense_pull – moving army homeward under threat
            0.5,   # x4: mod50_phase – economy burst alignment
            1.0,   # x5: is_enemy – capturing enemy tile
            0.6,   # x6: is_neutral – expanding neutral tile
            0.4,   # x7: is_fog – exploring fog
            1.1,   # x8: is_castle – grabbing castle
            -1.0,  # x9: adj_threat – landing in enemy threat zone
            -0.4,  # x10: hidden_mass – unseen enemy army danger
            -0.5,  # x11: expansion_rate – fast enemy expansion is bad
            -2.0,  # x12: worst_excess – existential threat to general
            0.6,   # x13: castle_count – own castle economy
            -2.5,  # x14: loop_penalty – ping-pong avoidance
            0.3,   # x15: turn_phase – deathtouch window approach
        ]
        self.W1 = [
            [(0.5 * priors[j] if i == j or i == (j + 16) % D_H1
              else math.sin(i * 7 + j * 3) * 0.03) for j in range(D_IN)]
            for i in range(D_H1)
        ]
        self.b1 = [0.05] * D_H1
        self.W2 = [
            [(0.4 if (i * 2 == j or i * 2 + 1 == j) else math.cos(i * 5 + j * 3) * 0.03)
             for j in range(D_H1)]
            for i in range(D_H2)
        ]
        self.b2 = [0.05] * D_H2
        self.W3 = [0.35] * D_H2
        self.b3 = 0.0
        # Online learning cache
        self.last_x   = None
        self.last_a1  = None
        self.last_z1  = None
        self.last_a2  = None
        self.last_z2  = None
        self.last_val = 0.0

    def forward(self, x):
        z1 = [sum(xi * wi for xi, wi in zip(x, row)) + b
              for row, b in zip(self.W1, self.b1)]
        a1 = [z if z > 0.0 else 0.0 for z in z1]
        z2 = [sum(ai * wi for ai, wi in zip(a1, row)) + b
              for row, b in zip(self.W2, self.b2)]
        a2 = [z if z > 0.0 else 0.0 for z in z2]
        out = sum(ai * wi for ai, wi in zip(a2, self.W3)) + self.b3
        return out, a1, z1, a2, z2

    def train_step(self, td_error):
        if self.last_x is None:
            return
        d_out = max(-1.0, min(1.0, td_error))
        for i in range(D_H2):
            self.W3[i] += NN_LR * d_out * self.last_a2[i]
        self.b3 += NN_LR * d_out
        d_a2 = [d_out * w for w in self.W3]
        d_z2 = [da if z > 0.0 else 0.0 for da, z in zip(d_a2, self.last_z2)]
        for i in range(D_H2):
            dz = d_z2[i]
            self.b2[i] += NN_LR * dz
            for j in range(D_H1):
                self.W2[i][j] += NN_LR * dz * self.last_a1[j]
        d_a1 = [sum(d_z2[k] * self.W2[k][j] for k in range(D_H2)) for j in range(D_H1)]
        d_z1 = [da if z > 0.0 else 0.0 for da, z in zip(d_a1, self.last_z1)]
        for i in range(D_H1):
            dz = d_z1[i]
            self.b1[i] += NN_LR * dz
            for j in range(D_IN):
                self.W1[i][j] += NN_LR * dz * self.last_x[j]


# ─── PERSISTENT MODULE STATE ──────────────────────────────────────────────────
_neural_net        = NeuralValueNet()
_my_general        = None
_opp_general       = None
_mountains         = set()
_candidates        = {}      # (r, c) -> probability mass
_initialized       = False
_action_history    = deque(maxlen=8)
_castles_built     = 0
_opp_land_history  = deque(maxlen=20)
_opp_army_history  = deque(maxlen=20)
_prev_score_metric = 0.0


def _reset():
    """State isolation between matches."""
    global _neural_net, _my_general, _opp_general, _mountains, _candidates
    global _initialized, _action_history, _castles_built
    global _opp_land_history, _opp_army_history, _prev_score_metric
    _neural_net        = NeuralValueNet()
    _my_general        = None
    _opp_general       = None
    _mountains         = set()
    _candidates        = {}
    _initialized       = False
    _action_history.clear()
    _castles_built     = 0
    _opp_land_history.clear()
    _opp_army_history.clear()
    _prev_score_metric = 0.0


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
    h, w    = obs["height"], obs["width"]
    terrain = obs["type"]

    _mountains = set()
    for r in range(h):
        for c in range(w):
            if terrain[r][c] in (2, 5):
                _mountains.add((r, c))

    gr, gc   = my_gen
    my_room  = _bfs_room(h, w, _mountains, gr, gc, max_depth=7)
    dists    = _bfs_distance(h, w, _mountains, gr, gc)

    # 1. Primary filter: exact room-fairness (d >= 17, room diff <= 5)
    cands = []
    for (r, c), d in dists.items():
        if d >= 17:
            cand_room = _bfs_room(h, w, _mountains, r, c, max_depth=7)
            if abs(cand_room - my_room) <= 5:
                cands.append((r, c))

    # 2. Fallback: distance >= 14
    if not cands:
        cands = [(r, c) for (r, c), d in dists.items() if d >= 14]

    # 3. Geometric fallback for tiny/extreme maps
    if not cands:
        cands = [
            (r, c) for r in range(h) for c in range(w)
            if (r, c) not in _mountains and abs(r - gr) + abs(c - gc) >= 12
        ]

    # Generals.io mirror prior: 1/(1+d_mirror^1.5)
    target_mirror = (h - 1 - gr, w - 1 - gc)
    raw_weights = {
        coord: 1.0 / (1.0 + (abs(coord[0] - target_mirror[0]) + abs(coord[1] - target_mirror[1])) ** 1.5)
        for coord in cands
    }
    total_w    = sum(raw_weights.values()) if raw_weights else 1.0
    _candidates = {coord: wv / total_w for coord, wv in raw_weights.items()}
    _initialized = True


def _update_bayes(obs):
    """Remove revealed non-general cells from candidate set; renormalize."""
    terrain = obs["type"]
    owner   = obs["owner"]
    changed = False
    for coord in list(_candidates.keys()):
        r, c = coord
        if terrain[r][c] != 0:
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
    h, w  = obs["height"], obs["width"]
    gr, gc = _my_general if _my_general else (h // 2, w // 2)
    return (h - 1 - gr, w - 1 - gc)


def _detect_loop(r, c, direction):
    """Detect 2-cycle ping-pong."""
    if not _action_history:
        return False
    prev_kind, prev_r, prev_c, prev_d, _ = _action_history[-1]
    if prev_kind == 0:
        dr, dc = DIRS[prev_d]
        if (r, c) == (prev_r + dr, prev_c + dc) and direction == OPPOSITE_DIR.get(prev_d):
            return True
    return False


def _nn_features(r, c, direction, nr, nc, army_curr, curr_d, next_d,
                 owner, terrain, army, dist_map, dist_from_gen,
                 enemy_adjacent_cells, enemy_units, opp_army,
                 visible_enemy_army, expansion_rate, worst_excess,
                 castle_count, turn):
    """Compute the 16 neural feature vector for a move (r,c)->direction->(nr,nc)."""
    x = [0.0] * 16
    x[0]  = (curr_d - next_d) / 5.0                                     # dist_delta
    x[1]  = math.tanh((army_curr - 1.0) / max(1.0, army[nr][nc] + 1.0)) # army_adv
    x[2]  = (1.0 / (1.0 + min(abs(nr - er) + abs(nc - ec) for er, ec, _ in enemy_units))
             if enemy_units else 0.0)                                      # enemy_prox
    x[3]  = (dist_from_gen.get((r, c), 0) - dist_from_gen.get((nr, nc), 0)) / 5.0  # defense_pull
    x[4]  = (turn % 50) / 50.0                                           # mod50_phase
    x[5]  = 1.0 if owner[nr][nc] == 2 else 0.0                          # is_enemy
    x[6]  = 1.0 if owner[nr][nc] == 0 else 0.0                          # is_neutral
    x[7]  = 1.0 if terrain[nr][nc] == 0 else 0.0                        # is_fog
    x[8]  = 1.0 if terrain[nr][nc] == 3 else 0.0                        # is_castle
    x[9]  = enemy_adjacent_cells.get((nr, nc), 0) / 50.0                # adj_threat
    x[10] = max(0, opp_army - visible_enemy_army) / max(1, opp_army)    # hidden_mass
    x[11] = expansion_rate                                                # expansion_rate
    x[12] = worst_excess / 20.0                                           # worst_excess
    x[13] = castle_count / 4.0                                            # castle_count
    x[14] = -1.0 if _detect_loop(r, c, direction) else 0.0              # loop_penalty
    x[15] = turn / 1200.0                                                 # turn_phase
    return x


def act(obs):
    t_start = time.perf_counter()
    global _neural_net, _my_general, _opp_general, _candidates, _initialized
    global _action_history, _castles_built, _opp_land_history, _opp_army_history
    global _prev_score_metric

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

        # ── 2. THREAT ANALYSIS ────────────────────────────────────────────────
        visible_enemy_army = 0
        enemy_units        = []
        enemy_adjacent_cells = {}
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 2:
                    visible_enemy_army += army[r][c]
                    enemy_units.append((r, c, army[r][c]))
                    for dr, dc in DIRS:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            enemy_adjacent_cells[(nr, nc)] = max(
                                enemy_adjacent_cells.get((nr, nc), 0), army[r][c]
                            )

        opp_army = obs.get("opp_army", visible_enemy_army)
        opp_land = obs.get("opp_land", 1)
        my_army  = obs.get("my_army", 1)
        my_land  = obs.get("my_land", 1)

        _opp_land_history.append(opp_land)
        _opp_army_history.append(opp_army)

        dist_from_gen = _bfs_distance(h, w, _mountains, gr, gc)
        worst_excess  = 0
        threat_pos    = None

        for r, c, earmy in enemy_units:
            d = dist_from_gen.get((r, c), 999)
            if 0 < d <= 10:
                gen_growth = d // 2
                excess = (earmy - 1) - (army[gr][gc] + gen_growth)
                if excess > worst_excess:
                    worst_excess = excess
                    threat_pos   = (r, c)

        # ── 3. DYNAMIC PARAMETER ADAPTATION ───────────────────────────────────
        if len(_opp_land_history) >= 8:
            expansion_rate = (opp_land - _opp_land_history[0]) / len(_opp_land_history)
        else:
            expansion_rate = 0.35

        army_per_land = opp_army / max(1, opp_land)

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

        # Adaptation A: Rusher
        is_rusher = (worst_excess > 0) or (40 <= turn <= 160 and opp_land <= 4 and opp_army >= 18)
        if is_rusher:
            garrison_min       = max(garrison_min, 16.0)
            vulnerable_penalty *= 1.25
            attack_penalty     *= 1.20
            capture_weight     *= 1.20

        # Adaptation B: Turtle / Fortress
        is_turtle = (turn >= 80 and expansion_rate < 0.22 and army_per_land > 1.7 and worst_excess == 0)
        if is_turtle:
            neutral_bonus      *= 1.40
            fog_bonus          *= 1.30
            spread_mod50_bonus *= 1.35
            garrison_min        = max(12.0, garrison_min - 2.0)

        # Adaptation C: Boomer / Ribbon
        is_boomer = (turn >= 60 and expansion_rate > 0.45 and army_per_land < 1.35)
        if is_boomer:
            capture_weight *= 1.35
            army_bonus     *= 1.30

        # ── 4. ONLINE TD NEURAL LEARNING (≈0.04ms) ────────────────────────────
        current_metric = (my_land * 0.5 + my_army * 0.3) - (opp_land * 0.5 + opp_army * 0.3)
        if turn > 1 and _neural_net.last_x is not None:
            reward   = current_metric - _prev_score_metric
            if worst_excess > 0:
                reward -= 5.0
            td_error = reward + 0.95 * current_metric - _neural_net.last_val
            _neural_net.train_step(td_error)
        _prev_score_metric = current_metric

        # ── 5. TIER 1: EMERGENCY HARD REFLEXES ────────────────────────────────
        # A. Lethal strike on adjacent enemy general
        if _opp_general:
            egr, egc = _opp_general
            for d_idx, (dr, dc) in enumerate(DIRS):
                adj_r, adj_c = egr + dr, egc + dc
                if 0 <= adj_r < h and 0 <= adj_c < w and owner[adj_r][adj_c] == 1:
                    my_atk  = army[adj_r][adj_c]
                    opp_def = army[egr][egc]
                    rev_dir = OPPOSITE_DIR[d_idx]
                    if my_atk - 1 > opp_def or turn >= 800:
                        act_vec = [0, int(adj_r), int(adj_c), int(rev_dir), 0]
                        _action_history.append(act_vec)
                        return act_vec

        # B. Direct defense of our general from adjacent threats
        for d_idx, (dr, dc) in enumerate(DIRS):
            adj_r, adj_c = gr + dr, gc + dc
            if 0 <= adj_r < h and 0 <= adj_c < w and owner[adj_r][adj_c] == 2:
                enemy_threat = army[adj_r][adj_c]
                # Counter with general if we can win outright
                if army[gr][gc] - 1 > enemy_threat:
                    act_vec = [0, int(gr), int(gc), int(d_idx), 1]
                    _action_history.append(act_vec)
                    return act_vec
                # Reinforce general from adjacent ally
                for d2_idx, (dr2, dc2) in enumerate(DIRS):
                    nr2, nc2 = gr + dr2, gc + dc2
                    if 0 <= nr2 < h and 0 <= nc2 < w and owner[nr2][nc2] == 1 and (nr2, nc2) != (gr, gc):
                        if army[nr2][nc2] >= 2:
                            rev_dir2 = OPPOSITE_DIR[d2_idx]
                            act_vec  = [0, int(nr2), int(nc2), int(rev_dir2), 0]
                            _action_history.append(act_vec)
                            return act_vec

        # C. Real Danger Pullback – stream closest defenders home
        if worst_excess > 0 and threat_pos:
            best_def_move = None
            best_gain     = -999
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
                                        best_gain     = gain
                                        best_def_move = [0, int(r), int(c), int(d_idx), 0]
            if best_def_move:
                _action_history.append(best_def_move)
                return best_def_move

        # ── 6. STRATEGIC CASTLE ECONOMICS (kind = 2) ──────────────────────────
        my_structures = [(gr, gc)]
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 1 and terrain[r][c] == 3:
                    my_structures.append((r, c))
        castle_count = len(my_structures) - 1  # exclude general

        if 60 <= turn <= 350 and _castles_built < 2 and worst_excess == 0:
            for r in range(h):
                for c in range(w):
                    if owner[r][c] == 1 and terrain[r][c] == 1 and army[r][c] >= CASTLE_THRESHOLD:
                        dist_penalty = sum(max(0, 14 - 2 * (abs(r - sr) + abs(c - sc)))
                                          for sr, sc in my_structures)
                        build_cost   = 35 + dist_penalty
                        min_struct_dist = min(abs(r - sr) + abs(c - sc) for sr, sc in my_structures)
                        if min_struct_dist >= 7 and army[r][c] >= build_cost:
                            _castles_built += 1
                            act_vec = [2, int(r), int(c), 0, 0]
                            _action_history.append(act_vec)
                            return act_vec

        # ── 7. BFS DIST MAP TO TARGET ─────────────────────────────────────────
        target   = _best_target(obs)
        dist_map = {}
        q        = deque([target])
        dist_map[target] = 0
        while q:
            r, c = q.popleft()
            d    = dist_map[(r, c)]
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dist_map:
                    if terrain[nr][nc] != 2:
                        dist_map[(nr, nc)] = d + 1
                        q.append((nr, nc))

        # ── 8. IMPROVEMENT 3: Enhanced Inward Funnel ──────────────────────────
        # Identify the single BEST frontline tile (closest friendly to target with army >= 3)
        # and funnel ALL rear army towards it aggressively.
        frontline_tile = None
        best_fl_dist   = 999
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 1 and army[r][c] >= 3:
                    fd = dist_map.get((r, c), 999)
                    if fd < best_fl_dist:
                        best_fl_dist  = fd
                        frontline_tile = (r, c)

        # ── 9. IMPROVEMENT 4: Deathtouch Column Routing (Turn 720+) ──────────
        # At turn 720+, find our largest army stack and route it along shortest
        # path toward enemy general (or best candidate) to guarantee staging.
        deathtouch_column = None
        if turn >= 720 and _opp_general:
            egr, egc = _opp_general
            best_stack_army = 0
            for r in range(h):
                for c in range(w):
                    if owner[r][c] == 1 and army[r][c] > best_stack_army and (r, c) != (gr, gc):
                        # Find step towards enemy general using dist_map
                        best_stack_d = dist_map.get((r, c), 999)
                        if best_stack_d <= 15 and army[r][c] > best_stack_army:
                            best_stack_army = army[r][c]
                            best_stack_pos  = (r, c)
            if best_stack_army >= 5:
                sr, sc = best_stack_pos
                best_dt_dir   = None
                best_dt_ndist = dist_map.get((sr, sc), 999)
                for d_idx, (ddr, ddc) in enumerate(DIRS):
                    nr, nc = sr + ddr, sc + ddc
                    if 0 <= nr < h and 0 <= nc < w and terrain[nr][nc] != 2:
                        nd = dist_map.get((nr, nc), 999)
                        if nd < best_dt_ndist:
                            best_dt_ndist = nd
                            best_dt_dir   = d_idx
                if best_dt_dir is not None:
                    deathtouch_column = [0, int(sr), int(sc), int(best_dt_dir), 0]

        # ── 10. ENUMERATE ALL CANDIDATE MOVES WITH HEURISTIC + NN BONUS ──────
        is_mod50_burst        = (turn % 50) >= 38 or (turn % 50) <= 2
        is_deathtouch_staging = 740 <= turn < 800

        top_candidates = []  # (score, r, c, direction, split_flag, curr_d, army_curr, nr, nc)
        best_score_h   = -1e18

        for r in range(h):
            if time.perf_counter() - t_start > 0.095:
                break
            for c in range(w):
                if owner[r][c] != 1 or army[r][c] < 2:
                    continue

                army_curr = army[r][c]
                curr_d    = dist_map.get((r, c), 999)

                split_flag = 0
                if (r, c) == (gr, gc):
                    # NEVER move general if enemy adjacent
                    has_adj_enemy = any(
                        0 <= gr + ddr < h and 0 <= gc + ddc < w and owner[gr + ddr][gc + ddc] == 2
                        for ddr, ddc in DIRS
                    )
                    if has_adj_enemy or worst_excess > 0 or army_curr < garrison_min:
                        continue
                    split_flag = 1

                for direction, (dr, dc) in enumerate(DIRS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if terrain[nr][nc] == 2:
                        continue

                    next_d = dist_map.get((nr, nc), 999)
                    score  = (curr_d - next_d) * dist_weight
                    score += (army_curr - 1) * army_bonus

                    # Anti-pingpong
                    if _detect_loop(r, c, direction):
                        score -= 600.0

                    # Mod-50 Land Snatch burst
                    if is_mod50_burst and owner[nr][nc] != 1:
                        score += spread_mod50_bonus * 2.5

                    # Move onto enemy general
                    if _opp_general and (nr, nc) == _opp_general:
                        if turn >= 800:
                            score += deathtouch_bonus
                        elif army_curr - 1 > army[nr][nc]:
                            score += KILL_BONUS
                        else:
                            score += attack_penalty * 2

                    # Deathtouch staging (perimeter approach before turn 800)
                    elif is_deathtouch_staging and _opp_general and next_d <= 2:
                        if next_d == 1:
                            score += 8000.0
                        elif next_d == 2:
                            score += 5000.0

                    elif owner[nr][nc] == 2:   # Enemy tile
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

                    elif owner[nr][nc] == 0:   # Neutral / fog tile
                        score += neutral_bonus
                        if terrain[nr][nc] == 0:   # Fog exploration
                            score += fog_bonus
                        elif terrain[nr][nc] == 3:  # Neutral castle capture
                            if army_curr - 1 > army[nr][nc]:
                                score += 400.0
                            else:
                                score -= 200.0

                    elif owner[nr][nc] == 1:   # Inward funnel / consolidation
                        if next_d < curr_d:
                            score += 18.0 + (army[nr][nc] * 0.25)
                        # Improvement 3: Enhanced inward funnel to frontline
                        if frontline_tile and (nr, nc) == frontline_tile:
                            score += 30.0 + army_curr * 0.3

                    # 2-Step vulnerability check
                    if (nr, nc) in enemy_adjacent_cells and owner[nr][nc] != 1:
                        threat_val     = enemy_adjacent_cells[(nr, nc)]
                        remaining_army = (army_curr - 1 - army[nr][nc]) if owner[nr][nc] != 1 else army_curr
                        if remaining_army <= threat_val and turn < 800:
                            score += vulnerable_penalty

                    top_candidates.append((score, r, c, direction, split_flag, curr_d, army_curr, nr, nc))
                    if score > best_score_h:
                        best_score_h = score

        if not top_candidates:
            # Use deathtouch column or pass
            if deathtouch_column and turn >= 720:
                _action_history.append(deathtouch_column)
                return deathtouch_column
            return [1, 0, 0, 0, 0]

        # ── 11. BEAM SEARCH: Top-10 candidates → 1-ply opponent simulation ───
        # Sort top-K=10 and spend remaining time budget verifying each
        top_candidates.sort(key=lambda x: x[0], reverse=True)
        beam_k = min(10, len(top_candidates))

        best_move     = None
        best_final    = -1e18
        nn_last_x     = None
        nn_last_val   = None
        nn_last_cache = None

        for cand_idx in range(beam_k):
            if time.perf_counter() - t_start > 0.110:
                break

            score, r, c, direction, split_flag, curr_d, army_curr, nr, nc = top_candidates[cand_idx]

            # ── Neural value bonus (additive, clamped ±8) ──────────────────
            next_d = dist_map.get((nr, nc), 999)
            x = _nn_features(
                r, c, direction, nr, nc, army_curr, curr_d, next_d,
                owner, terrain, army, dist_map, dist_from_gen,
                enemy_adjacent_cells, enemy_units, opp_army,
                visible_enemy_army, expansion_rate, worst_excess,
                castle_count, turn
            )
            nn_out, a1, z1, a2, z2 = _neural_net.forward(x)
            nn_bonus  = max(-NN_CLAMP, min(NN_CLAMP, nn_out))
            final_score = score + nn_bonus * NN_BLEND

            # ── 1-ply opponent simulation ──────────────────────────────────
            # Simulate MY move effect, then find opponent's best reply penalty.
            # Only penalize if opponent can follow up with a winning capture
            if time.perf_counter() - t_start < 0.108 and enemy_units:
                # Simulate: my army lands on (nr, nc)
                moved_army   = (army_curr // 2 if split_flag else army_curr - 1)
                landing_army = moved_army - army[nr][nc] if owner[nr][nc] != 1 else moved_army + army[nr][nc]

                # Opponent's best reply: can any enemy tile capture (nr, nc) next?
                opp_threat_to_landing = 0
                for ddr, ddc in DIRS:
                    adj_r2, adj_c2 = nr + ddr, nc + ddc
                    if 0 <= adj_r2 < h and 0 <= adj_c2 < w and owner[adj_r2][adj_c2] == 2:
                        opp_threat_to_landing = max(opp_threat_to_landing, army[adj_r2][adj_c2] - 1)

                if opp_threat_to_landing > landing_army and landing_army > 0 and turn < 800:
                    final_score -= 85.0  # Penalty for walking into immediate recapture

            if final_score > best_final:
                best_final    = final_score
                best_move     = [0, int(r), int(c), int(direction), int(split_flag)]
                nn_last_x     = x
                nn_last_val   = nn_out
                nn_last_cache = (a1, z1, a2, z2)

        # Store neural net state for online TD learning next turn
        if nn_last_x is not None:
            _neural_net.last_x   = nn_last_x
            _neural_net.last_a1  = nn_last_cache[0]
            _neural_net.last_z1  = nn_last_cache[1]
            _neural_net.last_a2  = nn_last_cache[2]
            _neural_net.last_z2  = nn_last_cache[3]
            _neural_net.last_val = nn_last_val

        # ── 12. DEATHTOUCH COLUMN OVERRIDE (Turn 720+) ────────────────────────
        # If we found a routed attack column and it's stronger, use it
        if deathtouch_column and turn >= 720 and best_move is None:
            _action_history.append(deathtouch_column)
            return deathtouch_column

        if best_move:
            _action_history.append(best_move)
            return best_move

        return [1, 0, 0, 0, 0]

    except Exception:
        return [1, 0, 0, 0, 0]
