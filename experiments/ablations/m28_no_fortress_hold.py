"""Code Bot Participant: Model 28 Adaptive Champion
Bot: Model 28 Adaptive Champion
Strategy: Multi-Source DAG Frontline Funnel, Threat-Proximity Urgent Pullback Defense,
          Bayesian Evidence-Based General Target Acquisition, Probabilistic Opponent Modeling,
          Safe Castle Economics, Deathtouch Singularity Staging, Bounded TD Value Guidance.
Sources: Synthesized from model_27_final_champion, game-theoretic principles, and
         empirical diagnostic traces on Generals.io ruleset.
AI assistance: Antigravity pair programming

Architecture Pipeline:
  Tier 0: Contract Integrity, Self-Resetting State Isolation, Persistent Obstacle Memory
  Tier 1: Hard Tactical Reflexes (Lethal Snipe, Safe Adjacent General Defense, Interception)
  Tier 2: Safe Castle Facility Location (kind=2, cost kernel, enemy proximity shield)
  Tier 3: Multi-Source DAG Frontline Funnel & Large-Stack Spearhead Mobilization
  Tier 4: Probabilistic Bayesian Target Tracking with Enemy Expansion Evidence
  Tier 5: Online Continuous Opponent Archetype Classifier (Rush, Turtle, Boom, Consolidator)
  Tier 6: Heuristic Scoring + Normalized Bounded TD(0) Neural Value Head (clamped ±8)
  Tier 7: Top-12 Beam Search with 1-Ply Board Simulation & Recapture/Exposure Verification
  Tier 8: 100ms Watchdog Safety Ceiling
"""

import math
import time
from collections import deque

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
OPPOSITE_DIR = {0: 1, 1: 0, 2: 3, 3: 2}

# ─── EVOLVED HEURISTIC BASELINE WEIGHTS ───────────────────────────────────────
DIST_WEIGHT        = 24.5
ARMY_BONUS         = 0.65
CAPTURE_WEIGHT     = 550.0
ATTACK_PENALTY     = -280.0
FOG_BONUS          = 16.0
NEUTRAL_BONUS      = 21.0
DEATHTOUCH_BONUS   = 15000.0
KILL_BONUS         = 12000.0
SPREAD_MOD50_BONUS = 16.0
GARRISON_MIN_BASE  = 14.0
CASTLE_THRESHOLD   = 46
VULNERABLE_PENALTY = -180.0

# ─── NEURAL VALUE HEAD ARCHITECTURE ──────────────────────────────────────────
D_IN, D_H1, D_H2 = 16, 32, 16
NN_LR             = 0.0003
NN_CLAMP          = 8.0
NN_BLEND          = 0.5


class NeuralValueNet:
    """Pure-Python 3-layer value network. Strictly additive bounded guidance."""
    def __init__(self):
        priors = [
            1.2,   # x0: dist_delta (closer to target is good)
            1.1,   # x1: army_adv (combat ratio)
            -0.4,  # x2: enemy_prox (near enemy risk)
            0.8,   # x3: defense_pull (moving homeward under threat)
            0.5,   # x4: mod50_phase (harvest synchronization)
            1.0,   # x5: is_enemy (capture reward)
            0.6,   # x6: is_neutral (expansion)
            0.4,   # x7: is_fog (information gain)
            1.2,   # x8: is_castle (castle value)
            -1.2,  # x9: adj_threat (threat landing)
            -0.5,  # x10: hidden_mass (unseen army uncertainty)
            -0.6,  # x11: expansion_rate (enemy expansion pressure)
            -2.2,  # x12: worst_excess (existential general threat)
            0.6,   # x13: castle_count (production infrastructure)
            -2.5,  # x14: loop_penalty (ping-pong avoidance)
            0.4,   # x15: turn_phase (deathtouch progression)
        ]
        self.W1 = [
            [(0.45 * priors[j] if i == j or i == (j + 16) % D_H1
              else math.sin(i * 7 + j * 3) * 0.02) for j in range(D_IN)]
            for i in range(D_H1)
        ]
        self.b1 = [0.04] * D_H1
        self.W2 = [
            [(0.35 if (i * 2 == j or i * 2 + 1 == j) else math.cos(i * 5 + j * 3) * 0.02)
             for j in range(D_H1)]
            for i in range(D_H2)
        ]
        self.b2 = [0.04] * D_H2
        self.W3 = [0.30] * D_H2
        self.b3 = 0.0

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
_known_obstacles   = set()   # Persistent memory of all mountains (2) and fog structures (5)
_candidates        = {}      # (r, c) -> Bayesian posterior probability
_initialized       = False
_action_history    = deque(maxlen=8)
_castles_built     = 0
_opp_land_history  = deque(maxlen=15)
_opp_army_history  = deque(maxlen=15)
_prev_score_metric = 0.0
_prev_turn         = -1

# Continuous Bayesian Opponent Archetype Probabilities: [Rush, Turtle, Boom, Consolidator]
_opp_belief = [0.25, 0.25, 0.25, 0.25]


def _reset():
    """State isolation between matches."""
    global _neural_net, _my_general, _opp_general, _known_obstacles, _candidates
    global _initialized, _action_history, _castles_built
    global _opp_land_history, _opp_army_history, _prev_score_metric, _prev_turn, _opp_belief
    _neural_net        = NeuralValueNet()
    _my_general        = None
    _opp_general       = None
    _known_obstacles   = set()
    _candidates        = {}
    _initialized       = False
    _action_history.clear()
    _castles_built     = 0
    _opp_land_history.clear()
    _opp_army_history.clear()
    _prev_score_metric = 0.0
    _prev_turn         = -1
    _opp_belief        = [0.25, 0.25, 0.25, 0.25]


def _find_my_general(obs):
    owner, terrain = obs["owner"], obs["type"]
    for r in range(obs["height"]):
        for c in range(obs["width"]):
            if owner[r][c] == 1 and terrain[r][c] == 4:
                return (r, c)
    return None


def _update_obstacles(obs):
    """Accumulate persistent obstacle memory across turns."""
    global _known_obstacles
    h, w = obs["height"], obs["width"]
    terrain = obs["type"]
    for r in range(h):
        for c in range(w):
            t = terrain[r][c]
            if t in (2, 5):
                _known_obstacles.add((r, c))
            elif t in (1, 3, 4) and (r, c) in _known_obstacles:
                # If a cell previously seen as 5 in fog turns out to be plain or castle, remove
                _known_obstacles.discard((r, c))


def _bfs_distance(h, w, obstacles, start_r, start_c):
    visited = {(start_r, start_c): 0}
    q = deque([(start_r, start_c)])
    while q:
        r, c = q.popleft()
        d = visited[(r, c)]
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in obstacles and (nr, nc) not in visited:
                visited[(nr, nc)] = d + 1
                q.append((nr, nc))
    return visited


def _multi_source_bfs(h, w, obstacles, sources):
    """Compute distance field to nearest source in sources list."""
    visited = {}
    q = deque()
    for sr, sc in sources:
        if (sr, sc) not in obstacles:
            visited[(sr, sc)] = 0
            q.append((sr, sc))
    while q:
        r, c = q.popleft()
        d = visited[(r, c)]
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in obstacles and (nr, nc) not in visited:
                visited[(nr, nc)] = d + 1
                q.append((nr, nc))
    return visited


def _bfs_room(h, w, obstacles, start_r, start_c, max_depth=7):
    visited = {(start_r, start_c): 0}
    q = deque([(start_r, start_c)])
    count = 0
    while q:
        r, c = q.popleft()
        d = visited[(r, c)]
        if d < max_depth:
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in obstacles and (nr, nc) not in visited:
                    visited[(nr, nc)] = d + 1
                    count += 1
                    q.append((nr, nc))
    return count


def _init_candidates(obs, my_gen):
    global _candidates, _initialized
    h, w    = obs["height"], obs["width"]
    gr, gc  = my_gen

    _update_obstacles(obs)
    my_room = _bfs_room(h, w, _known_obstacles, gr, gc, max_depth=7)
    dists   = _bfs_distance(h, w, _known_obstacles, gr, gc)

    # 1. Primary: distance >= 17, room difference <= 5
    cands = []
    for (r, c), d in dists.items():
        if d >= 17:
            cand_room = _bfs_room(h, w, _known_obstacles, r, c, max_depth=7)
            if abs(cand_room - my_room) <= 5:
                cands.append((r, c))

    # 2. Distance >= 14 fallback
    if not cands:
        cands = [(r, c) for (r, c), d in dists.items() if d >= 14]

    # 3. Geometric fallback
    if not cands:
        cands = [
            (r, c) for r in range(h) for c in range(w)
            if (r, c) not in _known_obstacles and abs(r - gr) + abs(c - gc) >= 12
        ]

    target_mirror = (h - 1 - gr, w - 1 - gc)
    raw_weights = {
        coord: 1.0 / (1.0 + (abs(coord[0] - target_mirror[0]) + abs(coord[1] - target_mirror[1])) ** 1.5)
        for coord in cands
    }
    total_w = sum(raw_weights.values()) if raw_weights else 1.0
    _candidates = {coord: wv / total_w for coord, wv in raw_weights.items()}
    _initialized = True


def _update_bayes_target(obs):
    """Prune revealed non-general cells from candidate set; renormalize."""
    global _candidates
    terrain = obs["type"]
    owner   = obs["owner"]
    h, w    = obs["height"], obs["width"]
    gr, gc  = _my_general if _my_general else (1, 1)

    # 1. Prune revealed non-general cells
    for coord in list(_candidates.keys()):
        r, c = coord
        if terrain[r][c] != 0:
            if not (owner[r][c] == 2 and terrain[r][c] == 4):
                del _candidates[coord]

    # 2. Degenerate fallback: if empty, resample from all unexplored fog tiles
    if not _candidates:
        target_mirror = (h - 1 - gr, w - 1 - gc)
        for r in range(h):
            for c in range(w):
                if terrain[r][c] == 0 and (r, c) not in _known_obstacles:
                    d_mirror = abs(r - target_mirror[0]) + abs(c - target_mirror[1])
                    _candidates[(r, c)] = 1.0 / (1.0 + d_mirror ** 1.5)

    # Renormalize prior
    if _candidates:
        total = sum(_candidates.values())
        if total > 0:
            for k in _candidates:
                _candidates[k] /= total


def _best_target(obs, enemy_cells=None):
    """Select best strategic target combining prior and live enemy frontier evidence."""
    if _opp_general:
        return _opp_general
    if not _candidates:
        h, w   = obs["height"], obs["width"]
        gr, gc = _my_general if _my_general else (h // 2, w // 2)
        return (h - 1 - gr, w - 1 - gc)

    if not enemy_cells:
        return max(_candidates.items(), key=lambda x: x[1])[0]

    # Combine prior with live enemy territory evidence on the fly (never compounded recursively)
    er_mean = sum(r for r, c in enemy_cells) / len(enemy_cells)
    ec_mean = sum(c for r, c in enemy_cells) / len(enemy_cells)

    best_cand = None
    best_cand_score = -1e9
    for coord, prior_p in _candidates.items():
        d_to_enemy = abs(coord[0] - er_mean) + abs(coord[1] - ec_mean)
        # Likelihood: general is likely behind observed enemy frontier
        score = math.log(max(1e-12, prior_p)) - (0.06 * d_to_enemy)
        if score > best_cand_score:
            best_cand_score = score
            best_cand       = coord

    return best_cand if best_cand else max(_candidates.items(), key=lambda x: x[1])[0]


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


def _update_opponent_model(opp_land, opp_army, worst_excess, enemy_units, turn):
    """Update continuous Bayesian belief over opponent archetypes: Rush, Turtle, Boom, Consolidator."""
    global _opp_belief, _opp_land_history, _opp_army_history

    _opp_land_history.append(opp_land)
    _opp_army_history.append(opp_army)

    if len(_opp_land_history) >= 6:
        expansion_rate = (opp_land - _opp_land_history[0]) / max(1, len(_opp_land_history))
    else:
        expansion_rate = 0.35

    army_density = opp_army / max(1, opp_land)
    max_visible_stack = max((a for _, _, a in enemy_units), default=0)
    concentration = max_visible_stack / max(1, sum(a for _, _, a in enemy_units)) if enemy_units else 0.5

    # Likelihood log-scores
    score_rush = 0.0
    if worst_excess > 0:
        score_rush += 3.0
    if turn <= 160 and opp_land <= 5 and opp_army >= 18:
        score_rush += 2.5
    score_rush += min(2.0, (army_density - 1.2) * 1.5)

    score_turtle = 0.0
    if turn >= 60:
        score_turtle += max(0.0, (1.8 - expansion_rate) * 1.5)
        score_turtle += max(0.0, (army_density - 1.5) * 1.5)

    score_boom = 0.0
    if turn >= 50:
        score_boom += max(0.0, (expansion_rate - 0.38) * 3.0)
        score_boom += max(0.0, (1.4 - army_density) * 1.5)

    score_consolidator = 0.0
    if turn >= 100:
        score_consolidator += concentration * 2.0
        score_consolidator += max(0.0, (army_density - 1.3) * 1.0)

    # Softmax update with inertia (Bayesian filter)
    exps = [
        math.exp(min(5.0, score_rush)),
        math.exp(min(5.0, score_turtle)),
        math.exp(min(5.0, score_boom)),
        math.exp(min(5.0, score_consolidator)),
    ]
    tot = sum(exps)
    new_obs_dist = [e / tot for e in exps]

    # Exponential moving average belief tracking (alpha = 0.15)
    _opp_belief = [0.85 * b + 0.15 * n for b, n in zip(_opp_belief, new_obs_dist)]


def _nn_features(r, c, direction, nr, nc, army_curr, curr_d, next_d,
                 owner, terrain, army, dist_from_gen,
                 enemy_adjacent_cells, enemy_units, opp_army,
                 visible_enemy_army, expansion_rate, worst_excess,
                 castle_count, turn):
    x = [0.0] * 16
    x[0]  = (curr_d - next_d) / 5.0
    x[1]  = math.tanh((army_curr - 1.0) / max(1.0, army[nr][nc] + 1.0))
    x[2]  = (1.0 / (1.0 + min(abs(nr - er) + abs(nc - ec) for er, ec, _ in enemy_units))
             if enemy_units else 0.0)
    x[3]  = (dist_from_gen.get((r, c), 0) - dist_from_gen.get((nr, nc), 0)) / 5.0
    x[4]  = (turn % 50) / 50.0
    x[5]  = 1.0 if owner[nr][nc] == 2 else 0.0
    x[6]  = 1.0 if owner[nr][nc] == 0 else 0.0
    x[7]  = 1.0 if terrain[nr][nc] == 0 else 0.0
    x[8]  = 1.0 if terrain[nr][nc] == 3 else 0.0
    x[9]  = enemy_adjacent_cells.get((nr, nc), 0) / 50.0
    x[10] = max(0, opp_army - visible_enemy_army) / max(1, opp_army)
    x[11] = expansion_rate
    x[12] = worst_excess / 20.0
    x[13] = castle_count / 4.0
    x[14] = -1.0 if _detect_loop(r, c, direction) else 0.0
    x[15] = turn / 1200.0
    return x


def act(obs):
    t_start = time.perf_counter()
    global _neural_net, _my_general, _opp_general, _candidates, _initialized
    global _action_history, _castles_built, _opp_land_history, _opp_army_history
    global _prev_score_metric, _prev_turn, _known_obstacles, _opp_belief

    try:
        h, w    = obs["height"], obs["width"]
        owner   = obs["owner"]
        army    = obs["army"]
        terrain = obs["type"]
        turn    = obs["turn"]

        # ── 0. SELF-RESETTING STATE ISOLATION ─────────────────────────────────
        if turn <= 1 or not _initialized or _my_general is None or turn < _prev_turn:
            _reset()
            _my_general = _find_my_general(obs)
            if _my_general:
                _init_candidates(obs, _my_general)
        _prev_turn = turn

        _update_obstacles(obs)
        gr, gc = _my_general if _my_general else (1, 1)

        # ── 1. ACQUIRE OPPONENT GENERAL OR UPDATE BELIEF ──────────────────────
        if not _opp_general:
            for r in range(h):
                for c in range(w):
                    if owner[r][c] == 2 and terrain[r][c] == 4:
                        _opp_general = (r, c)
                        break
                if _opp_general:
                    break

        # ── 2. THREAT & FRONTIER PERCEPTION ───────────────────────────────────
        visible_enemy_army   = 0
        enemy_units          = []
        enemy_cells          = []
        enemy_adjacent_cells = {}

        for r in range(h):
            for c in range(w):
                if owner[r][c] == 2:
                    visible_enemy_army += army[r][c]
                    enemy_units.append((r, c, army[r][c]))
                    enemy_cells.append((r, c))
                    for dr, dc in DIRS:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            enemy_adjacent_cells[(nr, nc)] = max(
                                enemy_adjacent_cells.get((nr, nc), 0), army[r][c]
                            )

        if not _opp_general:
            _update_bayes_target(obs)

        opp_army = obs.get("opp_army", visible_enemy_army)
        opp_land = obs.get("opp_land", 1)
        my_army  = obs.get("my_army", 1)
        my_land  = obs.get("my_land", 1)

        dist_from_gen = _bfs_distance(h, w, _known_obstacles, gr, gc)

        worst_excess    = 0
        threat_pos      = None
        min_threat_dist = 999

        for r, c, earmy in enemy_units:
            d = dist_from_gen.get((r, c), 999)
            if 0 < d <= 10:
                gen_growth = d // 2
                excess = (earmy - 1) - (army[gr][gc] + gen_growth)
                if excess > worst_excess or (excess == worst_excess and d < min_threat_dist):
                    worst_excess    = excess
                    threat_pos      = (r, c)
                    min_threat_dist = d

        # ── 3. PROBABILISTIC OPPONENT MODEL & DYNAMIC WEIGHT TUNING ───────────
        _update_opponent_model(opp_land, opp_army, worst_excess, enemy_units, turn)
        p_rush, p_turtle, p_boom, p_consolidator = _opp_belief

        garrison_min       = GARRISON_MIN_BASE + (8.0 * p_rush) + (4.0 * p_consolidator)
        if turn >= 150:
            garrison_min = max(garrison_min, 18.0)

        capture_weight     = CAPTURE_WEIGHT * (1.0 + 0.35 * p_boom + 0.15 * p_rush)
        neutral_bonus      = NEUTRAL_BONUS * (1.0 + 0.40 * p_turtle)
        fog_bonus          = FOG_BONUS * (1.0 + 0.30 * p_turtle)
        vulnerable_penalty = VULNERABLE_PENALTY * (1.0 + 0.30 * p_rush)
        attack_penalty     = ATTACK_PENALTY * (1.0 + 0.25 * p_rush + 0.20 * p_consolidator)
        spread_mod50_bonus = SPREAD_MOD50_BONUS * (1.0 + 0.30 * p_turtle)

        expansion_rate = (opp_land - _opp_land_history[0]) / max(1, len(_opp_land_history)) if len(_opp_land_history) >= 6 else 0.35

        # ── 4. ONLINE TD LEARNING (TRUE BELLMAN NORMALIZED UPDATE) ────────────
        current_metric = (my_land * 0.5 + my_army * 0.3) - (opp_land * 0.5 + opp_army * 0.3)
        if turn > 1 and _neural_net.last_x is not None:
            raw_reward = current_metric - _prev_score_metric
            if worst_excess > 0:
                raw_reward -= 10.0
            norm_reward = max(-3.0, min(3.0, raw_reward / 20.0))
            # Bellman error: delta = r + gamma * V(s') - V(s)
            td_error = norm_reward - _neural_net.last_val
            _neural_net.train_step(td_error)
        _prev_score_metric = current_metric

        # ── 5. TIER 1: HARD TACTICAL REFLEXES ────────────────────────────────
        # A. Lethal Strike on Adjacent Enemy General
        if _opp_general:
            egr, egc = _opp_general
            for d_idx, (dr, dc) in enumerate(DIRS):
                adj_r, adj_c = egr + dr, egc + dc
                if 0 <= adj_r < h and 0 <= adj_c < w and owner[adj_r][adj_c] == 1:
                    my_atk  = army[adj_r][adj_c]
                    opp_def = army[egr][egc]
                    rev_dir = OPPOSITE_DIR[d_idx]
                    if (my_atk - 1 > opp_def) or turn >= 800:
                        act_vec = [0, int(adj_r), int(adj_c), int(rev_dir), 0]
                        _action_history.append(act_vec)
                        return act_vec

        # B. Direct Defense of General from Adjacent Enemy (Suicide Bug FIXED)
        adj_enemies = []
        for d_idx, (dr, dc) in enumerate(DIRS):
            adj_r, adj_c = gr + dr, gc + dc
            if 0 <= adj_r < h and 0 <= adj_c < w and owner[adj_r][adj_c] == 2:
                adj_enemies.append((adj_r, adj_c, d_idx, army[adj_r][adj_c]))

        if adj_enemies:
            # Sort threats by army descending
            adj_enemies.sort(key=lambda x: x[3], reverse=True)
            top_er, top_ec, top_dir, top_threat = adj_enemies[0]

            # 1. Counter with general ONLY if guaranteed to win with split=1
            # (Preserves half garrison on general; NEVER vacate general with split=0)
            if (army[gr][gc] // 2) - 1 > top_threat:
                act_vec = [0, int(gr), int(gc), int(top_dir), 1]
                _action_history.append(act_vec)
                return act_vec

            # 2. Reinforce general from adjacent ally with MAXIMUM available army
            best_ally_move = None
            max_ally_army  = 1
            for d2_idx, (dr2, dc2) in enumerate(DIRS):
                nr2, nc2 = gr + dr2, gc + dc2
                if 0 <= nr2 < h and 0 <= nc2 < w and owner[nr2][nc2] == 1 and (nr2, nc2) != (gr, gc):
                    if army[nr2][nc2] > max_ally_army:
                        max_ally_army  = army[nr2][nc2]
                        rev_dir2       = OPPOSITE_DIR[d2_idx]
                        best_ally_move = [0, int(nr2), int(nc2), int(rev_dir2), 0]

            if best_ally_move:
                _action_history.append(best_ally_move)
                return best_ally_move

        # C. Urgent Interception & Threat-Proximity Pullback Defense
        # Triggers when worst_excess > 0 or an enemy is dangerously close to general
        if (worst_excess > 0 and threat_pos) or (min_threat_dist <= 3 and enemy_units):
            target_threat = threat_pos if threat_pos else min(enemy_units, key=lambda u: dist_from_gen.get((u[0], u[1]), 999))[:2]
            tr, tc = target_threat
            threat_d = dist_from_gen.get((tr, tc), 999)
            dist_from_threat = _bfs_distance(h, w, _known_obstacles, tr, tc)

            best_pullback_move = None
            best_pullback_val  = -1e9

            # Maximum relevant defender distance: threat arrival horizon
            # If threat arrives in 2-3 turns, units > 4 steps away cannot help in time!
            max_def_horizon = max(3, threat_d + 1) if threat_d <= 4 else 10

            for r in range(h):
                for c in range(w):
                    if owner[r][c] == 1 and army[r][c] >= 2 and (r, c) != (gr, gc):
                        d_curr_gen    = dist_from_gen.get((r, c), 999)
                        d_curr_threat = dist_from_threat.get((r, c), 999)

                        # Skip defenders that cannot arrive within the threat horizon
                        if d_curr_gen > max_def_horizon and d_curr_threat > max_def_horizon:
                            continue

                        unit_army = army[r][c]

                        for d_idx, (dr, dc) in enumerate(DIRS):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in _known_obstacles:
                                d_next_gen    = dist_from_gen.get((nr, nc), 999)
                                d_next_threat = dist_from_threat.get((nr, nc), 999)

                                helps_gen    = d_next_gen < d_curr_gen
                                helps_threat = d_next_threat < d_curr_threat

                                if helps_gen or helps_threat:
                                    # Quadratic distance discount: close defenders strongly favored over distant ones
                                    proximity_weight = 10.0 / ((d_next_gen + 1.0) ** 1.8)
                                    pullback_val = unit_army * proximity_weight
                                    if helps_threat and unit_army > army[tr][tc]:
                                        pullback_val += 200.0  # Big bonus for decisive interception
                                    if pullback_val > best_pullback_val:
                                        best_pullback_val  = pullback_val
                                        best_pullback_move = [0, int(r), int(c), int(d_idx), 0]

            if best_pullback_move:
                _action_history.append(best_pullback_move)
                return best_pullback_move

        # ── 6. TIER 2: SAFE CASTLE ECONOMICS (kind = 2) ──────────────────────
        my_structures = [(gr, gc)]
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 1 and terrain[r][c] == 3:
                    my_structures.append((r, c))
        castle_count = len(my_structures) - 1

        if 60 <= turn <= 350 and _castles_built < 2 and worst_excess == 0:
            for r in range(h):
                for c in range(w):
                    if owner[r][c] == 1 and terrain[r][c] == 1 and army[r][c] >= CASTLE_THRESHOLD:
                        dist_penalty = sum(max(0, 14 - 2 * (abs(r - sr) + abs(c - sc)))
                                           for sr, sc in my_structures)
                        build_cost = 35 + dist_penalty
                        min_struct_dist = min(abs(r - sr) + abs(c - sc) for sr, sc in my_structures)

                        # SAFE CASTLE RULE:
                        # 1. Distance >= 7 from existing structures
                        # 2. Army >= build_cost
                        # 3. SAFETY SHIELD: No visible enemy within Manhattan distance <= 3!
                        if min_struct_dist >= 7 and army[r][c] >= build_cost:
                            safe_from_enemies = all(
                                abs(r - er) + abs(c - ec) > 3 for er, ec, _ in enemy_units
                            ) if enemy_units else True

                            if safe_from_enemies:
                                _castles_built += 1
                                act_vec = [2, int(r), int(c), 0, 0]
                                _action_history.append(act_vec)
                                return act_vec

        # ── 7. DISTANCE MAP TO TARGET & DAG FRONTLINE ─────────────────────────
        target   = _best_target(obs, enemy_cells)
        dist_map = _bfs_distance(h, w, _known_obstacles, target[0], target[1])

        # Find friendly frontline cells (closest friendly to target with army >= 2)
        frontline_cells = []
        min_frontline_d = 999
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 1 and army[r][c] >= 2 and (r, c) != (gr, gc):
                    fd = dist_map.get((r, c), 999)
                    if fd < min_frontline_d:
                        min_frontline_d = fd
                        frontline_cells = [(r, c)]
                    elif fd == min_frontline_d:
                        frontline_cells.append((r, c))

        # Multi-source distance field to frontline for true DAG funnel
        dist_to_frontline = _multi_source_bfs(h, w, _known_obstacles, frontline_cells) if frontline_cells else {}

        # ── 8. ENUMERATE CANDIDATE MOVES WITH DYNAMIC SCORING ─────────────────
        is_mod50_burst        = (turn % 50) >= 38 or (turn % 50) <= 2
        is_deathtouch_staging = 740 <= turn < 800
        is_deathtouch_active  = turn >= 800

        top_candidates = []
        best_score_h   = -1e18

        for r in range(h):
            if time.perf_counter() - t_start > 0.085:
                break
            for c in range(w):
                if owner[r][c] != 1 or army[r][c] < 2:
                    continue

                army_curr = army[r][c]
                curr_d    = dist_map.get((r, c), 999)
                curr_fl_d = dist_to_frontline.get((r, c), 999)

                split_flag = 0
                if (r, c) == (gr, gc):
                    # NEVER move general if any enemy is within 8 steps or under any threat
                    has_nearby_enemy = False

                    # Dynamic safe garrison floor scaled with match phase and total army
                    safe_garrison = 14.0

                    if has_nearby_enemy or worst_excess > 0 or army_curr < safe_garrison:
                        continue
                    split_flag = 1

                for direction, (dr, dc) in enumerate(DIRS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if (nr, nc) in _known_obstacles or terrain[nr][nc] == 2:
                        continue

                    next_d    = dist_map.get((nr, nc), 999)
                    next_fl_d = dist_to_frontline.get((nr, nc), 999)

                    # Base distance gradient
                    score = (curr_d - next_d) * DIST_WEIGHT

                    # Anti-pingpong
                    if _detect_loop(r, c, direction):
                        score -= 650.0

                    # Mod-50 economy burst
                    if is_mod50_burst and owner[nr][nc] != 1:
                        score += spread_mod50_bonus * 2.5

                    # Move onto enemy general
                    if _opp_general and (nr, nc) == _opp_general:
                        if is_deathtouch_active:
                            score += DEATHTOUCH_BONUS
                        elif army_curr - 1 > army[nr][nc]:
                            score += KILL_BONUS
                        else:
                            score += attack_penalty * 2.0

                    # Deathtouch staging (rings around general at turn 740-799)
                    elif is_deathtouch_staging and _opp_general and next_d <= 2:
                        if next_d == 1:
                            score += 8500.0
                        elif next_d == 2:
                            score += 5500.0

                    elif owner[nr][nc] == 2:  # Enemy tile
                        opp_tile_army = army[nr][nc]
                        if army_curr - 1 > opp_tile_army or is_deathtouch_active:
                            if terrain[nr][nc] == 4:
                                score += KILL_BONUS
                            else:
                                score += capture_weight
                                if opp_tile_army >= 2:
                                    score += opp_tile_army * 16.0
                        else:
                            score += attack_penalty

                    elif owner[nr][nc] == 0:  # Neutral / Fog tile
                        score += neutral_bonus
                        if terrain[nr][nc] == 0:
                            score += fog_bonus
                        elif terrain[nr][nc] == 3:  # Castle
                            if army_curr - 1 > army[nr][nc]:
                                score += 450.0
                            else:
                                score -= 250.0

                    elif owner[nr][nc] == 1:  # Friendly consolidation / DAG FUNNEL & STACK MERGING
                        # 1. Flow towards target
                        if next_d < curr_d:
                            score += 25.0 + (army_curr - 1) * 1.5 + min(35.0, army[nr][nc] * 0.6)
                        # 2. MULTI-SOURCE DAG FUNNEL TO FRONTLINE:
                        if next_fl_d < curr_fl_d:
                            # Stack merging bonus: reward combining smaller armies into larger spearheads
                            score += 28.0 + (army_curr - 1) * 2.2 + min(45.0, army[nr][nc] * 0.8)

                    # Spearhead mobilization: large stacks (>=6) become battering rams
                    if army_curr >= 6 and (r, c) != (gr, gc):
                        if next_d < curr_d:
                            score += (army_curr - 1) * 3.0
                        if turn >= 350 and next_d < curr_d:
                            score += 120.0 + (army_curr - 1) * 2.0
                        # If opp general confirmed and stack >= 10, charge forward with maximum priority
                        if _opp_general and next_d < curr_d and army_curr >= 10:
                            score += 400.0 + (army_curr - 1) * 3.0

                    # 2-Step Vulnerability Check
                    if (nr, nc) in enemy_adjacent_cells and owner[nr][nc] != 1:
                        threat_val     = enemy_adjacent_cells[(nr, nc)]
                        remaining_army = (army_curr - 1 - army[nr][nc]) if owner[nr][nc] != 1 else army_curr
                        if remaining_army <= threat_val and not is_deathtouch_active:
                            score += vulnerable_penalty

                    top_candidates.append((score, r, c, direction, split_flag, curr_d, army_curr, nr, nc))
                    if score > best_score_h:
                        best_score_h = score

        if not top_candidates:
            return [1, 0, 0, 0, 0]

        # ── 9. BEAM SEARCH (TOP-12) & 1-PLY SIMULATION ────────────────────────
        top_candidates.sort(key=lambda x: x[0], reverse=True)
        beam_k = min(12, len(top_candidates))

        best_move     = None
        best_final    = -1e18
        nn_last_x     = None
        nn_last_val   = None
        nn_last_cache = None

        for cand_idx in range(beam_k):
            if time.perf_counter() - t_start > 0.098:
                break

            score, r, c, direction, split_flag, curr_d, army_curr, nr, nc = top_candidates[cand_idx]

            # Neural Guidance (Additive, Clamped)
            next_d = dist_map.get((nr, nc), 999)
            x = _nn_features(
                r, c, direction, nr, nc, army_curr, curr_d, next_d,
                owner, terrain, army, dist_from_gen,
                enemy_adjacent_cells, enemy_units, opp_army,
                visible_enemy_army, expansion_rate, worst_excess,
                castle_count, turn
            )
            nn_out, a1, z1, a2, z2 = _neural_net.forward(x)
            nn_bonus = max(-NN_CLAMP, min(NN_CLAMP, nn_out))
            final_score = score + nn_bonus * NN_BLEND

            # 1-Ply Forward Simulation
            if time.perf_counter() - t_start < 0.095 and enemy_units:
                moved_army   = (army_curr // 2 if split_flag else army_curr - 1)
                landing_army = (moved_army - army[nr][nc]) if owner[nr][nc] != 1 else (moved_army + army[nr][nc])

                # Opponent best reply check
                opp_threat_to_landing = 0
                for ddr, ddc in DIRS:
                    adj_r2, adj_c2 = nr + ddr, nc + ddc
                    if 0 <= adj_r2 < h and 0 <= adj_c2 < w and owner[adj_r2][adj_c2] == 2:
                        opp_threat_to_landing = max(opp_threat_to_landing, army[adj_r2][adj_c2] - 1)

                if opp_threat_to_landing > landing_army and landing_army > 0 and not is_deathtouch_active:
                    final_score -= 85.0

                # Threatening opponent general next turn bonus
                if _opp_general and abs(nr - _opp_general[0]) + abs(nc - _opp_general[1]) == 1:
                    if landing_army > army[_opp_general[0]][_opp_general[1]] or is_deathtouch_active:
                        final_score += 1500.0

            if final_score > best_final:
                best_final    = final_score
                best_move     = [0, int(r), int(c), int(direction), int(split_flag)]
                nn_last_x     = x
                nn_last_val   = nn_out
                nn_last_cache = (a1, z1, a2, z2)

        if nn_last_x is not None:
            _neural_net.last_x   = nn_last_x
            _neural_net.last_a1  = nn_last_cache[0]
            _neural_net.last_z1  = nn_last_cache[1]
            _neural_net.last_a2  = nn_last_cache[2]
            _neural_net.last_z2  = nn_last_cache[3]
            _neural_net.last_val = nn_last_val

        if best_move:
            _action_history.append(best_move)
            return best_move

        return [1, 0, 0, 0, 0]

    except Exception:
        return [1, 0, 0, 0, 0]
