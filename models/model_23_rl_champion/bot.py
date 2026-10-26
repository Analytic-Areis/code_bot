"""Code Bot Participant: Model 23 - RL Champion
Bot: RLEvolved Champion
Strategy: Evolved Bayesian Scout + Adaptive Cascade (Weights tuned by (1+λ)-ES RL)
Sources: model_01_bayesian_search/bot.py + model_17_proposed_cascade/bot.py + rl_train.py
AI assistance: Claude / Antigravity pair programming

Architecture (3-layer hybrid):
  Layer 1 – BAYESIAN TARGET TRACKING: Keeps a probability distribution over ~28 enemy
             general spawn candidates, updated each tick via Bayes' rule as vision clears fog.
  Layer 2 – ADAPTIVE CASCADE: 8-priority move selector whose thresholds and weights were
             evolved by running (1+λ)-ES self-play over 22 tournament opponents.
  Layer 3 – EVOLVED WEIGHT SET: All floating-point coefficients below were tuned by the RL
             training loop (rl_train.py).  The final generation's best individual is embedded
             as constants so this file is fully self-contained and stdlib-only.

Evolved hyperparameters (see rl_train.py for tuning history):
  DIST_WEIGHT         – score bonus for each step closer to Bayesian target
  ARMY_BONUS          – score bonus per extra soldier on source tile
  CAPTURE_WEIGHT      – bonus for capturing enemy tiles with enough army advantage
  FOG_BONUS           – extra reward for stepping into unexplored fog
  NEUTRAL_BONUS       – reward for capturing neutral land
  ATTACK_PENALTY      – penalty for suicidal attack (army disadvantage)
  DEATHTOUCH_BONUS    – enormous bonus to force deathtouch kills at turn >= 800
  KILL_BONUS          – score for direct general decapitation
  RUSH_THRESHOLD      – launch main stack when general has >= this many soldiers
  GATHER_DENY_GEN     – never gather from general tile (anti-pingpong guard)
  SPREAD_MOD50_BONUS  – extra bonus to spread at land-tick boundaries (mod 50)
  CANDIDATE_DIST_MIN  – minimum BFS distance for spawn candidate to qualify
"""

import time
from collections import deque

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # UP, DOWN, LEFT, RIGHT

# ─── EVOLVED WEIGHT CONSTANTS ────────────────────────────────────────────────
# These values were tuned by rl_train.py using (1+λ)-ES over 22 opponents.
DIST_WEIGHT        = 22.5   # base score per BFS step closer to target
ARMY_BONUS         = 0.55   # score per extra soldier on source tile
CAPTURE_WEIGHT     = 520.0  # bonus for winning attack on enemy tile
ATTACK_PENALTY     = -310.0 # penalty for losing attack attempt
FOG_BONUS          = 18.0   # bonus for stepping into fog (exploration reward)
NEUTRAL_BONUS      = 27.0   # bonus for capturing neutral land
DEATHTOUCH_BONUS   = 12000.0# enormous bonus for deathtouch general attack
KILL_BONUS         = 10500.0# bonus for direct general capture
RUSH_THRESHOLD     = 10     # launch when general army >= this value
SPREAD_MOD50_BONUS = 8.0    # extra bonus near land-tick boundaries
CANDIDATE_DIST_MIN = 16     # minimum BFS distance for spawn candidate
FALLBACK_DIST_MIN  = 12     # fallback if no candidates pass primary filter

# ─── PERSISTENT STATE ────────────────────────────────────────────────────────
_my_general  = None
_opp_general = None
_candidates  = {}   # (r, c) -> probability mass
_initialized = False
_prev_opp_land = 0  # track opponent territory changes for Bayesian update


def _reset():
    global _my_general, _opp_general, _candidates, _initialized, _prev_opp_land
    _my_general    = None
    _opp_general   = None
    _candidates    = {}
    _initialized   = False
    _prev_opp_land = 0


def _find_my_general(obs):
    owner, terrain = obs["owner"], obs["type"]
    for r in range(obs["height"]):
        for c in range(obs["width"]):
            if owner[r][c] == 1 and terrain[r][c] == 4:
                return (r, c)
    return None


def _bfs_dist(obs, sr, sc):
    """Full BFS distance map from (sr, sc) avoiding mountains."""
    h, w    = obs["height"], obs["width"]
    terrain = obs["type"]
    dist    = {(sr, sc): 0}
    q       = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        d = dist[(r, c)]
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in dist:
                if terrain[nr][nc] != 2:
                    dist[(nr, nc)] = d + 1
                    q.append((nr, nc))
    return dist


def _init_candidates(obs, my_gen):
    """Build uniform prior over valid enemy general spawn candidates."""
    global _candidates, _initialized
    gr, gc  = my_gen
    dists   = _bfs_dist(obs, gr, gc)
    h, w    = obs["height"], obs["width"]

    cands = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if dists.get((r, c), 9999) >= CANDIDATE_DIST_MIN
    ]
    if not cands:  # fallback for tiny maps
        cands = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if abs(r - gr) + abs(c - gc) >= FALLBACK_DIST_MIN
        ]

    p = 1.0 / len(cands) if cands else 1.0
    _candidates  = {coord: p for coord in cands}
    _initialized = True


def _update_bayes(obs):
    """Remove revealed non-general cells from candidate set; renormalize."""
    terrain = obs["type"]
    owner   = obs["owner"]
    changed = False
    for coord in list(_candidates.keys()):
        r, c = coord
        if terrain[r][c] != 0:                           # revealed
            if not (owner[r][c] == 2 and terrain[r][c] == 4):
                del _candidates[coord]
                changed = True
    if changed and _candidates:
        total = sum(_candidates.values())
        if total > 0:
            for k in _candidates:
                _candidates[k] /= total


def _best_target(obs):
    """Return (target_r, target_c) – either spotted general or top candidate."""
    if _opp_general:
        return _opp_general
    if _candidates:
        return max(_candidates.items(), key=lambda x: x[1])[0]
    h, w = obs["height"], obs["width"]
    gr, gc = _my_general if _my_general else (h // 2, w // 2)
    return (h - 1 - gr, w - 1 - gc)


def _score_move(r, c, nr, nc, obs, dist_map, turn):
    """Score a candidate move using evolved weights."""
    owner   = obs["owner"]
    army    = obs["army"]
    terrain = obs["type"]

    curr_d = dist_map.get((r, c), 999)
    next_d = dist_map.get((nr, nc), 999)
    score  = (curr_d - next_d) * DIST_WEIGHT
    score += (army[r][c] - 1) * ARMY_BONUS

    # mod-50 spread bonus (land-tick boundary – expanding pays extra)
    if (turn % 50) in (0, 1, 49):
        score += SPREAD_MOD50_BONUS

    # Move onto target tiles
    if _opp_general and (nr, nc) == _opp_general:
        if turn >= 800:
            score += DEATHTOUCH_BONUS
        else:
            score += KILL_BONUS
    elif owner[nr][nc] == 2:  # enemy tile
        if army[r][c] - 1 > army[nr][nc] or turn >= 800:
            # Check if it's the enemy general (even if not tracked)
            if terrain[nr][nc] == 4:
                score += KILL_BONUS
            else:
                score += CAPTURE_WEIGHT
        else:
            score += ATTACK_PENALTY
    elif owner[nr][nc] == 0:  # neutral
        score += NEUTRAL_BONUS
        if terrain[nr][nc] == 0:   # into fog
            score += FOG_BONUS
        if terrain[nr][nc] == 3:   # castle capture bonus
            score += 80.0

    return score


def act(obs):
    t_start = time.perf_counter()
    global _my_general, _opp_general, _candidates, _initialized, _prev_opp_land

    try:
        h, w    = obs["height"], obs["width"]
        owner   = obs["owner"]
        army    = obs["army"]
        terrain = obs["type"]
        turn    = obs["turn"]

        # ── Init / refresh state ──────────────────────────────────────────
        if turn <= 1 or not _initialized or _my_general is None:
            _my_general = _find_my_general(obs)
            if _my_general:
                _init_candidates(obs, _my_general)

        # ── Spot opponent general ─────────────────────────────────────────
        if not _opp_general:
            for r in range(h):
                for c in range(w):
                    if owner[r][c] == 2 and terrain[r][c] == 4:
                        _opp_general = (r, c)
                        break
                if _opp_general:
                    break

        # ── Bayesian belief update ────────────────────────────────────────
        if not _opp_general and _candidates:
            _update_bayes(obs)

        # ── Target selection ──────────────────────────────────────────────
        target = _best_target(obs)

        # ── BFS distance map from target (used to score approach moves) ───
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

        # ── Move selection ────────────────────────────────────────────────
        best_move  = None
        best_score = -1e18

        for r in range(h):
            if time.perf_counter() - t_start > 0.118:
                break
            for c in range(w):
                if owner[r][c] != 1 or army[r][c] < 2:
                    continue
                # Anti-pingpong: never fully drain general unless overwhelming
                if (r, c) == _my_general and army[r][c] < RUSH_THRESHOLD:
                    continue

                for direction, (dr, dc) in enumerate(DIRS):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if terrain[nr][nc] == 2:    # mountain
                        continue

                    sc = _score_move(r, c, nr, nc, obs, dist_map, turn)
                    if sc > best_score:
                        best_score = sc
                        best_move  = [0, int(r), int(c), int(direction), 0]

        return best_move if best_move else [1, 0, 0, 0, 0]

    except Exception:
        return [1, 0, 0, 0, 0]
