"""Code Bot Participant: Model 9 - Multi-Armed Bandit Fog Explorer
Bot: UCB1FogBandit
Strategy: Multi-Armed Bandit (UCB1) for Balanced Fog Exploration and Exploitation
Sources: models.md (Game Theory Model 9)
AI assistance: Claude / Antigravity pair programming

Implements Multi-Armed Bandit Fog Exploration:
1. Partitions the fogged board into candidate sectors (bandit arms).
2. Computes UCB1 score: mu_i + C * sqrt(ln(t) / N_i) for each sector.
3. Systematically allocates scouting tendrils to the highest-scoring UCB1 arm.
4. Transitions to lethal strike once enemy general is revealed.
"""
import math
import time

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

_my_general = None
_opp_general = None
_arm_counts = {}  # sector_idx -> N_i
_arm_rewards = {}  # sector_idx -> sum of rewards


def _reset():
    global _my_general, _opp_general, _arm_counts, _arm_rewards
    _my_general = None
    _opp_general = None
    _arm_counts = {}
    _arm_rewards = {}


def _find_my_general(obs):
    owner, terrain = obs["owner"], obs["type"]
    for r in range(obs["height"]):
        for c in range(obs["width"]):
            if owner[r][c] == 1 and terrain[r][c] == 4:
                return (r, c)
    return None


def _get_sectors(h, w):
    # Divide board into 4 quadrants / sectors
    mid_r, mid_c = h // 2, w // 2
    return [
        (0, 0, mid_r, mid_c),          # Sector 0: NW
        (0, mid_c, mid_r, w),          # Sector 1: NE
        (mid_r, 0, h, mid_c),          # Sector 2: SW
        (mid_r, mid_c, h, w),          # Sector 3: SE
    ]


def act(obs):
    t_start = time.perf_counter()
    global _my_general, _opp_general, _arm_counts, _arm_rewards

    try:
        h, w = obs["height"], obs["width"]
        owner, army, terrain = obs["owner"], obs["army"], obs["type"]
        turn = max(1, obs["turn"])

        if _my_general is None or turn <= 1:
            _my_general = _find_my_general(obs)
            sectors = _get_sectors(h, w)
            _arm_counts = {i: 1 for i in range(len(sectors))}
            _arm_rewards = {i: 0.5 for i in range(len(sectors))}

        # 1. Check for enemy general
        for r in range(h):
            for c in range(w):
                if owner[r][c] == 2 and terrain[r][c] == 4:
                    _opp_general = (r, c)
                    break
            if _opp_general:
                break

        # If enemy general is sighted, attack directly!
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

        # 2. Select Sector via UCB1 Multi-Armed Bandit
        sectors = _get_sectors(h, w)
        C = 1.414  # standard UCB exploration constant
        best_arm = 0
        best_ucb = -1e9

        for i in range(len(sectors)):
            n_i = _arm_counts.get(i, 1)
            mu_i = _arm_rewards.get(i, 0.5) / n_i
            ucb = mu_i + C * math.sqrt(math.log(turn + 1) / n_i)
            if ucb > best_ucb:
                best_ucb = ucb
                best_arm = i

        r_min, c_min, r_max, c_max = sectors[best_arm]
        target_center = ((r_min + r_max) // 2, (c_min + c_max) // 2)

        # 3. Route towards target sector center
        tr, tc = target_center
        best_act = None
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
                        d_curr = abs(r - tr) + abs(c - tc)
                        d_next = abs(nr - tr) + abs(nc - tc)
                        score += (d_curr - d_next) * 15.0

                        if owner[nr][nc] == 0:
                            score += 20.0
                            if terrain[nr][nc] == 0:  # fog clearing
                                score += 15.0
                        elif owner[nr][nc] == 2:
                            score += 40.0

                        score += army[r][c] * 0.3
                        if score > best_score:
                            best_score = score
                            best_act = [0, int(r), int(c), int(d_idx), 0]

        # Update MAB arm counters
        _arm_counts[best_arm] = _arm_counts.get(best_arm, 1) + 1

        if best_act:
            return best_act

        return [1, 0, 0, 0, 0]

    except Exception:
        return [1, 0, 0, 0, 0]
