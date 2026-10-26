# Model 10: Nash Bargaining Solution for Resource Allocation

## 1. Overview & Paradigm
- **Archetype**: Strategy III - Balanced
- **Mathematical Framework**: Axiomatic Cooperative Game Theory / Nash Bargaining
- **Bot Name**: `NashResourceAllocator`
- **File**: `bot.py`
- **Test Suite**: `tester.py`

## 2. Theoretical Principles
Balancing military resources between defending the home base and conducting offensive operations is formulated as a Nash Bargaining Solution between two internal agents: Defense and Offense.
$$\max_{x, y} (U_{\text{def}}(x) - d_{\text{def}}) \times (U_{\text{off}}(y) - d_{\text{off}}) \quad \text{s.t.} \quad x + y \le A_{\text{total}}$$
where:
- $d_{\text{def}}$ is the catastrophic failure threshold (the minimum garrison required to withstand the largest visible or inferred enemy strike stack).
- $d_{\text{off}}$ is the minimum force required to retain expansion tempo on the frontier.
- The Nash product is maximized, ensuring the home base is never dangerously drained to fuel an overextended attack, completely resolving the common "ping-pong / base abandonment" vulnerability.

## 3. Algorithmic Implementation
- **Dynamic Threat Calculation**: Evaluates the closest and largest enemy army relative to $(g_r, g_c)$, scaling $d_{\text{def}}$ proportionally.
- **Priority Gating**: If the garrison on the general falls below $d_{\text{def}}$, all external units are recalled to fortify the general.
- **Offensive Projection**: All surplus military assets above $d_{\text{def}}$ are projected outward into frontier expansion and pressure.

## 4. How to Test
Run the autonomous tester:
```bash
python3 tester.py
```
Or validate using the organizer evaluator:
```bash
python3 ../evaluator/evaluate.py validate bot.py
```
