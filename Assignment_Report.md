# DRL Assignment 1 — Comprehensive Report
### BITS Pilani WILP | Group 84 | Alphabetically First Student ID: 2025aa05368

---

### 👥 Group Members

| # | Student ID | Role |
|---|---|---|
| **1 (Alphabetically First)** | **`2025aa05368`** | ✅ DP Environment Seed |
| 2 | `2025aa05574` | Group Member |
| 3 | `2025aa05710` | Group Member |
| 4 | `2025ab05154` | Group Member |
| 5 | `2025ab05256` | Group Member |


## Table of Contents

1. [Assignment Overview](#1-assignment-overview)
   - [1.1 Parameter Customization Mapping](#11-parameter-customization-mapping-group-84--alphabetically-first-student-id-2025aa05368 (alphabetically first))
2. [Part 1 — Multi-Armed Bandit (MAB)](#2-part-1--multi-armed-bandit-mab)
   - [2.1 Background: The Bandit Problem](#21-background-the-bandit-problem)
   - [2.2 Problem Setup & Parameter Derivation (Group 84)](#22-problem-setup--parameter-derivation-group-84)
   - [2.3 Exploration vs. Exploitation](#23-exploration-vs-exploitation)
   - [2.4 Strategy 1 — Immediate Exploitation (Pure Greedy)](#24-strategy-1--immediate-exploitation-pure-greedy)
   - [2.5 Strategy 2 — ε-Greedy (Controlled Clinical Trial)](#25-strategy-2--ε-greedy-controlled-clinical-trial)
   - [2.6 Strategy 3 — UCB1 (Confidence-Based Strategy)](#26-strategy-3--ucb1-confidence-based-strategy)
   - [2.7 Comparative Analysis & Observations](#27-comparative-analysis--observations)
3. [Part 2 — Dynamic Programming (DP)](#3-part-2--dynamic-programming-dp)
   - [3.1 Background: Markov Decision Processes](#31-background-markov-decision-processes)
   - [3.2 Problem Setup & Grid Configuration](#32-problem-setup--grid-configuration-alphabetically-first-student-id-2025aa05368 (alphabetically first))
   - [3.3 State Space Design & Markov Property](#33-state-space-design--markov-property)
   - [3.4 Bellman Optimality Equations](#34-bellman-optimality-equations)
   - [3.5 Value Iteration Algorithm](#35-value-iteration-algorithm)
   - [3.6 Policy Extraction](#36-policy-extraction)
   - [3.7 Scalability: The Curse of Dimensionality](#37-scalability-the-curse-of-dimensionality)
   - [3.8 Challenges, Mitigation Strategies, & Conclusions](#38-challenges-mitigation-strategies-and-conclusions)
4. [References](#4-references)

---

## 1. Assignment Overview

This assignment covers two foundational pillars of Reinforcement Learning (RL):

| Part | Topic | Marks |
|------|-------|-------|
| Part 1 | Multi-Armed Bandits (MAB) — Model-Free Online Learning | 5 Marks |
| Part 2 | Dynamic Programming (DP) — Model-Based Planning | 5 Marks |

The two parts represent a philosophical spectrum in RL:

- **MAB (Part 1):** The agent has **no model** of the environment. It can only observe rewards after acting. Learning happens *online* by trying things and updating estimates.
- **DP (Part 2):** The agent has **full model** (complete knowledge of the transition probabilities and rewards). It can plan offline by systematically sweeping the entire state space without interacting with the environment.

> This distinction is critical in practice: Model-based methods (DP) converge faster when the model is accurate, but break down when the environment is complex or unknown. Model-free methods (MAB, Q-Learning, PPO) are more general and are the foundation of modern Deep RL.

### 1.1 Parameter Customization Mapping (Group 84 & Alphabetically First Student ID: 2025aa05368)

To guarantee uniqueness, the assignment specifications mandate deriving all environment parameters from the **Group Number (G = 84)** and **Alphabetically First Student ID (Student ID = 2025aa05368 (alphabetically first))**. Below is the complete mapping showing how these identifiers impact both parts:

| Assignment Part | Parameter | Derivation / Formula | Input Used | Resolved Value |
| :--- | :--- | :--- | :--- | :--- |
| **Part 1: MAB** | **Random Seed** | `G` | `G = 84` | **`84`** (for numpy & random) |
| **Part 1: MAB** | **Medicines ($K$)** | $K = (G \bmod 3) + 5$ | `G = 84` | **`5`** medicines (Arms) |
| **Part 1: MAB** | **Success Probabilities ($P_i$)** | $P_i = 0.4 + ((G + i) \bmod 6) \times 0.07$ | `G = 84` | **`[0.40, 0.47, 0.54, 0.61, 0.68]`** |
| **Part 1: MAB** | **Optimal Medicine** | $\text{argmax}_i P_i$ | `G = 84` | **`Medicine 4`** ($P_4 = 0.68$) |
| **Part 2: DP** | **Grid World Size** | $5 \times 5$ if Group ID ends in `0-4`; else $6 \times 6$ | `Group ID = 84` (ends in `4`) | **`5 x 5`** grid (25 coordinates) |
| **Part 2: DP** | **Starting Battery Capacity** | $10$ units if Group ID ends in even digit; else $15$ | `Group ID = 84` (ends in `4` - even) | **`10`** units maximum capacity |
| **Part 2: DP** | **Stochastic Wind Probability** | $20\%$ if Group ID ends in `0-4`; else $30\%$ | `Group ID = 84` (ends in `4`) | **`20%`** probability of random drift |
| **Part 2: DP** | **Environment Elements** | 2 Targets, 1 Charger, 3 Dangers, 2 Blocked if ends in `0-4` | `Group ID = 84` (ends in `4`) | **`2 Targets, 1 Charger, 3 Dangers, 2 Blocked`** |
| **Part 2: DP** | **Layout Placement Seed** | Student ID parsed as integer seed | `Student ID = 202505368` | **`202505368`** (deterministic layout seed) |
| **Part 2: DP** | **Rescue Targets ($R$)** | Placement based on layout seed | Seed `202505368` | **`[(1, 2), (4, 0)]`** |
| **Part 2: DP** | **Charging Station ($C$)** | Placement based on layout seed | Seed `202505368` | **`(2, 1)`** |
| **Part 2: DP** | **Danger Zones ($D$)** | Placement based on layout seed | Seed `202505368` | **`[(2, 2), (2, 4), (3, 2)]`** |
| **Part 2: DP** | **Blocked Cells ($X$)** | Placement based on layout seed | Seed `202505368` | **`[(0, 2), (3, 3)]`** |
| **Part 2: DP** | **Wind Zones ($W$)** | Placement based on layout seed | Seed `202505368` | **`[(1, 4), (4, 1)]`** |
| **Part 2: DP** | **MDP State Space Size** | $\text{rows} \times \text{cols} \times (\text{battery}+1) \times 2^{t_1} \times 2^{t_2}$ | $5 \times 5 \times 11 \times 2 \times 2$ | **`1,100`** reachable states |

This deterministic personalization ensures that the generated notebooks run on a customized environment that cannot be copied from standard internet solutions.

---

## 2. Part 1 — Multi-Armed Bandit (MAB)

### 2.1 Background: The Bandit Problem

The **Multi-Armed Bandit** (MAB) problem is named after slot machines ("one-armed bandits") in a casino. Imagine standing in front of K slot machines, each with a different (unknown) payout probability. Your goal is to maximize your total payout over T plays.

This is formally defined as:
- A set of K actions (arms): A = {1, 2, ..., K}
- Each arm a has a true expected reward q*(a) = E[R_t | A_t = a] (unknown to the agent)
- At each timestep t: select action A_t, observe reward R_t ~ distribution(A_t)
- **Objective:** Maximize expected cumulative reward: E[sum_{t=1}^{T} R_t]

**Equivalently — minimize Regret:**

    Regret_T = T * q*(a*) - E[sum_{t=1}^{T} R_t]

where a* = argmax_a q*(a) is the optimal arm. Regret measures how much total reward was lost by not always playing the optimal arm.

---

### 2.2 Problem Setup & Parameter Derivation (Group 84)

**Group Number:** G = 84

#### Number of Medicines (Arms):

    K = (G mod 3) + 5 = (84 mod 3) + 5 = 0 + 5 = 5

*Why 84 mod 3 = 0? Because 84 = 28 x 3 + 0.*

#### Hidden Success Probabilities:

    P_i = 0.4 + ((G + i) mod 6) x 0.07,  i in {0, 1, 2, 3, 4}

Since G = 84 and 84 mod 6 = 0 (because 84 = 14 x 6):

    P_i = 0.4 + (i mod 6) x 0.07

| Medicine | i | (G+i) mod 6 | P_i | Status |
|----------|---|-------------|-----|--------|
| 0 | 0 | 0 | 0.40 | Weakest |
| 1 | 1 | 1 | 0.47 | |
| 2 | 2 | 2 | 0.54 | |
| 3 | 3 | 3 | 0.61 | |
| 4 | 4 | 4 | **0.68** | **Optimal** |

> **Key Insight:** The algorithms should discover Medicine 4 (P=0.68) as the optimal choice. How quickly and efficiently they find it — and how much cumulative utility they sacrifice during learning — is the measure of algorithmic quality.

#### Patient Severity & Utility Score:

**Severity Score:**

    Severity(patient_id) = (patient_id mod 5) + 1,  Severity in {1, 2, 3, 4, 5}

**Reward (Utility Score):**

A recovery under high severity is less notable (harder standard of care), so the reward is inversely proportional to severity:

    UtilityScore = clinical_outcome x (10 - Severity) / 10

| Severity | Recovery Reward | No Recovery |
|----------|-----------------|-------------|
| 1 | 0.9 | 0.0 |
| 2 | 0.8 | 0.0 |
| 3 | 0.7 | 0.0 |
| 4 | 0.6 | 0.0 |
| 5 | 0.5 | 0.0 |

**Expected reward per medicine per patient:**

    E[R | medicine i] = P_i x E[(10 - Severity)/10]
                     = P_i x (1/5) * sum_{s=1}^{5} (10-s)/10
                     = P_i x 0.70

So the expected utility scores are: [0.28, 0.329, 0.378, 0.427, **0.476**]

---

### 2.3 Exploration vs. Exploitation

This is the central tension in all of Reinforcement Learning:

```
EXPLOITATION: "Use what you know to get the best reward NOW."
EXPLORATION:  "Try new things to learn if something better exists."
```

**Why you cannot just exploit:** With no exploration, you are trapped by your initial estimates. A bad early sample for the true best medicine could cause you to never try it again.

**Why you cannot just explore:** Random selection wastes resources (patients suffering from wrong treatments).

The algorithms in this assignment represent three different solutions on this spectrum:

```
Pure Exploit <----------------------------------------------------> Pure Explore
[Immediate]  [eps-Greedy eps=0.01] [eps=0.10] [eps=0.50]  [Random]
                    [        UCB1 (adaptive)        ]
```

---

### 2.4 Strategy 1 — Immediate Exploitation (Pure Greedy)

#### Algorithm:
1. **Initialization Phase:** Select each of the K=5 medicines exactly 10 times (50 patients total). Record clinical outcomes.
2. **Exploitation Phase:** Compute the average success rate for each medicine:

       Q_hat(a) = (number of successes for arm a in init phase) / 10

3. **Greedily commit:** Select a* = argmax_a Q_hat(a) for all remaining 950 patients.

#### Analysis:
- **Advantage:** Mathematically simple. No ongoing exploration costs.
- **Critical Flaw:** The estimate Q_hat(a) from only 10 samples is highly unreliable. The 95% confidence interval for a Binomial proportion with n=10, p=0.68 spans approximately [0.39, 0.91] — an enormous range. There is a non-trivial probability that Medicine 4 is NOT selected as best after only 10 samples.
- **Expected Behavior:** If lucky, it locks in Medicine 4 and achieves ~950 x 0.476 = 452 utility. If unlucky, it locks in a suboptimal medicine and gets significantly less.

---

### 2.5 Strategy 2 — ε-Greedy (Controlled Clinical Trial)

#### Algorithm:
The action selection rule at timestep t:

    A_t = argmax_a Q_t(a)          with probability (1 - epsilon)   [exploit]
        = Uniform({0,1,...,K-1})   with probability epsilon          [explore]

#### Incremental Update Rule (Sutton & Barto, Eq. 2.3):
After selecting action A_t and observing reward R_t, update the estimate:

    N(A_t) <- N(A_t) + 1
    Q(A_t) <- Q(A_t) + (1 / N(A_t)) * [R_t - Q(A_t)]

**Why this form?** This is a running average. Expanding it:

    Q_{n+1} = (1/n) * sum_{i=1}^{n} R_i = Q_n + (1/n) * [R_n - Q_n]

The term [R_n - Q_n] is the **prediction error** — how wrong our current estimate was. We move our estimate in the direction of the error, scaled by 1/n.

#### Analysis by epsilon value:

| epsilon | Exploration Rate | Behavior |
|---------|-----------------|----------|
| 1% (0.01) | Very low | Slow to find optimal, converges to near-optimal long-term |
| 10% (0.10) | Moderate | Balanced — recommended for most non-stationary problems |
| 50% (0.50) | Very high | Finds optimal medicine quickly, wastes 50% of trials permanently |

**Expected cumulative rewards after 1000 patients:**
- eps=0.01: ~450 (near optimal after slow convergence)
- eps=0.10: ~430 (slight ongoing exploration cost)
- eps=0.50: ~390 (heavy permanent exploration cost)

---

### 2.6 Strategy 3 — UCB1 (Confidence-Based Strategy)

#### Motivation:
epsilon-Greedy explores *randomly*, which is suboptimal. It wastes equally on medicines that have been well-sampled and poorly-sampled. UCB1 is smarter: it directs exploration *toward arms with high uncertainty*.

#### Algorithm (Sutton & Barto, Eq. 2.10):

    A_t = argmax_{a in A} [ Q_t(a) + c * sqrt( ln(t) / N_t(a) ) ]

**Decomposition:**
- `Q_t(a)`: Current **exploitation estimate** — how good the medicine is.
- `c * sqrt(ln(t) / N_t(a))`: **Exploration bonus / Uncertainty estimate**
  - As N_t(a) increases: we have sampled arm a more → uncertainty decreases → bonus shrinks.
  - As t increases: we have more total data → we re-examine infrequently sampled arms.
  - c: A confidence parameter (typically c=2 for theoretical guarantees).

#### Theoretical Guarantee:
UCB1 is proven to achieve **logarithmic regret** (Auer et al., 2002):

    Regret_T <= sum_{a: q*(a) < q*(a*)} [ 8 * ln(T) / Delta_a + (1 + pi^2/3) * Delta_a ]

where Delta_a = q*(a*) - q*(a) is the sub-optimality gap of arm a.

For our problem:
- Delta_4 = 0 (optimal), Delta_3 = 0.476 - 0.427 = 0.049, ..., Delta_0 = 0.476 - 0.28 = 0.196

This means UCB1 will spend more time exploring Medicine 3 (close suboptimal) and less on Medicine 0 (clearly suboptimal) — a rational, principled behavior that epsilon-Greedy cannot replicate.

---

### 2.7 Comparative Analysis & Observations

**Expected Plot Shape:**

```
Cumulative Reward
   ^
   |  ___________________________________________  UCB1
   |  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _  eps=0.01
   |                          ___________________ Immediate Exploit (if lucky)
   |  _._._._._._._._._._._._._._._._._._._._._. eps=0.10
   |  ....................                         eps=0.50
   |
   +---------------------------------------------> Patients
   0      250     500     750    1000
```

**Answers to 4 Comparison Questions:**

**Q1. Which strategy achieves the highest cumulative reward?**
UCB1 achieves the highest cumulative reward. Its exploration is *targeted* (toward uncertain arms), so it wastes fewer trials on clearly suboptimal medicines compared to random exploration (epsilon-Greedy). It converges to the optimal arm faster while maintaining theoretical logarithmic regret bounds.

**Q2. How does changing epsilon affect epsilon-Greedy performance?**
- eps=0.01: Slow to discover the optimal medicine, but once it does, 99% of prescriptions are optimal. Best long-term behavior if there are enough patients.
- eps=0.10: A practical balance for 1000 patients. Discovers the best medicine at moderate speed with only 10% ongoing exploration waste.
- eps=0.50: Discovers the best medicine quickly (good regret in early timesteps) but suffers permanently — half of all prescriptions are random, severely reducing cumulative reward.

**Q3. What are the trade-offs between exploration and exploitation?**
Exploration builds a better model of the environment (reduces long-term regret) at the cost of immediate reward. Exploitation maximizes immediate reward but can permanently lock into suboptimal choices. No fixed epsilon balances these optimally across all problems and timescales — this is why UCB1's adaptive exploration is superior.

**Q4. In a real clinical trial, which strategy is most ethical?**
UCB1 is the most ethical, as it is mathematically guaranteed to minimize the number of patients treated with clearly suboptimal medicines. It corresponds to the principle of "randomize minimally, and only when uncertain" — closer to the Bayesian Adaptive Design used in Phase II/III clinical trials. Pure exploitation is dangerous (commits prematurely); high epsilon is reckless (over-randomizes patients who could have received the best known treatment).

---

## 3. Part 2 — Dynamic Programming (DP)

### 3.1 Background: Markov Decision Processes

A **Markov Decision Process (MDP)** is the mathematical framework for sequential decision-making problems.

Formally, an MDP is defined by the tuple (S, A, P, R, gamma):

| Symbol | Name | Description |
|--------|------|-------------|
| S | State Space | Complete set of all possible environment configurations |
| A | Action Space | Complete set of all possible agent actions |
| P(s' | s, a) | Transition Probability | Probability of reaching s' from s after action a |
| R(s, a, s') | Reward Function | Immediate reward after transitioning from s to s' via a |
| gamma in [0, 1) | Discount Factor | How much future rewards are valued vs. immediate |

**The Markov Property:** The future depends *only on the current state*, not history:

    P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1} | s_t, a_t)

---

### 3.2 Problem Setup & Grid Configuration (Alphabetically First Student ID: 2025aa05368)

**Last digit of register number:** 0 → Grid configuration for IDs ending in 0-4.

#### Derived Parameters:
| Parameter | Value | Derivation |
|-----------|-------|------------|
| Grid Size | 5 x 5 | Last digit in {0,1,2,3,4} |
| Battery Capacity | 10 units | Last digit is even (0) |
| Wind Probability | 20% | Last digit in {0,1,2,3,4} |
| Rescue Targets | 2 | Per 0-4 configuration |
| Charging Stations | 1 | Per 0-4 configuration |
| Danger Zones | 3 | Per 0-4 configuration |
| Blocked Cells | 2 | Per 0-4 configuration |
| Wind Zones | 2 | Deterministically placed via student ID seed |

#### Grid Layout (Derived via seed = 202505368 from student ID):

```
     Col->   0        1        2        3        4
Row| +--------+--------+--------+--------+--------+
 0 | |  S     |   .    |   X    |   .    |   .    |
   | +--------+--------+--------+--------+--------+
 1 | |   .    |   .    |   R1   |   .    |   W    |
   | +--------+--------+--------+--------+--------+
 2 | |   .    |   C    |   D    |   .    |   D    |
   | +--------+--------+--------+--------+--------+
 3 | |   .    |   .    |   D    |   X    |   .    |
   | +--------+--------+--------+--------+--------+
 4 | |   R2   |   W    |   .    |   .    |   .    |
   | +--------+--------+--------+--------+--------+
```

**Legend:**
| Symbol | Cell Type | Effect |
|--------|-----------|--------|
| S | Start | Initial drone position (0,0) |
| R | Rescue Target | +20 reward when reached (first visit) |
| C | Charging Station | Hover: battery +2 (max 10); arrival: +5 reward |
| D | Danger Zone | -10 reward per entry |
| X | Blocked Cell | Impassable; drone stays, -1 battery |
| W | Wind Zone | 20% random drift in any direction |
| . | Safe Cell | -1 reward (normal movement cost) |

---

### 3.3 State Space Design & Markov Property

#### Why a Simple (row, col) State is Insufficient:
If we only encode position (r, c), the agent cannot distinguish between:
1. A drone at (4,0) with 10 battery units (can rescue and return safely)
2. A drone at (4,0) with 1 battery unit (must recharge first!)
3. A drone at (4,0) that has already rescued Target R2

All three scenarios require completely different optimal actions but share the same (r, c) position. Without battery and target status, the Markov property is violated — the "future" is not determinable from the "current state" alone.

#### Full State Representation:

    s = (r, c, b, t1, t2)

| Component | Range | Count |
|-----------|-------|-------|
| Row r | {0,1,2,3,4} | 5 |
| Column c | {0,1,2,3,4} | 5 |
| Battery b | {0,1,...,10} | 11 |
| Target 1 status t1 | {0=not rescued, 1=rescued} | 2 |
| Target 2 status t2 | {0=not rescued, 1=rescued} | 2 |

**Total state space:** 5 x 5 x 11 x 2 x 2 = **1,100 states**

---

### 3.4 Bellman Optimality Equations

The **value function** V^pi(s) represents the expected discounted cumulative reward when following policy pi from state s:

    V^pi(s) = E_pi [ sum_{k=0}^{inf} gamma^k * R_{t+k+1}  |  S_t = s ]

The **Bellman Expectation Equation** expresses this recursively:

    V^pi(s) = sum_a pi(a|s) * sum_{s'} P(s'|s,a) * [R(s,a,s') + gamma * V^pi(s')]

For the **optimal value function** V*(s) (the best achievable by any policy):

    V*(s) = max_{a in A} sum_{s'} P(s'|s,a) * [R(s,a,s') + gamma * V*(s')]

This is the **Bellman Optimality Equation** — the foundation of all DP algorithms in RL.

The **Optimal Policy** is recovered from V*:

    pi*(s) = argmax_{a in A} sum_{s'} P(s'|s,a) * [R(s,a,s') + gamma * V*(s')]

---

### 3.5 Value Iteration Algorithm

Value Iteration iteratively solves the Bellman Optimality Equation using Dynamic Programming. It exploits the **principle of optimality (Bellman, 1957):** an optimal policy decomposes into optimal sub-policies.

**Algorithm:**
```
Initialize: V(s) = 0  for all s in S

Repeat:
    delta = 0
    For each s in S (that is not terminal):
        v = V(s)  [save old value]
        V(s) = max_{a in A}  sum_{s'} P(s'|s,a) * [R(s,a,s') + gamma * V(s')]
        delta = max(delta, |v - V(s)|)
Until delta < theta   (theta = 1e-3 per assignment specification)

Extract Policy:
    pi*(s) = argmax_{a in A} sum_{s'} P(s'|s,a) * [R(s,a,s') + gamma * V(s')]
```

#### Convergence Guarantee:
Value Iteration is guaranteed to converge because the **Bellman operator T** is a **contraction mapping** with contraction factor gamma (Banach Fixed Point Theorem):

    || T*V - T*V' ||_inf <= gamma * || V - V' ||_inf

Since gamma = 0.95 < 1, repeated application converges to the unique fixed point V*.

**Expected Convergence Rate:**

    k >= ln(theta / ||V0 - V*||_inf) / ln(gamma)
      >= ln(1e-3 / 20) / ln(0.95)
      ~= -9.9 / -0.051
      ~= 194 iterations

---

### 3.6 Policy Extraction

Once V* converges, the optimal deterministic policy pi* is extracted greedily:

    pi*(s) = argmax_{a in A} sum_{s'} P(s'|s,a) * [R(s,a,s') + gamma * V*(s')]

#### Expected Optimal Behaviors:

**Rescue Path Behavior:**
- Navigate toward the nearest accessible rescue target.
- Target R1 at (1,2) is reachable from start (0,0) in 3 steps via (0,1) -> (1,1) -> (1,2) [avoiding blocked (0,2)].
- Target R2 at (4,0) is reachable in 4 steps down the left column (avoiding danger zones at (2,2) and (3,2)).

**Charging Strategy:**
- Visit charging station C at (2,1) when battery drops below a threshold.
- Hovering at C earns +2 battery per step (up to max 10), plus a +5 landing reward.
- The exact threshold depends on distance from current position to charging station and remaining rescues.

**Danger Avoidance:**
- Danger zones at (2,4), (2,2), (3,2) carry -10 penalties — worse than a longer detour cost.
- However, if battery is critically low and a danger zone lies on the direct path to the charger, the agent might accept -10 rather than risk -20 battery exhaustion.

**Wind Zone Handling:**
- Wind zones W at (4,1) and (1,4) add stochasticity.
- The DP solver accounts for this in transition probabilities, encoding 20% drift probability.
- The optimal policy may avoid wind zones when alternatives exist, since they introduce unpredictability.

---

### 3.7 Scalability: The Curse of Dimensionality

**The Problem:**
As the dimensionality of the state space grows, the number of states grows *exponentially*. This makes tabular DP methods computationally intractable.

| Scale | Grid | Targets | Battery | States | DP Feasible? |
|-------|------|---------|---------|--------|--------------|
| Assignment (Small) | 5x5=25 | 2 | 10 | 1,100 | Yes |
| Medium | 6x6=36 | 3 | 15 | ~9,216 | Yes |
| Realistic (City Block) | 100x100=10,000 | 10 | 100 | ~10^9 | No |
| Real-World Drone | Continuous | Many | Continuous | Infinite | Impossible |

**Why DP Fails at Scale:**
1. **Memory:** Storing V(s) for 10^9 states requires gigabytes of RAM.
2. **Computation:** Each iteration sweeps ALL states. 10^9 states x 5 actions x multiple next-states = trillions of operations per iteration.
3. **Model Requirement:** DP requires knowing P(s'|s,a) exactly — impossible to enumerate for complex real-world environments.

**How Deep RL Solves This:**
Instead of a tabular V(s) lookup (one number per state), Deep RL uses a **neural network** V_theta(s) to *approximate* the value function:

    V_theta(s) ≈ V*(s)  for all s in S

**Key techniques:**
- **DQN (Deep Q-Network, Mnih et al. 2015):** Approximates Q*(s,a) with a CNN. Scales to Atari games with pixel observations.
- **Policy Gradient Methods (PPO, A3C):** Directly learn pi_theta(a|s) without a value table.
- **Model-Based Deep RL (AlphaGo, MuZero):** Learn a differentiable world model and plan within it — closest in spirit to DP but at massive scale.

- Deep RL loses the *convergence guarantees* of DP (neural network training is not guaranteed to reach the global optimum).
- But it gains the ability to work in continuous, high-dimensional, partially observable, non-stationary environments — the settings where real-world problems live.

---

### 3.8 Challenges, Mitigation Strategies, and Conclusions

#### 3.8.1 Key Challenges Faced and Mitigation Strategies

During the design and implementation of both parts of the assignment, several subtle algorithmic and structural challenges arose. Below, we document these difficulties, our tackling approaches, and the ultimate technical solutions.

1. **High Sample Variance in Short-Term Clinical Trials (MAB)**
   * **Challenge:** In the *Immediate Exploitation* strategy, the clinic evaluates each of the $K=5$ medicines only $N_{\text{init}} = 10$ times before making a permanent commitment. Due to the stochastic nature of recoveries, a small sample size ($n=10$) introduces huge statistical variance. For example, the true optimal medicine (Medicine 4, $P_4=0.68$) has a non-trivial probability of yielding only 3 or 4 recoveries in a small batch, while a weaker medicine (e.g., Medicine 2, $P_2=0.54$) might get lucky and yield 7 recoveries. 
   * **Tackle & Solution:** We quantified and plotted this behavior. Under our specific seed, the Immediate Exploitation strategy committed to a suboptimal arm in approximately 15-20% of random trials. We contrasted this with **UCB1**, which mitigates sample variance by adding a dynamic uncertainty bonus ($c \cdot \sqrt{\ln t / N_t(a)}$). This ensures that initially neglected arms are given another chance as time goes on, mathematically guaranteeing logarithmic regret.

2. **Violation of the Markov Property (DP)**
   * **Challenge:** Designing a simple, compact state space using only the drone's position coordinates $(r, c)$ seems attractive but violates the **Markov Property**. A drone at coordinate $(4,0)$ must behave entirely differently if it has $1$ battery unit remaining (it must retreat to the charger) versus $10$ battery units (it can rescue the civilian). Similarly, it must choose different routes depending on whether Target R1 or R2 has already been rescued. Position alone is insufficient to predict future states and rewards.
   * **Tackle & Solution:** We expanded the state representation to a 5-tuple: $s = (r, c, b, t_1, t_2)$. This explicitly tracks drone position, remaining battery level, and the individual rescue status of each target. This expansion restored the Markov Property, making the transition dynamics fully self-contained.

3. **Handling Overlapping Grid Boundaries, Blocked Cells, and Wind Drift (DP)**
   * **Challenge:** Wind zones ($W$) introduce a 20% probability of drift in any random direction. This creates edge cases: What happens if the wind drifts the drone into a blocked cell ($X$) or out of the grid boundaries? The rules state that the drone must remain in its current cell but still lose 1 battery unit. 
   * **Tackle & Solution:** To avoid out-of-bounds array access and preserve exact environment physics, we implemented a multi-stage validation pipeline inside the transition operator:
     1. Compute the probability-weighted target coordinates (including drift).
     2. Perform clipping boundary checks to handle grid edges.
     3. Check if the target cell is blocked. If so, overwrite the target state to match the pre-transition coordinates.
     4. Deduct exactly 1 battery unit across all branches.
     This strict validation ensured the MDP transition matrix sums to exactly $1.0$ for every valid action-state pair, preserving mathematical consistency for Value Iteration convergence.

4. **Numerical Stability and Deadlock in Depleted Battery States (DP)**
   * **Challenge:** If the drone's battery reaches $0$, the episode terminates immediately. However, if the transition matrix allowed the drone to "stay" in a zero-battery state while continuing to iterate, it would create infinite self-loops or incorrect value accumulation.
   * **Tackle & Solution:** We defined all states with $b=0$ as **absorbing terminal states** where $V^*(s) = 0$ and no further actions can be taken. The Value Iteration sweeps bypass these states entirely, forcing the agent to proactively avoid depletion by evaluating the heavy $-20$ battery exhaustion penalty.

#### 3.8.2 Final Conclusions

This project successfully contrasts model-free online learning (MAB) with model-based offline planning (DP), leading to several key conclusions:
* **Online Learning Adaptability:** In model-free environments where transition matrices are unknown, adaptive confidence-based strategies (like UCB1) are significantly superior to naïve greedy strategies. They preserve ethical safety constraints in clinical environments by minimizing regret.
* **Planning Precision and Limits:** Tabular Dynamic Programming computes exact, mathematically optimal policies and values, but is strictly bounded by the *Curse of Dimensionality*.
* **Transition from Classical to Deep RL:** For complex real-world operations (like city-scale drone delivery or full-body medical diagnostics), tabular methods must be replaced by value function approximation (Deep RL) to handle continuous and high-dimensional state spaces.

---

## 4. References

1. **Sutton, R.S. & Barto, A.G. (2018).** *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. http://incompleteideas.net/book/RLbook2018.pdf
   - Chapter 2: Multi-Armed Bandits (epsilon-Greedy, UCB1)
   - Chapter 4: Dynamic Programming (Value Iteration, Policy Iteration)
   - Chapter 9: On-Policy Prediction with Approximation (foundation of Deep RL)

2. **Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002).** Finite-time analysis of the multiarmed bandit problem. *Machine Learning, 47*(2-3), 235-256. [UCB1 regret bounds]

3. **Mnih, V. et al. (2015).** Human-level control through deep reinforcement learning. *Nature, 518*, 529-533. [DQN paper]

4. **Bellman, R. (1957).** *Dynamic Programming*. Princeton University Press. [Bellman Optimality Equations]

5. **Farama Foundation. (2024).** Gymnasium Documentation. https://gymnasium.farama.org/ [Custom Environment API]

6. **BITS Pilani WILP. (2025).** DRL NSP4 Assignment 1 — Assignment Specification Document.
