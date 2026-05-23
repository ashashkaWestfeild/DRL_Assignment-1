# Deep Reinforcement Learning (DRL) — Conceptual Study Guide
## Multi-Armed Bandits (MAB) & Dynamic Programming (DP)
### Prepared for BITS Pilani WILP | DRL Assignment 1 | Group 84 | Student ID: 2025aa05710

---

## 📖 Introduction to the Study Guide

This document is designed as a highly intuitive, comprehensive study guide to help you master the core reinforcement learning theories implemented in this assignment. Use this guide to prepare for presentations, project defense (viva), or to add deep conceptual explanations when sharing this project in your portfolio/Git repository.

It is structured into two parts:
1. **Multi-Armed Bandits (MAB):** Model-free online learning under uncertainty.
2. **Dynamic Programming (DP):** Model-based offline planning via Value Iteration.

For each part, we explain the core theories, provide detailed instructions on **how to read and interpret the generated plots**, outline **what critical areas you should focus on**, and list **potential viva questions**.

---

## 🎮 Part 1: Multi-Armed Bandits (MAB) — Online Learning

The Multi-Armed Bandit problem represents **model-free online learning**. In this paradigm:
*   The agent has **no model of the environment** (it does not know the success probabilities of the treatments).
*   It must learn *interactively* by taking actions (prescribing medicines) and observing outcomes (patient recovery).
*   It faces the fundamental **Exploration-Exploitation Dilemma**: balancing the collection of new information (exploration) with the maximization of immediate rewards using current knowledge (exploitation).

### 📊 Plot 1: `mab_comparison_main.png` (Core Strategy Comparison)

#### What is it?
A dual-axis line plot showing:
1.  **Cumulative Reward** (Left y-axis, upward curves) over $1000$ patients.
2.  **% Optimal Action Selection** (Right y-axis, showing how often the algorithm chooses the true best arm—Medicine 4) over $1000$ patients.

#### How to Read it:
*   **The UCB1 Curve (Green/Top):** Notice how it starts slightly slower because UCB1 is systematically exploring all arms to resolve uncertainty. However, once it builds confidence, its *Optimal Action Selection* rate shoots up toward 95%+, and its *Cumulative Reward* curve rises steeply, outperforming the other strategies.
*   **$\epsilon$-Greedy with $\epsilon=0.10$ (Blue/Steady):** This curve shows a steady, linear increase. Since it explores exactly 10% of the time, its *Optimal Action Selection* rate eventually stabilizes and caps out at **$90\%$** (since the remaining 10% is spent on random exploration).
*   **$\epsilon$-Greedy with $\epsilon=0.50$ (Red/Poor):** This curve rises quickly at the very beginning because high exploration discovers the best medicine immediately. However, because it permanently spends 50% of its trials on random exploration, its *Optimal Action Selection* rate is permanently capped at **$50\%$**, resulting in a much flatter cumulative reward curve.
*   **Immediate Exploitation (Purple/Volatile):** This curve is highly dependent on initialization. With only 10 trials per arm, the sample variance is enormous. If it gets a lucky draw, it tracks near-optimal; if it gets an unlucky draw, it permanently locks onto a suboptimal arm, resulting in a low-slope, flat cumulative reward.

#### What to Focus on:
*   **Regret Minimization:** Explain that cumulative reward maximization is mathematically identical to minimizing **cumulative regret** (the reward lost by not choosing the optimal arm). 
*   **Logarithmic Regret:** Focus on UCB1's theoretical guarantee: its regret grows **logarithmically** ($O(\ln T)$), which means the rate of making mistakes drops to near-zero as time goes on. Contrast this with $\epsilon$-greedy, which suffers **linear regret** ($O(T)$) due to permanent, non-decaying random exploration.

---

### 📊 Plot 2: `eps_greedy_convergence.png` (Q-Value Tracking)

#### What is it?
A multi-panel line plot showing the estimated success probability ($Q$-value) of each of the 5 medicines over 1000 patients for the three $\epsilon$ values ($0.01, 0.10, 0.50$).

#### How to Read it:
*   The **horizontal dashed lines** represent the true hidden success probabilities of your medicines ($P_0=0.40, P_1=0.47, P_2=0.54, P_3=0.61, P_4=0.68$).
*   The **solid lines** represent the agent's running estimates ($Q_t(a)$) over time.
*   Observe **$\epsilon=0.10$ and $0.50$**: The solid lines rapidly converge to their corresponding dashed lines. The high exploration ensures every medicine is sampled enough times to satisfy the **Law of Large Numbers**.
*   Observe **$\epsilon=0.01$**: The lines are highly unstable and flat. Since exploration is extremely low, suboptimal arms are rarely sampled, meaning their estimated $Q$-values remain stuck at their initial, inaccurate values.

#### What to Focus on:
*   **Running Average Update Rule:** Be ready to explain the incremental update equation:
    $$Q_{n+1} = Q_n + \frac{1}{n} \left[ R_n - Q_n \right]$$
    Explain that $\left[ R_n - Q_n \right]$ is the **temporal difference error** (prediction error) and $\frac{1}{n}$ is the step size.

---

### 📊 Plot 3: `mab_final_rewards_bar.png` (Final Performance Summary)

#### What is it?
A vertical bar chart comparing the final cumulative rewards of all implemented strategies at patient 1000.

#### How to Read it:
*   This is a summary graph that immediately shows the winner. **UCB1** and **$\epsilon=0.10$** stand tall, while **$\epsilon=0.50$** and failed **Immediate Exploitation** runs are significantly shorter.

#### What to Focus on:
*   **The Sweet Spot of Exploration:** Use this plot to show that both **too much exploration** (which wastes trials on bad medicines) and **too little exploration** (which risks committing to a bad medicine permanently) degrade performance. The optimal lies in **adaptive, confidence-based exploration** (UCB1).

---

## 🚁 Part 2: Dynamic Programming (DP) — Offline Planning

Dynamic Programming represents **model-based offline planning**. In this paradigm:
*   The agent has **perfect, complete knowledge of the environment** (it knows the map transition probabilities $P(s'|s,a)$ and the reward function $R(s,a,s')$).
*   The agent does not interact with the world to learn; instead, it performs mathematical sweeps over the entire state space offline to calculate the absolute optimal policy.

---

### 📊 Plot 4: `dp_convergence.png` (Value Iteration Convergence)

#### What is it?
A dual-panel plot showing the number of Value Iteration sweeps (x-axis) vs. the maximum value change ($\Delta$) in the state space (y-axis) on both linear and logarithmic scales.

#### How to Read it:
*   On the **linear plot**, the curve drops sharply, crossing your stopping threshold ($\theta = 10^{-3}$) at exactly **94 iterations**.
*   On the **logarithmic plot**, the curve is a **perfect straight downward line**.

#### What to Focus on:
*   **Contraction Mapping:** A straight line on a log scale proves **exponential convergence**. This is the physical proof of the **Banach Fixed Point Theorem**. Since the discount factor $\gamma = 0.95 < 1$, the Bellman Optimality Operator is a contraction mapping, mathematically guaranteeing that Value Iteration will converge to a unique, stable, optimal value function $V^*$.

---

### 📊 Plot 5: `dp_policy_grid.png` (The "Mind" of the Drone)

#### What is it?
A 6-panel grid layout of your $5\times5$ map. Each cell contains an **arrow** indicating the optimal action ($\uparrow, \downarrow, \leftarrow, \rightarrow$, or $\bullet$ for Hover) determined by the policy $\pi^*(s)$ under different battery levels and target configurations.

#### How to Read it:
*   **High Battery (Battery = 10):** Look at the arrows—they point directly toward the rescue targets ($R_1$ at `(1,2)` and $R_2$ at `(4,0)`). The drone has plenty of energy, so it ignores the charging station ($C$ at `(2,1)`) and navigates greedily to save civilians.
*   **Low Battery (Battery = 2):** Notice how all arrows bend and point directly toward the charging station `(2,1)`, even if the drone is standing adjacent to a target. Avoiding battery depletion (which carries a severe $-20$ penalty) becomes the absolute priority.
*   **Target Status Change:** Compare the panel where both targets are active to the panel where Target 1 is already rescued. The arrows surrounding coordinate `(1,2)` no longer point toward it; instead, they route the drone directly toward Target 2 at `(4,0)`.

#### What to Focus on:
*   **The Markov Property:** Use this plot to explain why position alone is not a valid Markov state. A state must contain *all* information needed to decide the future. Since the optimal action at coordinate `(4,0)` changes based on battery and target status, the state space must be expanded to a 5-tuple: $s = (row, col, battery, t_1, t_2)$.

---

### 📊 Plot 6: `dp_value_heatmap.png` (The Utility Terrain)

#### What is it?
A 6-panel heatmap representing the state-values $V^*(s)$ for every coordinate. Bright, warm colors (red/orange) represent highly valuable states, while cool, dark colors (blue/purple) represent low-value, dangerous states.

#### How to Read it:
*   **Target Proximity:** Cells adjacent to the targets are bright red, representing high state-value.
*   **Danger Zones:** The cells containing or surrounding the three danger zones ($D$ at `(2,2)`, `(2,4)`, `(3,2)`) are dark blue/purple, reflecting their heavy negative reward.
*   **Battery Influence:** At Battery = 2, the charging station `(2,1)` lights up as a bright yellow/red beacon of safety, while the rest of the map turns dark, showing that states far from the charger are now highly risky and low in value.

#### What to Focus on:
*   **Temporal Discounting:** Explain that state-values represent *discounted future rewards*. A cell adjacent to a target is worth more than a cell 3 steps away because future rewards are discounted by $\gamma^k$ (where $\gamma = 0.95$ and $k$ is the number of steps).

---

### 📊 Plot 7: `dp_curse_of_dimensionality.png` (Scalability Limits)

#### What is it?
A bar chart demonstrating how the size of the state space grows exponentially as you scale the grid size (from $5\times5$ to $10\times10$) or increase the number of rescue targets.

#### How to Read it:
*   Your $5\times5$ grid with 2 targets has **1,100 states**.
*   A larger $10\times10$ grid with 10 targets would require **billions of states**, representing an exponential spike.

#### What to Focus on:
*   **The Curse of Dimensionality:** Explain that tabular Dynamic Programming requires storing a value for every single state and sweeping the entire state space during every iteration. At scale, memory and computational power are completely exhausted.
*   **The Deep RL Solution:** To solve large-scale or continuous real-world problems, we must transition to **Deep RL** (like DQN, PPO, or MuZero). Deep RL uses **neural networks** as value function approximators ($V_\theta(s) \approx V^*(s)$) instead of rigid tables, allowing the model to generalize and handle infinite or massive state spaces.

---

## 🎙️ Sample Viva / Project Defense Questions

To ensure you are fully prepared, review these typical questions an evaluator might ask:

1.  **Q: Why does UCB1 perform better than $\epsilon$-greedy in the MAB clinical trial?**
    *   *A:* $\epsilon$-greedy explores randomly, wasting trials on clearly bad medicines. UCB1 explores *optimistically under uncertainty*. It allocates more exploration trials to arms that are either promising or highly uncertain, and mathematically guarantees optimal logarithmic regret.
2.  **Q: In Part 2, what would happen if we set the discount factor $\gamma = 1$?**
    *   *A:* If $\gamma = 1$ in an infinite horizon task, the value function could infinite-loop or diverge to infinity. Moreover, setting $\gamma = 1$ removes the contraction mapping guarantee, meaning Value Iteration might fail to converge.
3.  **Q: Why does the drone choose to hover at a charging station instead of moving?**
    *   *A:* The hover action is specifically designed such that hovering *on* a charging station increases battery capacity by $+2$ units without leaving the safety of the charger, whereas moving away consumes battery.
4.  **Q: How does the drone handle the 20% wind zone probability?**
    *   *A:* In wind zones, the transition dynamics distribute 20% of the transition probability mass uniformly across all four adjacent directions. The Value Iteration solver calculates the expected value across all these potential drift directions, leading the drone's optimal policy to naturally take detours around wind zones when battery limits allow.
