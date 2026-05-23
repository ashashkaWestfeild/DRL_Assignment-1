# Deep Reinforcement Learning (DRL) — Core Foundations
## Multi-Armed Bandits (MAB) & Dynamic Programming (DP)
### BITS Pilani WILP | DRL Assignment 1 | Group 84 | Student ID: 2025aa05710

---

## 📖 Project Overview
This repository contains the complete implementation and theoretical analysis for **DRL Assignment 1 (BITS Pilani WILP)**. The project bridges the two fundamental paradigms of Reinforcement Learning:
1. **Multi-Armed Bandits (Part 1):** Model-free online learning under uncertainty, simulating an adaptive clinical trial treatment recommendation system.
2. **Dynamic Programming (Part 2):** Model-based planning via Value Iteration in a custom-designed stochastically disturbed grid world for an autonomous rescue drone.

---

## 📓 Repository Contents

*   **`Team_84_MAB_executed.ipynb`**: Complete Multi-Armed Bandit implementation with full outputs and charts.
*   **`Team_84_DP_executed.ipynb`**: Complete Dynamic Programming rescue drone implementation with full value maps and policy arrows.
*   **`generate_notebooks.py`**: Clean generator script to easily refresh or recreate the submission-ready Jupyter notebooks.
*   **`Assignment_Report.md`**: Exhaustive theoretical manual containing complete derivations, mathematical algorithms, comparisons, and structural challenges/mitigations.
*   **`plots/`**: Directory containing high-resolution visual plots of convergence, heatmaps, policy grids, and comparisons.
*   **`LICENSE`**: MIT Open-Source License.

---

## 💻 Tech Stack & Dependencies
The project is built entirely in python using modern standard scientific and visualization packages:
*   **Python 3.11+**
*   **NumPy** (Numerical and array computation)
*   **Matplotlib** (Data visualization and plotting)
*   **Jupyter/Notebook** (Interactive environments)
*   **nbformat / nbconvert** (Notebook formatting and execution)

To install the required dependencies:
```bash
pip install numpy matplotlib notebook nbformat nbconvert pypdf
```

---

## 🎮 Part 1: Multi-Armed Bandits (MAB)
### Scenario
An adaptive treatment recommendation system that assigns K=5 medicines to patients over 1000 trials, balancing the exploration of unknown treatments with the exploitation of known successful treatments.

### Implemented Strategies
1.  **Immediate Exploitation (Pure Greedy)**: Initial 10 exploratory trials per arm, followed by a permanent greedy commitment to the highest average arm.
2.  **$\epsilon$-Greedy Strategy**: Controlled randomized trials using ongoing exploration at rates of $\epsilon \in \{0.01, 0.10, 0.50\}$.
3.  **UCB1 (Upper Confidence Bound)**: Principle of *optimism in the face of uncertainty*, dynamically calculating:
    $$A_t = \arg\max_{a} \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]$$

---

## 🚁 Part 2: Dynamic Programming (DP)
### Scenario
An autonomous rescue drone operating in a disaster grid zone must rescue stranded civilians, avoid dangerous fire/radiation cells, manage a battery budget, and navigate random wind disturbances.

### State & MDP Specification
*   **State Space**: $S = (row, col, battery, t_1, t_2) \rightarrow 1,100 \text{ states}$.
*   **Action Space**: $\text{Up}, \text{Down}, \text{Left}, \text{Right}, \text{Hover}$.
*   **Wind Drift**: 20% random drift probability in wind zones.
*   **Battery Constraint**: 10 units max capacity.
*   **Solver**: Tabular **Value Iteration** using the Bellman Optimality Equation:
    $$V^*(s) = \max_{a} \sum_{s'} P(s' | s, a) \left[ R(s, a, s') + \gamma V^*(s') \right]$$
    converged in 94 iterations under stopping threshold $\theta = 10^{-3}$.

---

## 📈 Visualizations (Stored in `./plots/`)
*   **`mab_comparison_main.png`**: Multi-arm comparison showing cumulative reward and percentage of optimal action selection.
*   **`dp_policy_grid.png`**: Optimal policy arrows demonstrating danger avoidance, rescue sequences, and low-battery charging routing.
*   **`dp_value_heatmap.png`**: State-value $V^*(s)$ heatmaps at multiple battery levels.
*   **`dp_curse_of_dimensionality.png`**: Dimensionality scaling and DP vs Deep RL scalability analysis.

---

## 🛠️ How to Run & Reproduce
To regenerate both executed notebooks from the generator script:
```bash
python generate_notebooks.py
```
To run the execution pipeline manually to test for correctness:
```bash
python -m jupyter nbconvert --to notebook --execute Team_84_MAB.ipynb --output Team_84_MAB_executed.ipynb
python -m jupyter nbconvert --to notebook --execute Team_84_DP.ipynb --output Team_84_DP_executed.ipynb
```

---

## 📄 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
