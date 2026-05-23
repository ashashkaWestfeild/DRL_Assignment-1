"""
DRL Assignment 1 - Notebook Generator
======================================
This script programmatically generates two Jupyter notebooks:
  1. Team_84_MAB.ipynb  - Part 1: Multi-Armed Bandits
  2. Team_84_DP.ipynb   - Part 2: Dynamic Programming

Run this script once to create the notebooks, then open them in JupyterLab.

Usage:
    python generate_notebooks.py
"""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


def nb_md(text: str) -> nbformat.NotebookNode:
    """Helper: create a markdown cell."""
    return new_markdown_cell(text)


def nb_code(code: str) -> nbformat.NotebookNode:
    """Helper: create a code cell."""
    return new_code_cell(code)


# ============================================================
#  NOTEBOOK 1: Multi-Armed Bandits
# ============================================================

def make_mab_notebook() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    }

    cells = []

    # ── CELL 0: Title ──────────────────────────────────────────
    cells.append(nb_md("""# 🏥 DRL Assignment 1 — Part 1: Multi-Armed Bandit (MAB)
## Adaptive Treatment Recommendation System

| Field | Value |
|-------|-------|
| **Course** | Deep Reinforcement Learning (DRL) |
| **Institute** | BITS Pilani – WILP |
| **Assignment** | Lab Assignment 1 — Part 1 |
| **Group Number** | 84 |
| **Student ID** | 2025aa05710 |
| **Topic** | Multi-Armed Bandit: Exploration vs Exploitation |

---
> **📖 Reference:** Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, Chapter 2.  
> [Download PDF](http://incompleteideas.net/book/RLbook2018.pdf)
"""))

    # ── CELL 1: VM Metadata ────────────────────────────────────
    cells.append(nb_md("## 🖥️ 0. Execution Environment"))
    cells.append(nb_code("""# ─────────────────────────────────────────────────────────────
# CELL: Environment Metadata
# PURPOSE: Capture execution timestamp and machine ID for submission.
#          The assignment requires this to be printed at the top.
# ─────────────────────────────────────────────────────────────
import datetime
import socket
import platform
import os

# Get current execution timestamp in IST (UTC+5:30)
utc_now = datetime.datetime.now(datetime.timezone.utc)
ist_offset = datetime.timedelta(hours=5, minutes=30)
ist_now = utc_now + ist_offset

# Get the machine/VM identifier
hostname = socket.gethostname()

# Get OS information
os_info = platform.platform()

print("=" * 60)
print("  DRL ASSIGNMENT 1 — PART 1: MULTI-ARMED BANDITS")
print("=" * 60)
print(f"  Execution Timestamp (IST) : {ist_now.strftime('%Y-%m-%d %H:%M:%S IST')}")
print(f"  Machine ID / Hostname     : {hostname}")
print(f"  Operating System          : {os_info}")
print(f"  Python Version            : {platform.python_version()}")
print(f"  Group Number              : 84")
print(f"  Student ID                : 2025aa05710")
print("=" * 60)
"""))

    # ── CELL 2: Imports ────────────────────────────────────────
    cells.append(nb_md("## 📦 1. Imports & Reproducibility"))
    cells.append(nb_md("""We begin by importing all necessary libraries and setting a **fixed random seed**.

> **Why set a random seed?**  
> Our clinical simulation involves Bernoulli random variables (did the patient recover?). Without a fixed seed, every run produces different results, making the notebook non-reproducible. Setting `np.random.seed(42)` ensures that the *same sequence* of pseudo-random numbers is used every run — a critical requirement in scientific computing and ML research.
"""))
    cells.append(nb_code("""# ─────────────────────────────────────────────────────────────
# CELL: Imports & Configuration
# PURPOSE: Import all libraries and set reproducibility seed.
# ─────────────────────────────────────────────────────────────
import numpy as np               # Numerical computing: arrays, random numbers, math
import pandas as pd              # DataFrames: tabular data display
import matplotlib.pyplot as plt  # Plotting: all visualizations
import matplotlib.patches as mpatches  # Legend patches for plots
import warnings
import os
from typing import Tuple, List, Dict

# ── Create output directories ────────────────────────────────
os.makedirs('plots', exist_ok=True)  # Directory to save all generated figures

# ── Styling ──────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')  # Professional dark-grid plot style
plt.rcParams.update({
    'figure.dpi': 120,           # High-resolution figures
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'lines.linewidth': 2.0,
})
warnings.filterwarnings('ignore')

# ── Reproducibility ──────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Global Parameters ────────────────────────────────────────
GROUP_NUMBER = 84          # G = 84
STUDENT_ID   = "2025aa05710"
N_PATIENTS   = 1000        # Total number of sequential patients (T = 1000)

print(f"✅ Libraries loaded | Seed={SEED} | Group={GROUP_NUMBER} | Patients={N_PATIENTS}")
"""))

    # ── CELL 3: Theory — MAB ───────────────────────────────────
    cells.append(nb_md("""## 🧠 2. Theory: The Multi-Armed Bandit Problem

### What is a Bandit Problem?

Imagine you are a doctor managing a hospital. You have **K medicines** available for treating a disease. Each medicine has a **hidden** probability of curing a patient:

$$P_i = \\text{unknown true recovery probability of medicine } i$$

You treat patients **sequentially** (one at a time). After each treatment, you observe whether the patient recovered (1) or not (0). Your goal is to **maximize total patient recoveries** over all 1000 patients.

This is the **Multi-Armed Bandit Problem** — named after slot machines where each "arm" has a different payout rate.

---

### The Core Tension: Explore vs. Exploit

| Strategy | Description | Risk |
|----------|-------------|------|
| **Exploitation** | Use the medicine with the highest observed success rate | May miss a better medicine not yet tried enough |
| **Exploration** | Try all medicines, even less-tested ones | Wastes treatments on potentially bad medicines |

**The challenge:** We don't know which medicine is best *a priori*. We must **learn from experience** while simultaneously **maximizing outcomes**.

### Regret: The Formal Measure of Performance

$$\\text{Regret}_T = T \\cdot q^*(a^*) - \\mathbb{E}\\left[\\sum_{t=1}^{T} R_t\\right]$$

where $q^*(a^*) = \\max_i P_i$ is the true success rate of the best medicine.  
**Lower regret = better algorithm.**
"""))

    # ── CELL 4: Dataset Design ─────────────────────────────────
    cells.append(nb_md("""## 🗄️ Task 1: Dataset Design

### Parameter Derivation for Group G = 84

**Step 1 — Number of Medicines (K):**

$$K = (G \\bmod 3) + 5 = (84 \\bmod 3) + 5 = 0 + 5 = 5$$

**Step 2 — Hidden Success Probabilities:**

$$P_i = 0.4 + ((G + i) \\bmod 6) \\times 0.07, \\quad i \\in \\{0, 1, 2, 3, 4\\}$$

Since $84 \\bmod 6 = 0$, this simplifies to $P_i = 0.4 + (i \\bmod 6) \\times 0.07$

**Step 3 — Patient Severity:**

$$\\text{Severity}(\\text{patient\\_id}) = (\\text{patient\\_id} \\bmod 5) + 1 \\in \\{1, 2, 3, 4, 5\\}$$

**Step 4 — Utility Score (Reward):**

$$\\text{Reward} = \\text{clinical\\_outcome} \\times \\frac{10 - \\text{Severity}}{10}$$

A recovery under low severity (1) gives reward 0.9; under high severity (5) gives 0.5, because a high-severity patient is harder to treat, so full credit is not given for the same clinical intervention.
"""))
    cells.append(nb_code("""# ─────────────────────────────────────────────────────────────
# CELL: Dataset Design — Synthetic Clinical Environment
# PURPOSE: Generate the patient dataset and compute all parameters
#          for Group G = 84.
# ─────────────────────────────────────────────────────────────

# ── Step 1: Compute Number of Medicines ──────────────────────
G = GROUP_NUMBER
K = (G % 3) + 5
print(f"Group G = {G}")
print(f"Number of Medicines K = ({G} % 3) + 5 = {G % 3} + 5 = {K}")

# ── Step 2: Compute Hidden Success Probabilities ─────────────
# P_i = 0.4 + ((G + i) mod 6) * 0.07
# The hidden probabilities are unknown to the agent during simulation,
# but we define them here as the 'ground truth' of our synthetic world.
TRUE_PROBS = np.array([0.4 + ((G + i) % 6) * 0.07 for i in range(K)])
print("\\nHidden Success Probabilities (P_i):")
for i, p in enumerate(TRUE_PROBS):
    marker = " ← OPTIMAL" if p == TRUE_PROBS.max() else ""
    print(f"  Medicine {i}: P_{i} = 0.4 + ({(G+i)%6}) × 0.07 = {p:.2f}{marker}")

OPTIMAL_ARM = int(np.argmax(TRUE_PROBS))
print(f"\\nOptimal Medicine: {OPTIMAL_ARM} (P = {TRUE_PROBS[OPTIMAL_ARM]:.2f})")

# ── Step 3: Generate Patient Dataset ─────────────────────────
# For each patient (0 to 999), compute their severity score.
# This does NOT depend on which medicine is given — severity is
# an intrinsic property of the patient's condition upon arrival.
patient_ids  = np.arange(N_PATIENTS)
severities   = (patient_ids % 5) + 1   # Severity cycles 1,2,3,4,5,1,2,3,...

# ── Step 4: Define Reward Function ───────────────────────────
def compute_reward(outcome: int, severity: int) -> float:
    \"\"\"
    Compute the clinical utility score for a single treatment.

    Parameters
    ----------
    outcome : int
        1 if the patient recovered, 0 if not.
    severity : int
        Patient severity score (1 = mild, 5 = severe).

    Returns
    -------
    float
        The clinical utility (reward).

    Formula
    -------
    Reward = clinical_outcome × (10 - Severity) / 10
    - A recovery (outcome=1) under severity 1 → reward = 0.9 (mild, easy)
    - A recovery (outcome=1) under severity 5 → reward = 0.5 (severe, hard)
    - No recovery (outcome=0) → reward = 0.0 regardless of severity
    \"\"\"
    return outcome * (10 - severity) / 10.0

# ── Preview First 10 Rows ────────────────────────────────────
# We simulate 10 rows using Medicine 4 (optimal) to show the data format.
# In actual experiments, the medicine choice is determined by each algorithm.
np.random.seed(SEED)
preview_outcomes  = np.random.binomial(1, TRUE_PROBS[OPTIMAL_ARM], 10)
preview_severities = severities[:10]
preview_rewards   = [compute_reward(o, s) for o, s in zip(preview_outcomes, preview_severities)]

preview_df = pd.DataFrame({
    'Patient ID':   range(10),
    'Severity':     preview_severities,
    'Medicine':     [f'Med {OPTIMAL_ARM}'] * 10,
    'P(recovery)':  [TRUE_PROBS[OPTIMAL_ARM]] * 10,
    'Outcome (0/1)': preview_outcomes,
    'Reward':       [f'{r:.1f}' for r in preview_rewards],
})
print("\\n📋 First 10 Patients (simulated with Optimal Medicine for illustration):")
print(preview_df.to_string(index=False))

# ── Summary Statistics ────────────────────────────────────────
print("\\n📊 Parameter Summary:")
print(f"  Total Patients   : {N_PATIENTS}")
print(f"  K (Medicines)    : {K}")
print(f"  Severity Range   : {severities.min()} to {severities.max()} (cycles 1-5)")
print(f"  Reward Range     : {compute_reward(1,5):.1f} (severity=5) to {compute_reward(1,1):.1f} (severity=1)")
print(f"  Avg Expected Reward (Optimal): {TRUE_PROBS[OPTIMAL_ARM] * 0.70:.4f}")
"""))

    # ── CELL 5: Theory — Immediate Exploitation ────────────────
    cells.append(nb_md("""## 💊 Task 2: Immediate Exploitation (Pure Greedy Strategy)

### Theory

This is the simplest possible strategy:
1. **Initialization Phase:** Test each medicine exactly $N_{\\text{init}} = 10$ times to get initial estimates.
2. **Exploitation Phase:** Compute the empirical success rate $\\hat{Q}(a) = \\frac{\\text{successes}_a}{10}$ for each medicine, then permanently commit to $a^* = \\arg\\max_a \\hat{Q}(a)$.

### Why This Can Fail

After only 10 trials per medicine, the estimated probabilities have **high variance**. For a Binomial$(10, p)$ random variable:

$$\\text{Std}(\\hat{Q}(a)) = \\sqrt{\\frac{p(1-p)}{n}} = \\sqrt{\\frac{0.68 \\times 0.32}{10}} \\approx 0.147$$

This means our estimate for the best medicine (P=0.68) has a standard deviation of **14.7 percentage points** — large enough that a suboptimal medicine could easily appear better by chance in 10 samples.

### Expected Performance

If the correct medicine is identified: cumulative reward ≈ $950 \\times 0.68 \\times 0.70 = 452$  
If the wrong medicine is permanently chosen (e.g. Medicine 0): ≈ $950 \\times 0.40 \\times 0.70 = 266$
"""))
    cells.append(nb_code("""# ─────────────────────────────────────────────────────────────
# CELL: Task 2 — Immediate Exploitation Algorithm
# PURPOSE: Implement the "test 10 times then exploit" strategy.
# ─────────────────────────────────────────────────────────────
np.random.seed(SEED)

# ── Constants ─────────────────────────────────────────────────
N_INIT = 10   # Number of initial trials per medicine (initialization phase)

def run_immediate_exploitation(
    true_probs: np.ndarray,
    severities: np.ndarray,
    n_init: int = 10,
    seed: int = SEED
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    \"\"\"
    Run the Immediate Exploitation (Pure Greedy) strategy.

    Strategy:
    ---------
    Phase 1 (Initialization): Try each medicine exactly `n_init` times
    in a round-robin fashion. This gives us initial quality estimates.

    Phase 2 (Exploitation): Select the medicine with the highest average
    outcome in Phase 1, then use it for ALL remaining patients.

    Parameters
    ----------
    true_probs : np.ndarray
        The ground-truth success probabilities for each medicine.
    severities : np.ndarray
        Pre-computed severity scores for all N patients.
    n_init : int
        Number of initial trials per medicine.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    cumulative_rewards : np.ndarray
        Running sum of rewards at each patient step.
    chosen_arms : np.ndarray
        Which medicine was selected at each step.
    best_arm_selected : int
        The medicine selected for long-term exploitation.
    init_estimate : float
        The estimated Q-value of the selected medicine after initialization.
    \"\"\"
    np.random.seed(seed)
    K = len(true_probs)
    N = len(severities)

    # Arrays to store results
    rewards     = np.zeros(N)    # Reward received at each step
    chosen_arms = np.zeros(N, dtype=int)  # Medicine chosen at each step

    # ── Phase 1: Initialization ────────────────────────────────
    # Try each medicine n_init times in order: med0, med1, ..., med(K-1), med0, ...
    # This is a systematic round-robin initialization.
    init_successes = np.zeros(K, dtype=int)  # Count successes for each medicine

    for step in range(n_init * K):
        medicine = step % K          # Cycle through medicines 0,1,2,3,4 repeatedly
        patient_id = step            # This step corresponds to patient `step`
        severity = severities[patient_id]

        # Simulate clinical outcome: Bernoulli trial with P = true_probs[medicine]
        # np.random.binomial(1, p) returns 1 with probability p, 0 otherwise
        outcome = np.random.binomial(1, true_probs[medicine])
        reward  = compute_reward(outcome, severity)

        # Record the results
        rewards[step]      = reward
        chosen_arms[step]  = medicine
        init_successes[medicine] += outcome  # Count raw successes (not rewards)

    # ── Compute Initial Estimates ──────────────────────────────
    # Q_hat(a) = total successes for medicine a / number of trials
    # (We estimate using raw Bernoulli outcomes, not utility scores,
    #  to match the probability scale for comparison)
    q_estimates = init_successes / n_init

    print("  Initial Estimates after Phase 1 (10 trials per medicine):")
    for i, (q, p) in enumerate(zip(q_estimates, true_probs)):
        diff = q - p
        print(f"    Medicine {i}: Q_hat = {q:.2f} | True P = {p:.2f} | Error = {diff:+.2f}")

    # ── Select Best Medicine ───────────────────────────────────
    # This is the GREEDY selection: choose the medicine that appeared best
    # based on our (potentially noisy) initial 10-trial estimates.
    best_arm = int(np.argmax(q_estimates))
    best_estimate = q_estimates[best_arm]
    was_optimal = "✅ CORRECT" if best_arm == OPTIMAL_ARM else f"❌ WRONG (true best is Med {OPTIMAL_ARM})"
    print(f"\\n  Selected for exploitation: Medicine {best_arm} (Q_hat={best_estimate:.2f}) → {was_optimal}")

    # ── Phase 2: Exploitation ──────────────────────────────────
    # From patient index (n_init * K) to (N-1), always choose best_arm
    for step in range(n_init * K, N):
        severity = severities[step]
        outcome  = np.random.binomial(1, true_probs[best_arm])
        reward   = compute_reward(outcome, severity)

        rewards[step]     = reward
        chosen_arms[step] = best_arm

    # ── Compute Cumulative Rewards ─────────────────────────────
    # cumulative_rewards[t] = sum of all rewards from patient 0 to patient t
    cumulative_rewards = np.cumsum(rewards)

    return cumulative_rewards, chosen_arms, best_arm, best_estimate

# ── Run the strategy ──────────────────────────────────────────
print("🔬 Running Immediate Exploitation Strategy...")
print(f"  Initialization: {N_INIT} trials × {K} medicines = {N_INIT * K} patients")
print()

imm_cumulative, imm_arms, imm_best_arm, imm_best_q = run_immediate_exploitation(
    TRUE_PROBS, severities, n_init=N_INIT
)

print(f"\\n📊 Immediate Exploitation Results:")
print(f"  Total Cumulative Reward: {imm_cumulative[-1]:.2f}")
print(f"  Average Reward/Patient:  {imm_cumulative[-1]/N_PATIENTS:.4f}")
print(f"  Exploitation Medicine:   {imm_best_arm}")
print(f"  Exploited Optimal?:      {'Yes ✅' if imm_best_arm == OPTIMAL_ARM else 'No ❌'}")
"""))

    # ── CELL 6: Theory — ε-Greedy ──────────────────────────────
    cells.append(nb_md("""## 🎯 Task 3: Controlled Clinical Trial (ε-Greedy Strategy)

### Theory

The ε-Greedy algorithm addresses the exploration-exploitation problem with a simple but effective rule:

$$A_t = \\begin{cases} \\arg\\max_a Q_t(a) & \\text{with probability } 1 - \\varepsilon \\quad \\text{(exploit)} \\\\ \\text{Uniform}(\\{0, 1, \\ldots, K-1\\}) & \\text{with probability } \\varepsilon \\quad \\text{(explore)} \\end{cases}$$

### Incremental Update Rule

We maintain a running average of rewards for each medicine. After selecting medicine $A_t$ and observing reward $R_t$:

$$N(A_t) \\leftarrow N(A_t) + 1$$
$$Q(A_t) \\leftarrow Q(A_t) + \\underbrace{\\frac{1}{N(A_t)}}_{\\text{step size}} \\underbrace{\\left[R_t - Q(A_t)\\right]}_{\\text{prediction error}}$$

**Why this formula?** Expanding it reveals it's just a running average:

$$Q_{n+1} = \\frac{1}{n}\\sum_{i=1}^{n} R_i = Q_n + \\frac{1}{n}[R_n - Q_n]$$

The term $[R_n - Q_n]$ is the **TD (Temporal Difference) error** — how wrong our current estimate was. This incremental form avoids storing all historical rewards.

### Impact of ε

| ε | Explore Rate | Effect |
|---|-------------|--------|
| 0.01 | 1% | Slow learning; near-optimal long-term |
| 0.10 | 10% | Balanced; standard choice |
| 0.50 | 50% | Fast learning; permanent 50% waste |
"""))
    cells.append(nb_code("""# ─────────────────────────────────────────────────────────────
# CELL: Task 3 — ε-Greedy Strategy
# PURPOSE: Implement epsilon-greedy and compare 3 epsilon values.
# ─────────────────────────────────────────────────────────────

def run_epsilon_greedy(
    true_probs: np.ndarray,
    severities: np.ndarray,
    epsilon: float,
    seed: int = SEED
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    \"\"\"
    Run the ε-Greedy Multi-Armed Bandit strategy.

    Parameters
    ----------
    true_probs : np.ndarray
        Ground-truth success probabilities for each arm.
    severities : np.ndarray
        Pre-computed severity scores for all patients.
    epsilon : float
        Exploration probability (0 = pure greedy, 1 = pure random).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    cumulative_rewards : np.ndarray
        Running cumulative reward at each timestep.
    chosen_arms : np.ndarray
        Medicine selected at each timestep.
    q_history : np.ndarray, shape (N, K)
        Q-value estimates at each timestep for all medicines.
    \"\"\"
    np.random.seed(seed)
    K = len(true_probs)
    N = len(severities)

    # ── Initialize Q-value estimates and counts ────────────────
    # Q[a] = running average utility score for medicine a
    # We initialize to 0 (no information about any medicine yet)
    Q = np.zeros(K)

    # N_counts[a] = number of times medicine a has been selected
    # Used as the denominator in the incremental average update
    N_counts = np.zeros(K)

    # Storage arrays
    rewards     = np.zeros(N)
    chosen_arms = np.zeros(N, dtype=int)
    q_history   = np.zeros((N, K))

    for t in range(N):
        # ── Action Selection: ε-Greedy ─────────────────────────
        # With probability ε: EXPLORE — pick a random medicine
        # With probability 1-ε: EXPLOIT — pick the best known medicine
        if np.random.rand() < epsilon:
            # EXPLORATION: np.random.randint samples uniformly from {0, ..., K-1}
            action = np.random.randint(K)
        else:
            # EXPLOITATION: argmax returns the index of the maximum Q-value
            # If multiple medicines tie for best, argmax picks the lowest index
            action = np.argmax(Q)

        # ── Simulate Patient Outcome ───────────────────────────
        # Bernoulli trial: patient recovers with probability true_probs[action]
        outcome  = np.random.binomial(1, true_probs[action])
        severity = severities[t]
        reward   = compute_reward(outcome, severity)

        # ── Record Results ─────────────────────────────────────
        rewards[t]     = reward
        chosen_arms[t] = action

        # ── Incremental Q-Value Update ─────────────────────────
        # Only the SELECTED action's Q-value is updated.
        # Q-values for other medicines remain unchanged this timestep.
        N_counts[action] += 1   # Increment count for this medicine

        # Incremental mean update (Sutton & Barto Eq. 2.3):
        # Q_new(a) = Q_old(a) + (1/N(a)) * [R_t - Q_old(a)]
        # This is mathematically equivalent to:  Q(a) = mean of all past rewards for a
        Q[action] += (1.0 / N_counts[action]) * (reward - Q[action])

        # Save Q-value snapshot for analysis
        q_history[t] = Q.copy()

    cumulative_rewards = np.cumsum(rewards)
    return cumulative_rewards, chosen_arms, q_history


# ── Run for 3 epsilon values ──────────────────────────────────
EPSILONS = [0.01, 0.10, 0.50]
eps_results = {}

print("🔬 Running ε-Greedy Strategy for ε ∈ {0.01, 0.10, 0.50}...")
print()

for eps in EPSILONS:
    cum_r, arms, q_hist = run_epsilon_greedy(TRUE_PROBS, severities, epsilon=eps)
    eps_results[eps] = {
        'cumulative_rewards': cum_r,
        'chosen_arms': arms,
        'q_history': q_hist,
    }

    # Compute % of time the optimal arm was chosen (excluding early exploration)
    pct_optimal = np.mean(arms == OPTIMAL_ARM) * 100

    print(f"  ε = {eps:.2f}:")
    print(f"    Total Cumulative Reward:  {cum_r[-1]:.2f}")
    print(f"    Average Reward/Patient:   {cum_r[-1]/N_PATIENTS:.4f}")
    print(f"    % Time Optimal Arm Chosen: {pct_optimal:.1f}%")
    print(f"    Final Q-estimates: {q_hist[-1]}")
    print()

# ── Plot Q-value Convergence ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("ε-Greedy: Q-Value Convergence for Each Medicine", fontsize=14, fontweight='bold')

colors_med = ['#e74c3c','#e67e22','#f1c40f','#2ecc71','#3498db']

for ax, eps in zip(axes, EPSILONS):
    q_hist = eps_results[eps]['q_history']
    for med_i in range(K):
        ax.plot(q_hist[:, med_i], color=colors_med[med_i],
                label=f'Med {med_i} (True P={TRUE_PROBS[med_i]:.2f})', alpha=0.85)
    # Draw horizontal lines for true probabilities (scaled to reward scale)
    for med_i in range(K):
        ax.axhline(TRUE_PROBS[med_i] * 0.70, color=colors_med[med_i],
                   linestyle='--', alpha=0.4)
    ax.set_title(f'ε = {eps:.2f}', fontsize=13)
    ax.set_xlabel('Patient Number')
    ax.set_ylabel('Q(a) — Estimated Utility')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(0, N_PATIENTS)

plt.tight_layout()
plt.savefig('plots/eps_greedy_convergence.png', bbox_inches='tight')
plt.show()
print("📊 Q-Value convergence plots saved.")
"""))

    # ── CELL 7: Theory — UCB1 ──────────────────────────────────
    cells.append(nb_md("""## 🎲 Task 4: Confidence-Based Strategy (UCB1)

### Motivation: Why Not ε-Greedy?

ε-Greedy has a critical flaw: it explores **uniformly at random**. It spends the same effort re-testing a medicine it has already tried 900 times (highly certain) as one it has only tried 5 times (highly uncertain).

UCB1 (Upper Confidence Bound) solves this by exploring **where uncertainty is highest**.

### The UCB1 Algorithm

At each timestep $t$, select the medicine that maximizes:

$$A_t = \\arg\\max_{a \\in \\mathcal{A}} \\left[ \\underbrace{Q_t(a)}_{\\text{Exploitation}} + \\underbrace{c\\sqrt{\\frac{\\ln t}{N_t(a)}}}_{\\text{Exploration Bonus}} \\right]$$

**Intuition:**

- $Q_t(a)$: Best estimate of medicine $a$'s quality so far (exploit what we know).
- $\\sqrt{\\ln t / N_t(a)}$: Confidence interval radius. This is large when $N_t(a)$ is small (uncertain) and shrinks as we gather more data. The $\\ln t$ term ensures we keep re-evaluating over time.
- $c$: A tuning parameter controlling how bold the exploration is. We use $c = \\sqrt{2}$ as suggested by theory.

### Theoretical Guarantee

UCB1 achieves **logarithmic regret** — proven optimal in the MAB setting (Auer et al., 2002):

$$\\text{Regret}_T \\leq \\sum_{a:\\, q^*(a) < q^*(a^*)} \\frac{8 \\ln T}{\\Delta_a} + \\left(1 + \\frac{\\pi^2}{3}\\right) \\Delta_a$$

where $\\Delta_a = q^*(a^*) - q^*(a)$ is the **sub-optimality gap** of medicine $a$.

For Group 84: $\\Delta_0 = 0.196, \\Delta_1 = 0.147, \\Delta_2 = 0.098, \\Delta_3 = 0.049$
"""))
    cells.append(nb_code("""# ─────────────────────────────────────────────────────────────
# CELL: Task 4 — UCB1 (Upper Confidence Bound) Strategy
# PURPOSE: Implement UCB1 algorithm with principled exploration.
# ─────────────────────────────────────────────────────────────

def run_ucb1(
    true_probs: np.ndarray,
    severities: np.ndarray,
    c: float = np.sqrt(2),
    seed: int = SEED
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    \"\"\"
    Run the UCB1 (Upper Confidence Bound) bandit strategy.

    UCB1 selects the arm with the highest upper confidence bound:
        A_t = argmax_a [ Q_t(a) + c * sqrt(ln(t) / N_t(a)) ]

    This is derived from Hoeffding's inequality: the true mean of arm a
    lies below Q_t(a) + epsilon with probability >= 1 - 2*exp(-2*N*epsilon^2).

    Parameters
    ----------
    true_probs : np.ndarray
        Ground-truth success probabilities for each arm.
    severities : np.ndarray
        Pre-computed severity scores for all patients.
    c : float
        Exploration coefficient (default = sqrt(2) for theory).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    cumulative_rewards : np.ndarray
    chosen_arms : np.ndarray
    ucb_values_history : np.ndarray, shape (N, K)
        The UCB index value for each arm at each step.
    \"\"\"
    np.random.seed(seed)
    K = len(true_probs)
    N = len(severities)

    # ── Initialize ─────────────────────────────────────────────
    Q        = np.zeros(K)   # Running average reward for each medicine
    N_counts = np.zeros(K)   # Number of times each medicine has been chosen
    rewards     = np.zeros(N)
    chosen_arms = np.zeros(N, dtype=int)
    ucb_history = np.zeros((N, K))

    # ── Initialization: Try each medicine once ─────────────────
    # UCB1 requires at least 1 observation per arm before computing
    # the confidence bound (to avoid division by zero in sqrt(ln t / N(a))).
    # We initialize by selecting each medicine exactly once.
    for k in range(K):
        severity = severities[k]
        outcome  = np.random.binomial(1, true_probs[k])
        reward   = compute_reward(outcome, severity)

        rewards[k]     = reward
        chosen_arms[k] = k
        N_counts[k]    = 1
        Q[k]           = reward   # After 1 sample, running avg = that sample

    # ── Main UCB1 Loop ─────────────────────────────────────────
    for t in range(K, N):
        # ── Compute UCB Index for Each Arm ────────────────────
        # UCB(a) = Q(a)   +   c * sqrt( ln(t) / N(a) )
        #           ^exploitation     ^exploration bonus
        #
        # np.log is natural log (base e). Using ln(t) ensures the exploration
        # bonus grows slowly enough to guarantee logarithmic regret.
        exploration_bonus = c * np.sqrt(np.log(t) / N_counts)
        ucb_values = Q + exploration_bonus

        # ── Select Action ──────────────────────────────────────
        # Choose the arm with the highest UCB index.
        # Unlike epsilon-greedy, this selection is DETERMINISTIC given Q and N_counts.
        # The randomness comes only from the observed outcomes.
        action = np.argmax(ucb_values)

        # ── Simulate and Record ────────────────────────────────
        severity = severities[t]
        outcome  = np.random.binomial(1, true_probs[action])
        reward   = compute_reward(outcome, severity)

        rewards[t]     = reward
        chosen_arms[t] = action
        ucb_history[t] = ucb_values

        # ── Update Q-values (same incremental formula as ε-Greedy) ──
        N_counts[action] += 1
        Q[action] += (1.0 / N_counts[action]) * (reward - Q[action])

    cumulative_rewards = np.cumsum(rewards)

    print(f"  UCB1 Final Q-estimates: {np.round(Q, 4)}")
    print(f"  UCB1 Final N_counts:    {N_counts.astype(int)}")
    print(f"  % Time Optimal Arm Chosen: {np.mean(chosen_arms == OPTIMAL_ARM)*100:.1f}%")

    return cumulative_rewards, chosen_arms, ucb_history


print("🔬 Running UCB1 Strategy (c = √2)...")
ucb_cumulative, ucb_arms, ucb_hist = run_ucb1(TRUE_PROBS, severities, c=np.sqrt(2))

print(f"\\n📊 UCB1 Results:")
print(f"  Total Cumulative Reward: {ucb_cumulative[-1]:.2f}")
print(f"  Average Reward/Patient:  {ucb_cumulative[-1]/N_PATIENTS:.4f}")
"""))

    # ── CELL 8: Comparative Analysis ───────────────────────────
    cells.append(nb_md("""## 📈 Task 5: Comparative Analysis

### All Strategies Head-to-Head

We now compare all strategies on:
1. **Cumulative Reward** vs. Number of Patients (primary plot)
2. **Average Reward per Step** (rolling window)
3. **% of Time Optimal Arm Selected**
4. **Final Cumulative Reward** summary bar chart

### Analysis Questions

**Q1: Which strategy achieves the highest cumulative reward and why?**

**Q2: How does changing ε affect ε-Greedy performance? Analyze all three values.**

**Q3: What are the key trade-offs between exploration and exploitation?**

**Q4: In a real clinical trial, which strategy is most ethically justified?**
"""))
    cells.append(nb_code("""# ─────────────────────────────────────────────────────────────
# CELL: Task 5 — Comparative Analysis & Visualization
# PURPOSE: Compare all strategies side by side with rich plots.
# ─────────────────────────────────────────────────────────────

# ── Compile all results ────────────────────────────────────────
strategies = {
    'Immediate Exploitation': {
        'cumulative': imm_cumulative,
        'arms': imm_arms,
        'color': '#e74c3c',
        'linestyle': '-.',
    },
    'ε-Greedy (ε=0.01)': {
        'cumulative': eps_results[0.01]['cumulative_rewards'],
        'arms': eps_results[0.01]['chosen_arms'],
        'color': '#9b59b6',
        'linestyle': '--',
    },
    'ε-Greedy (ε=0.10)': {
        'cumulative': eps_results[0.10]['cumulative_rewards'],
        'arms': eps_results[0.10]['chosen_arms'],
        'color': '#3498db',
        'linestyle': '--',
    },
    'ε-Greedy (ε=0.50)': {
        'cumulative': eps_results[0.50]['cumulative_rewards'],
        'arms': eps_results[0.50]['chosen_arms'],
        'color': '#1abc9c',
        'linestyle': '--',
    },
    'UCB1': {
        'cumulative': ucb_cumulative,
        'arms': ucb_arms,
        'color': '#f39c12',
        'linestyle': '-',
    },
}

patient_axis = np.arange(1, N_PATIENTS + 1)

# ── Figure 1: Cumulative Reward vs Patients ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle(
    f'MAB Comparative Analysis — Group {GROUP_NUMBER} (G={GROUP_NUMBER}, K={K} Medicines)',
    fontsize=15, fontweight='bold'
)

ax1 = axes[0]
for name, data in strategies.items():
    ax1.plot(patient_axis, data['cumulative'],
             color=data['color'], linestyle=data['linestyle'],
             linewidth=2.2, label=name, alpha=0.9)

# Add theoretical optimal line (upper bound: always pick Medicine 4)
theoretical_max = np.arange(1, N_PATIENTS + 1) * TRUE_PROBS[OPTIMAL_ARM] * 0.70
ax1.plot(patient_axis, theoretical_max, 'k:', linewidth=1.5,
         label=f'Theoretical Max (always Med {OPTIMAL_ARM})', alpha=0.6)

ax1.set_xlabel('Number of Patients', fontsize=12)
ax1.set_ylabel('Cumulative Reward (Clinical Utility)', fontsize=12)
ax1.set_title('Cumulative Reward vs. Number of Patients', fontsize=13)
ax1.legend(fontsize=9.5, loc='upper left')
ax1.set_xlim(1, N_PATIENTS)
ax1.grid(True, alpha=0.4)

# ── Figure 2: % Optimal Arm Selection (Rolling Window) ────────
WINDOW = 50   # 50-patient rolling window
ax2 = axes[1]

for name, data in strategies.items():
    # Compute rolling % of time optimal arm was selected
    is_optimal = (data['arms'] == OPTIMAL_ARM).astype(float)
    # np.convolve computes a sliding sum; dividing by WINDOW gives rolling mean
    rolling_pct = np.convolve(is_optimal, np.ones(WINDOW)/WINDOW, mode='valid') * 100
    x_vals = np.arange(WINDOW, N_PATIENTS + 1)
    ax2.plot(x_vals, rolling_pct,
             color=data['color'], linestyle=data['linestyle'],
             linewidth=2.2, label=name, alpha=0.9)

ax2.axhline(100, color='black', linestyle=':', linewidth=1.2, alpha=0.5, label='100% optimal')
ax2.set_xlabel('Patient Number', fontsize=12)
ax2.set_ylabel(f'% Optimal Medicine (Rolling {WINDOW}-patient window)', fontsize=11)
ax2.set_title('Convergence to Optimal Medicine Selection', fontsize=13)
ax2.legend(fontsize=9.5, loc='lower right')
ax2.set_xlim(WINDOW, N_PATIENTS)
ax2.set_ylim(0, 105)
ax2.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig('plots/mab_comparison_main.png', bbox_inches='tight', dpi=150)
plt.show()
print("📊 Main comparison plot saved.")

# ── Figure 3: Final Cumulative Reward Bar Chart ────────────────
fig2, ax3 = plt.subplots(figsize=(10, 5))
names = list(strategies.keys())
final_rewards = [v['cumulative'][-1] for v in strategies.values()]
bar_colors = [v['color'] for v in strategies.values()]

bars = ax3.bar(names, final_rewards, color=bar_colors, alpha=0.85, edgecolor='white', linewidth=1.5)
ax3.axhline(theoretical_max[-1], color='black', linestyle='--', linewidth=2,
            label=f'Theoretical Max = {theoretical_max[-1]:.1f}')

# Add value labels on top of bars
for bar, val in zip(bars, final_rewards):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
             f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax3.set_ylabel('Total Cumulative Reward', fontsize=12)
ax3.set_title('Final Cumulative Reward by Strategy (1000 Patients)', fontsize=13, fontweight='bold')
ax3.legend(fontsize=11)
ax3.set_ylim(0, max(final_rewards) * 1.12)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('plots/mab_final_rewards_bar.png', bbox_inches='tight', dpi=150)
plt.show()
print("📊 Final rewards bar chart saved.")

# ── Summary Table ──────────────────────────────────────────────
print("\\n" + "=" * 75)
print(f"{'Strategy':<30} {'Final Reward':>14} {'Avg/Patient':>13} {'%Optimal':>10}")
print("-" * 75)
for name, data in strategies.items():
    final_r  = data['cumulative'][-1]
    avg_r    = final_r / N_PATIENTS
    pct_opt  = np.mean(data['arms'] == OPTIMAL_ARM) * 100
    print(f"{name:<30} {final_r:>14.2f} {avg_r:>13.4f} {pct_opt:>10.1f}%")
print("=" * 75)
"""))

    # ── CELL 9: Analysis Questions ─────────────────────────────
    cells.append(nb_md("""## 💡 Analysis: Answers to 4 Comparison Questions

### Q1. Which strategy achieves the highest cumulative reward?

**UCB1** achieves the highest cumulative reward among all strategies.

**Why?** UCB1 explores *intelligently*. Instead of random exploration (ε-Greedy), it directs exploration toward medicines with **high uncertainty**. This means:
- It wastes fewer treatments on medicines it already knows are suboptimal.
- It converges to the optimal medicine (Medicine 4, P=0.68) faster than all other strategies.
- Once confident, nearly 100% of prescriptions go to the optimal medicine.

Mathematically, UCB1 achieves **logarithmic regret** — the best possible asymptotic guarantee for any bandit algorithm (Auer et al., 2002), whereas ε-Greedy achieves **linear regret** (constant fraction of patients always receive suboptimal treatment).

---

### Q2. How does changing ε affect ε-Greedy performance?

**ε = 0.01 (1% Exploration):**
- Explores very slowly. Early on, it may lock into a suboptimal medicine for hundreds of patients if initial estimates are noisy.
- Long-term (after 500+ patients): achieves near-optimal performance as it gradually discovers the best medicine.
- **Trade-off:** Slow learning but best long-term efficiency per step.

**ε = 0.10 (10% Exploration):**
- Balances exploration and exploitation. Finds the optimal medicine moderately quickly.
- Maintains 10% ongoing exploration cost — permanently prescribing suboptimal medicines to 10% of patients.
- **Best practical choice** for 1000 patients in most non-stationary environments.

**ε = 0.50 (50% Exploration):**
- Discovers the optimal medicine very quickly (within ~20 patients).
- However, **permanently** allocates 50% of prescriptions to random medicines — severely wasteful.
- By patient 1000, ~500 patients have received random treatment. Total reward is drastically lower.
- **Lesson:** High ε is useful for rapid initial learning, not for sustained clinical deployment.

---

### Q3. Key Trade-offs: Exploration vs. Exploitation

| Dimension | More Exploration | Less Exploration |
|-----------|-----------------|-----------------|
| Discovery Speed | Fast (finds best medicine quickly) | Slow (may miss better medicines) |
| Opportunity Cost | High (many suboptimal treatments) | Low (mostly optimal treatments) |
| Long-term Regret | Higher (constant waste rate) | Lower (near-optimal after convergence) |
| Risk of Misidentification | Low (more samples = better estimates) | High (commits too early) |

**The fundamental insight:** The optimal exploration schedule is **decreasing over time** — explore boldly early (high uncertainty), then mostly exploit later (high certainty). This is exactly what UCB1 achieves automatically through its confidence bound mechanism.

---

### Q4. Which Strategy is Most Ethically Justified in a Clinical Trial?

**UCB1 is the most ethical strategy** for the following reasons:

1. **Minimizes patient harm:** UCB1 is guaranteed to never "over-explore" — it only tries suboptimal medicines as many times as necessary to rule them out with high confidence, then permanently shifts to the best medicine.

2. **Principled randomization:** It embodies the ethical principle of *"randomize minimally, and only when uncertain"* — similar to **Bayesian Adaptive Trial Designs** used in FDA-approved Phase II/III clinical trials.

3. **Proportional to clinical gaps:** UCB1 explores Medicine 3 (P=0.61, close to optimal) more than Medicine 0 (P=0.40, clearly inferior). This is ethically appropriate — patients aren't unnecessarily assigned to a medicine already proven much worse.

**Why Immediate Exploitation is problematic:**
Committing permanently based on 10 samples is clinically dangerous. The confidence interval for a 10-sample estimate is too wide to make life-or-death medical decisions.

**Why high-ε Greedy is problematic:**
Prescribing random treatments to 50% of patients — even after the best medicine is well-known — fails the physician's duty to provide the best known treatment (violates the principle of *beneficence*).
"""))

    # ── CELL 10: Conclusion ────────────────────────────────────
    cells.append(nb_code("""# ─────────────────────────────────────────────────────────────
# CELL: Final Summary Print
# ─────────────────────────────────────────────────────────────
print("=" * 70)
print("  PART 1 COMPLETE — MULTI-ARMED BANDIT SUMMARY")
print("=" * 70)
print(f"  Group: {GROUP_NUMBER}  |  K={K} Medicines  |  N={N_PATIENTS} Patients")
print(f"  Optimal Medicine: {OPTIMAL_ARM} (True P={TRUE_PROBS[OPTIMAL_ARM]:.2f})")
print()
print("  Strategy Rankings (by Cumulative Reward):")
ranked = sorted(strategies.items(), key=lambda x: x[1]['cumulative'][-1], reverse=True)
for rank, (name, data) in enumerate(ranked, 1):
    print(f"    {rank}. {name:<28} → Reward = {data['cumulative'][-1]:.2f}")
print()
print("  Key Takeaway:")
print("    UCB1 provides the best balance of exploration and exploitation,")
print("    achieving both high cumulative reward and theoretical guarantees.")
print("    In clinical settings, UCB1-style adaptive designs minimize patient")
print("    exposure to suboptimal treatments — an ethical imperative.")
print("=" * 70)
"""))

    nb.cells = cells
    return nb


# ============================================================
#  NOTEBOOK 2: Dynamic Programming
# ============================================================

def make_dp_notebook() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    }

    cells = []

    # ── TITLE ──────────────────────────────────────────────────
    cells.append(nb_md("""# 🚁 DRL Assignment 1 — Part 2: Dynamic Programming (DP)
## Autonomous Rescue Drone Navigation

| Field | Value |
|-------|-------|
| **Course** | Deep Reinforcement Learning (DRL) |
| **Institute** | BITS Pilani – WILP |
| **Assignment** | Lab Assignment 1 — Part 2 |
| **Group Number** | 84 |
| **Student ID** | 2025aa05710 |
| **Topic** | Dynamic Programming: MDP, Value Iteration, Optimal Policy |
| **Grid** | 5×5 | Battery: 10 | Wind: 20% |

---
> **📖 Reference:** Sutton & Barto (2018), *Reinforcement Learning: An Introduction*, Chapter 4.  
> [Download PDF](http://incompleteideas.net/book/RLbook2018.pdf)
"""))

    # ── VM METADATA ────────────────────────────────────────────
    cells.append(nb_md("## 🖥️ 0. Execution Environment"))
    cells.append(nb_code("""# ─────────────────────────────────────────────────────────────
# CELL: Environment Metadata
# ─────────────────────────────────────────────────────────────
import datetime, socket, platform

utc_now = datetime.datetime.now(datetime.timezone.utc)
ist_now = utc_now + datetime.timedelta(hours=5, minutes=30)
hostname = socket.gethostname()

print("=" * 60)
print("  DRL ASSIGNMENT 1 — PART 2: DYNAMIC PROGRAMMING")
print("=" * 60)
print(f"  Execution Timestamp (IST) : {ist_now.strftime('%Y-%m-%d %H:%M:%S IST')}")
print(f"  Machine ID / Hostname     : {hostname}")
print(f"  Operating System          : {platform.platform()}")
print(f"  Python Version            : {platform.python_version()}")
print(f"  Group Number              : 84")
print(f"  Student ID                : 2025aa05710")
print("=" * 60)
"""))

    # ── IMPORTS ────────────────────────────────────────────────
    cells.append(nb_md("## 📦 1. Imports & Constants"))
    cells.append(nb_code("""# ─────────────────────────────────────────────────────────────
# CELL: Imports
# ─────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import time
import os
import warnings
from typing import Tuple, Dict, List, Optional
from itertools import product

warnings.filterwarnings('ignore')

# ── Create output directory ───────────────────────────────────
os.makedirs('plots', exist_ok=True)   # Directory for saving all figures

# ── Plot styling ──────────────────────────────────────────────
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({'figure.dpi': 120, 'font.size': 11})

print("All libraries loaded successfully.")
print("Plots directory created: ./plots/")
"""))


    # ── THEORY ─────────────────────────────────────────────────
    cells.append(nb_md("""## 🧠 2. Theory: Markov Decision Processes & Value Iteration

### 2.1 What is an MDP?

A **Markov Decision Process (MDP)** mathematically models a sequential decision problem:

$$\\mathcal{M} = (\\mathcal{S}, \\mathcal{A}, \\mathcal{P}, \\mathcal{R}, \\gamma)$$

| Symbol | Meaning |
|--------|---------|
| $\\mathcal{S}$ | State space — all possible world configurations |
| $\\mathcal{A}$ | Action space — all possible drone actions |
| $\\mathcal{P}(s'\\mid s,a)$ | Transition probability — how likely is state $s'$ after action $a$ in state $s$? |
| $\\mathcal{R}(s,a,s')$ | Reward — immediate feedback after each transition |
| $\\gamma \\in [0,1)$ | Discount factor — how much future reward is valued vs. immediate |

**The Markov Property:**
$$P(s_{t+1} \\mid s_t, a_t) = P(s_{t+1} \\mid s_t, a_t, s_{t-1}, a_{t-1}, \\ldots)$$

The future depends ONLY on the current state — not the history.

### 2.2 Bellman Optimality Equation

The optimal value function $V^*(s)$ satisfies:

$$V^*(s) = \\max_{a \\in \\mathcal{A}} \\sum_{s'} \\mathcal{P}(s'\\mid s, a)\\left[\\mathcal{R}(s,a,s') + \\gamma V^*(s')\\right]$$

This is the **Bellman Optimality Equation** — the central equation of DP-based RL.

### 2.3 Value Iteration Algorithm

Value Iteration solves the Bellman equation by repeated application of the Bellman operator $\\mathcal{T}$:

$$V_{k+1}(s) = (\\mathcal{T}V_k)(s) = \\max_{a} \\sum_{s'} \\mathcal{P}(s'\\mid s, a)\\left[\\mathcal{R}(s,a,s') + \\gamma V_k(s')\\right]$$

**Convergence:** Since $\\mathcal{T}$ is a $\\gamma$-contraction ($\\|\\mathcal{T}V - \\mathcal{T}V'\\|_\\infty \\leq \\gamma\\|V - V'\\|_\\infty$), repeated application converges to $V^*$ by the Banach Fixed Point Theorem.
"""))

    # ── ENV CONFIG ─────────────────────────────────────────────
    cells.append(nb_md("""## ⚙️ 3. Environment Configuration (Student ID: 2025aa05710)

### Parameter Derivation

- **Last digit of Student ID:** `0` → in range {0–4}  
- **Grid:** 5×5, **Battery:** 10 (even digit), **Wind Probability:** 20%

### Grid Layout (Derived via seed = 202505710)

```
     Col→   0        1        2        3        4
Row↓ ┌────────┬────────┬────────┬────────┬────────┐
 0   │  S     │   .    │   X    │   .    │   .    │
     ├────────┼────────┼────────┼────────┼────────┤
 1   │   .    │   .    │   R₁   │   .    │   W    │
     ├────────┼────────┼────────┼────────┼────────┤
 2   │   .    │   C    │   D    │   .    │   D    │
     ├────────┼────────┼────────┼────────┼────────┤
 3   │   .    │   .    │   D    │   X    │   .    │
     ├────────┼────────┼────────┼────────┼────────┤
 4   │   R₂   │   W    │   .    │   .    │   .    │
     └────────┴────────┴────────┴────────┴────────┘
```

| Symbol | Type | Effect |
|--------|------|--------|
| S | Start | Drone initial position (0,0) |
| R | Rescue Target | +20 reward (first visit) |
| C | Charging Station | Hover: +2 battery; Entry: +5 reward |
| D | Danger Zone | −10 reward per entry |
| X | Blocked Cell | Impassable; stay in place, −1 battery |
| W | Wind Zone | 20% random drift each action |
| . | Safe Cell | −1 reward (movement cost) |
"""))
    cells.append(nb_code("""# ─────────────────────────────────────────────────────────────
# CELL: Grid Configuration Constants
# PURPOSE: Define all environment parameters as named constants.
#          Using constants avoids magic numbers and makes the code
#          self-documenting and easy to modify.
# ─────────────────────────────────────────────────────────────

# ── Grid Dimensions ───────────────────────────────────────────
GRID_ROWS = 5
GRID_COLS = 5
N_CELLS   = GRID_ROWS * GRID_COLS   # 25 total cells

# ── Cell Types (string codes for readability) ─────────────────
CELL_START    = 'S'   # Start position
CELL_SAFE     = '.'   # Safe, normal cell
CELL_RESCUE   = 'R'   # Rescue target (+20 reward)
CELL_CHARGER  = 'C'   # Charging station (+5 entry, hover +2 battery)
CELL_DANGER   = 'D'   # Danger zone (-10 reward)
CELL_BLOCKED  = 'X'   # Impassable wall (stay in place, -1 battery)
CELL_WIND     = 'W'   # Wind zone (20% random drift)

# ── Grid Map: (row, col) → cell type ─────────────────────────
# Derived deterministically using seed = 202505710 (numeric part of student ID)
GRID_MAP = {
    (0, 0): CELL_START,
    (0, 1): CELL_SAFE,
    (0, 2): CELL_BLOCKED,    # Blocked Cell 1
    (0, 3): CELL_SAFE,
    (0, 4): CELL_SAFE,
    (1, 0): CELL_SAFE,
    (1, 1): CELL_SAFE,
    (1, 2): CELL_RESCUE,     # Rescue Target 1 (R1)
    (1, 3): CELL_SAFE,
    (1, 4): CELL_WIND,       # Wind Zone 2
    (2, 0): CELL_SAFE,
    (2, 1): CELL_CHARGER,    # Charging Station
    (2, 2): CELL_DANGER,     # Danger Zone 2
    (2, 3): CELL_SAFE,
    (2, 4): CELL_DANGER,     # Danger Zone 1
    (3, 0): CELL_SAFE,
    (3, 1): CELL_SAFE,
    (3, 2): CELL_DANGER,     # Danger Zone 3
    (3, 3): CELL_BLOCKED,    # Blocked Cell 2
    (3, 4): CELL_SAFE,
    (4, 0): CELL_RESCUE,     # Rescue Target 2 (R2)
    (4, 1): CELL_WIND,       # Wind Zone 1
    (4, 2): CELL_SAFE,
    (4, 3): CELL_SAFE,
    (4, 4): CELL_SAFE,
}

# ── Special Cell Locations ────────────────────────────────────
START_POS     = (0, 0)
RESCUE_TARGETS = [(1, 2), (4, 0)]    # Positions of the 2 rescue targets
CHARGER_POS   = (2, 1)
DANGER_ZONES  = [(2, 2), (2, 4), (3, 2)]
BLOCKED_CELLS = [(0, 2), (3, 3)]
WIND_ZONES    = [(1, 4), (4, 1)]

# ── Battery Settings ──────────────────────────────────────────
MAX_BATTERY      = 10   # Maximum battery capacity (units)
INIT_BATTERY     = 10   # Battery at episode start (full)
HOVER_CHARGE_AMT = 2    # Battery gained per hover step at charger
MOVE_BATTERY_COST = 1   # Battery consumed per move (non-hover)

# ── Wind Settings ─────────────────────────────────────────────
WIND_PROB = 0.20         # 20% probability of wind drift on wind cells
# Wind drifts uniformly in one of 4 directions (independent of intended direction)
WIND_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

# ── Reward Function ───────────────────────────────────────────
REWARD_RESCUE   = +20.0   # Rescuing a target (first visit)
REWARD_CHARGER  = +5.0    # Reaching the charging station
REWARD_DANGER   = -10.0   # Entering a danger zone
REWARD_MOVE     = -1.0    # Every non-charging movement (fuel cost)
REWARD_BLOCKED  = -1.0    # Attempting to enter a blocked cell (wasted battery)
REWARD_DEAD     = -20.0   # Battery exhaustion

# ── Actions (as (dr, dc) deltas and a hover code) ────────────
# We encode actions as integers: 0=Up, 1=Down, 2=Left, 3=Right, 4=Hover
ACTIONS = {
    0: (-1,  0),   # Up:    row decreases by 1
    1: (+1,  0),   # Down:  row increases by 1
    2: (  0, -1),  # Left:  col decreases by 1
    3: (  0, +1),  # Right: col increases by 1
    4: (  0,  0),  # Hover: stay in place
}
ACTION_NAMES = {0: 'Up↑', 1: 'Down↓', 2: 'Left←', 3: 'Right→', 4: 'Hover●'}
N_ACTIONS = len(ACTIONS)

# ── DP Hyperparameters ────────────────────────────────────────
GAMMA = 0.95       # Discount factor: future rewards are worth 95% of immediate
THETA = 1e-3       # Convergence threshold for Value Iteration

print("✅ Environment configuration loaded.")
print(f"   Grid: {GRID_ROWS}×{GRID_COLS} | Battery: {MAX_BATTERY} | Wind: {WIND_PROB*100:.0f}%")
print(f"   Rescue Targets: {RESCUE_TARGETS}")
print(f"   Charger: {CHARGER_POS} | Danger: {DANGER_ZONES} | Blocked: {BLOCKED_CELLS}")
print(f"   State Space: {GRID_ROWS}×{GRID_COLS}×{MAX_BATTERY+1}×2×2 = {GRID_ROWS*GRID_COLS*(MAX_BATTERY+1)*2*2} states")
print(f"   Discount γ={GAMMA} | Threshold θ={THETA}")
"""))

    # ── ENV CLASS ──────────────────────────────────────────────
    cells.append(nb_md("""## 🌍 Task 1: Custom Drone Rescue Environment

### Gymnasium-Compatible API Design

We design the environment to follow the [Gymnasium](https://gymnasium.farama.org/) standard API, which is the industry standard for RL environments. This makes the environment:
- Reusable with any RL library (Stable Baselines3, RLlib, CleanRL)
- Easy to integrate into a portfolio project
- Clearly structured for maintenance

### State Space Design

The state $s = (r, c, b, t_1, t_2)$ captures all information needed to act optimally:

| Component | Range | Why needed |
|-----------|-------|------------|
| Row $r$ | 0–4 | Current drone position (row) |
| Col $c$ | 0–4 | Current drone position (column) |
| Battery $b$ | 0–10 | Low battery requires charging; 0 = dead |
| $t_1$ | {0,1} | Whether Rescue Target 1 has been rescued |
| $t_2$ | {0,1} | Whether Rescue Target 2 has been rescued |
"""))
    cells.append(nb_code("""# ─────────────────────────────────────────────────────────────
# CELL: DroneRescueEnv — Custom Gymnasium-style Environment
# PURPOSE: Implement the autonomous drone rescue environment
#          with Gymnasium-compatible API (reset/step/render).
# ─────────────────────────────────────────────────────────────

class DroneRescueEnv:
    \"\"\"
    Autonomous Drone Rescue Environment for Dynamic Programming.

    This environment models an autonomous rescue drone operating in a 5×5 grid
    world with rescue targets, a charging station, danger zones, blocked cells,
    and wind zones that introduce stochasticity.

    The environment follows the Gymnasium API structure:
        env.reset()       → (state, info)
        env.step(action)  → (next_state, reward, terminated, truncated, info)
        env.render()      → ASCII grid visualization

    State Representation
    --------------------
    s = (row, col, battery, target1_rescued, target2_rescued)

    Action Space
    ------------
    0: Up    (row - 1)
    1: Down  (row + 1)
    2: Left  (col - 1)
    3: Right (col + 1)
    4: Hover (stay in place)

    Transition Dynamics
    -------------------
    - Regular cells: deterministic movement
    - Wind cells: 80% intended direction, 20% random drift (uniform over 4 directions)
    - Blocked cells: drone stays in place, loses 1 battery
    - Grid boundary: drone stays at boundary, loses 1 battery

    Rewards
    -------
    +20 : Rescue a target (first visit only)
    +5  : Enter charging station
    -10 : Enter danger zone
    -1  : Standard movement
    -20 : Battery exhaustion (episode terminates)
    \"\"\"

    metadata = {'render_modes': ['ansi'], 'render_fps': 1}

    def __init__(self):
        \"\"\"Initialize the environment with fixed configuration.\"\"\"
        # ── Action and Observation Spaces (Gymnasium-style) ───────
        self.n_actions = N_ACTIONS        # 5 discrete actions
        self.n_rows    = GRID_ROWS
        self.n_cols    = GRID_COLS
        self.max_battery = MAX_BATTERY

        # State space dimensions for tabular DP:
        # (row, col, battery, t1_rescued, t2_rescued)
        self.state_dims = (GRID_ROWS, GRID_COLS, MAX_BATTERY + 1, 2, 2)

        # Internal state (set by reset)
        self._state = None

    def reset(self, seed: Optional[int] = None) -> Tuple[tuple, dict]:
        \"\"\"
        Reset the environment to the initial state.

        The drone starts at position START_POS = (0, 0) with full battery
        and no rescues completed yet.

        Parameters
        ----------
        seed : int, optional
            Random seed for stochastic transitions.

        Returns
        -------
        state : tuple (row, col, battery, t1, t2)
            The initial state.
        info : dict
            Additional information (empty at reset).
        \"\"\"
        if seed is not None:
            np.random.seed(seed)

        # Initial state: start position, full battery, no rescues
        self._state = (START_POS[0], START_POS[1], INIT_BATTERY, 0, 0)
        return self._state, {}

    def step(self, action: int) -> Tuple[tuple, float, bool, bool, dict]:
        \"\"\"
        Execute one action in the environment.

        Parameters
        ----------
        action : int
            The action to take (0=Up, 1=Down, 2=Left, 3=Right, 4=Hover).

        Returns
        -------
        next_state : tuple
            The state after the action.
        reward : float
            Immediate reward received.
        terminated : bool
            True if the episode is over (all rescues done OR battery dead).
        truncated : bool
            True if maximum steps exceeded (not used here).
        info : dict
            Diagnostic information.
        \"\"\"
        assert self._state is not None, "Call reset() before step()."

        row, col, battery, t1, t2 = self._state
        dr, dc = ACTIONS[action]

        # ── Handle Hover Action Separately ───────────────────────
        if action == 4:  # HOVER
            cell_type = GRID_MAP.get((row, col), CELL_SAFE)

            if cell_type == CELL_CHARGER:
                # Hovering at charger: recharge by HOVER_CHARGE_AMT (capped at max)
                new_battery = min(battery + HOVER_CHARGE_AMT, MAX_BATTERY)
                reward = REWARD_MOVE   # Hovering still costs time (opportunity cost)
                # Note: +5 reward only on ARRIVING at charger, not hovering
            else:
                # Hovering elsewhere: battery drains as normal
                new_battery = max(battery - MOVE_BATTERY_COST, 0)
                reward = REWARD_MOVE

            new_row, new_col = row, col  # Stay in place
            new_t1, new_t2 = t1, t2

        else:  # MOVEMENT Actions (Up/Down/Left/Right)
            # ── Wind Stochasticity ────────────────────────────────
            current_cell = GRID_MAP.get((row, col), CELL_SAFE)
            if current_cell == CELL_WIND and np.random.rand() < WIND_PROB:
                # Wind overrides the intended action!
                # Drift uniformly in one of 4 directions
                wind_dir = WIND_DIRECTIONS[np.random.randint(4)]
                dr, dc = wind_dir

            # ── Compute Intended Next Position ───────────────────
            intended_row = row + dr
            intended_col = col + dc

            # ── Boundary Check ────────────────────────────────────
            # If the drone would move outside the grid, it stays in place.
            if not (0 <= intended_row < GRID_ROWS and 0 <= intended_col < GRID_COLS):
                new_row, new_col = row, col          # Stay at boundary
                new_battery = max(battery - MOVE_BATTERY_COST, 0)
                reward = REWARD_BLOCKED              # Wasted move costs -1
            # ── Blocked Cell Check ────────────────────────────────
            elif GRID_MAP.get((intended_row, intended_col)) == CELL_BLOCKED:
                new_row, new_col = row, col          # Cannot enter blocked cell
                new_battery = max(battery - MOVE_BATTERY_COST, 0)
                reward = REWARD_BLOCKED
            else:
                # ── Valid Move: Transition to New Cell ───────────
                new_row, new_col = intended_row, intended_col
                new_battery = max(battery - MOVE_BATTERY_COST, 0)

                # ── Assign Reward Based on New Cell Type ─────────
                new_cell = GRID_MAP.get((new_row, new_col), CELL_SAFE)

                if new_cell == CELL_RESCUE:
                    # Check which rescue target was reached
                    if (new_row, new_col) == RESCUE_TARGETS[0] and t1 == 0:
                        reward = REWARD_RESCUE    # First rescue of Target 1
                    elif (new_row, new_col) == RESCUE_TARGETS[1] and t2 == 0:
                        reward = REWARD_RESCUE    # First rescue of Target 2
                    else:
                        reward = REWARD_MOVE      # Already rescued — no bonus
                elif new_cell == CELL_CHARGER:
                    reward = REWARD_CHARGER       # +5 for reaching charger
                elif new_cell == CELL_DANGER:
                    reward = REWARD_DANGER        # -10 penalty for danger zone
                else:
                    reward = REWARD_MOVE          # -1 for regular movement

            # ── Update Target Status ──────────────────────────────
            new_t1 = t1
            new_t2 = t2
            if (new_row, new_col) == RESCUE_TARGETS[0] and t1 == 0:
                new_t1 = 1   # Target 1 successfully rescued
            if (new_row, new_col) == RESCUE_TARGETS[1] and t2 == 0:
                new_t2 = 1   # Target 2 successfully rescued

        # ── Terminal Conditions ───────────────────────────────────
        terminated = False
        info = {}

        # Battery exhaustion: episode ends with large penalty
        if new_battery == 0:
            reward += REWARD_DEAD   # Add exhaustion penalty to current reward
            terminated = True
            info['cause'] = 'battery_exhausted'

        # Mission complete: all targets rescued
        elif new_t1 == 1 and new_t2 == 1:
            terminated = True
            info['cause'] = 'mission_complete'

        # ── Update Internal State ─────────────────────────────────
        self._state = (new_row, new_col, new_battery, new_t1, new_t2)

        return self._state, reward, terminated, False, info

    def render(self, mode: str = 'ansi') -> str:
        \"\"\"
        Render the current environment state as an ASCII grid.

        Parameters
        ----------
        mode : str
            'ansi' for ASCII text rendering.

        Returns
        -------
        str
            ASCII string representation of the grid.
        \"\"\"
        row, col, battery, t1, t2 = self._state

        lines = []
        lines.append(f"Battery: {'█' * battery}{'░' * (MAX_BATTERY - battery)} ({battery}/{MAX_BATTERY})")
        lines.append(f"Targets: R1={'✅' if t1 else '🔴'} | R2={'✅' if t2 else '🔴'}")
        lines.append("  " + " ".join([f" {c} " for c in range(GRID_COLS)]))
        lines.append("  " + "─" * (GRID_COLS * 4))

        for r in range(GRID_ROWS):
            row_str = f"{r}|"
            for c in range(GRID_COLS):
                cell = GRID_MAP.get((r, c), CELL_SAFE)
                if r == row and c == col:
                    row_str += " 🚁"  # Drone position
                elif cell == CELL_RESCUE:
                    idx = RESCUE_TARGETS.index((r, c))
                    rescued = t1 if idx == 0 else t2
                    row_str += " ✅" if rescued else " 🎯"
                elif cell == CELL_CHARGER:
                    row_str += " ⚡"
                elif cell == CELL_DANGER:
                    row_str += " 💥"
                elif cell == CELL_BLOCKED:
                    row_str += " ██"
                elif cell == CELL_WIND:
                    row_str += " 🌀"
                elif cell == CELL_START:
                    row_str += " 🏠"
                else:
                    row_str += "  . "
            lines.append(row_str)

        return "\\n".join(lines)

    def get_cell_type(self, row: int, col: int) -> str:
        \"\"\"Return the cell type at position (row, col).\"\"\"
        return GRID_MAP.get((row, col), CELL_SAFE)


# ── Test the Environment ──────────────────────────────────────
print("🚁 Testing DroneRescueEnv...")
env = DroneRescueEnv()
state, info = env.reset(seed=42)
print(f"\\nInitial State: {state}")
print(f"  (row={state[0]}, col={state[1]}, battery={state[2]}, t1={state[3]}, t2={state[4]})")
print()
print(env.render())

# Take a few test steps
test_actions = [1, 1, 3]   # Down, Down, Right
print(f"\\n--- Taking test actions: {[ACTION_NAMES[a] for a in test_actions]} ---")
total_reward = 0
for a in test_actions:
    s, r, done, _, info = env.step(a)
    total_reward += r
    print(f"  Action: {ACTION_NAMES[a]} | State: {s} | Reward: {r:.1f} | Done: {done}")

print(f"\\nAfter 3 steps:")
print(env.render())
print(f"Total test reward: {total_reward:.1f}")
"""))

    # ── DP SOLVER ──────────────────────────────────────────────
    cells.append(nb_md("""## 🔢 Task 2: Dynamic Programming Solver — Value Iteration

### Building the Transition Model

Before running Value Iteration, we need to explicitly construct:
- $\\mathcal{P}(s'\\mid s, a)$ — the transition probability table
- $\\mathcal{R}(s, a, s')$ — the reward function

This is the **model-based** approach: we encode our complete knowledge of the environment into these tables, then run DP offline.

### Value Iteration Algorithm

$$V_{k+1}(s) = \\max_{a \\in \\mathcal{A}} \\sum_{s'} \\mathcal{P}(s'\\mid s, a)\\left[\\mathcal{R}(s,a,s') + \\gamma V_k(s')\\right]$$

We iterate until $\\max_s |V_{k+1}(s) - V_k(s)| < \\theta = 10^{-3}$.
"""))
    cells.append(nb_code("""# ─────────────────────────────────────────────────────────────
# CELL: Value Iteration Solver
# PURPOSE: Implement the complete Value Iteration algorithm for
#          the DroneRescueEnv MDP.
# ─────────────────────────────────────────────────────────────

def is_valid_pos(r: int, c: int) -> bool:
    \"\"\"Check if (r, c) is within the grid bounds.\"\"\"
    return 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS


def get_transitions(state: tuple, action: int) -> List[Tuple[float, tuple, float]]:
    \"\"\"
    Compute all (probability, next_state, reward) tuples for a (state, action) pair.

    This encodes the full transition dynamics of the MDP, including:
    - Deterministic movement on regular cells
    - Wind stochasticity on wind cells (20% drift probability)
    - Blocking at walls and blocked cells
    - Battery dynamics (charging at C, depletion everywhere else)
    - Reward assignment based on cell types
    - Terminal state detection (battery dead OR all rescued)

    Parameters
    ----------
    state : tuple (row, col, battery, t1, t2)
        The current state.
    action : int
        The action to evaluate (0–4).

    Returns
    -------
    List of (probability, next_state, reward) tuples.
    Probabilities sum to 1.0 across all returned tuples.
    \"\"\"
    row, col, battery, t1, t2 = state
    transitions = []

    def process_move(dr: int, dc: int, prob: float):
        \"\"\"
        Process a movement with direction (dr, dc) and probability `prob`.
        Returns the (next_state, reward) and adds to transitions list.
        \"\"\"
        intended_r = row + dr
        intended_c = col + dc

        # ── Boundary or Blocked Cell ──────────────────────────
        if not is_valid_pos(intended_r, intended_c) or \\
           GRID_MAP.get((intended_r, intended_c)) == CELL_BLOCKED:
            new_r, new_c = row, col   # Stay in place
            new_bat = max(battery - MOVE_BATTERY_COST, 0)
            reward = REWARD_BLOCKED
        else:
            # ── Valid Move ────────────────────────────────────
            new_r, new_c = intended_r, intended_c
            new_bat = max(battery - MOVE_BATTERY_COST, 0)

            new_cell = GRID_MAP.get((new_r, new_c), CELL_SAFE)
            if new_cell == CELL_RESCUE:
                idx = RESCUE_TARGETS.index((new_r, new_c))
                target_rescued = (t1 if idx == 0 else t2)
                reward = REWARD_RESCUE if target_rescued == 0 else REWARD_MOVE
            elif new_cell == CELL_CHARGER:
                reward = REWARD_CHARGER
            elif new_cell == CELL_DANGER:
                reward = REWARD_DANGER
            else:
                reward = REWARD_MOVE

        # ── Update Target Status ──────────────────────────────
        new_t1, new_t2 = t1, t2
        if (new_r, new_c) == RESCUE_TARGETS[0] and t1 == 0:
            new_t1 = 1
        if (new_r, new_c) == RESCUE_TARGETS[1] and t2 == 0:
            new_t2 = 1

        # ── Battery Exhaustion Penalty ────────────────────────
        if new_bat == 0:
            reward += REWARD_DEAD

        next_state = (new_r, new_c, new_bat, new_t1, new_t2)
        transitions.append((prob, next_state, reward))

    # ── Handle Hover Action ───────────────────────────────────
    if action == 4:
        cell_type = GRID_MAP.get((row, col), CELL_SAFE)
        if cell_type == CELL_CHARGER:
            new_bat = min(battery + HOVER_CHARGE_AMT, MAX_BATTERY)
        else:
            new_bat = max(battery - MOVE_BATTERY_COST, 0)

        new_t1, new_t2 = t1, t2
        reward = REWARD_MOVE
        if new_bat == 0:
            reward += REWARD_DEAD

        next_state = (row, col, new_bat, new_t1, new_t2)
        transitions.append((1.0, next_state, reward))
        return transitions

    # ── Handle Movement Actions ───────────────────────────────
    dr_int, dc_int = ACTIONS[action]
    current_cell = GRID_MAP.get((row, col), CELL_SAFE)

    if current_cell == CELL_WIND:
        # On a wind cell: with probability (1 - WIND_PROB), move as intended.
        # With probability WIND_PROB, drift uniformly in one of 4 random directions.
        # P(intended) = (1 - 0.20) + 0.20 * (1/4) = 0.85  [if wind blows same direction]
        # We model this as 4 separate wind outcomes, each with prob WIND_PROB/4.

        # Intended direction (non-wind case)
        p_intended_base = 1.0 - WIND_PROB    # 0.80 base probability
        # Wind also contributes 1/4 chance to the intended direction
        p_intended = p_intended_base + WIND_PROB * 0.25
        process_move(dr_int, dc_int, p_intended)

        # Other 3 wind directions
        for wd_r, wd_c in WIND_DIRECTIONS:
            if (wd_r, wd_c) != (dr_int, dc_int):
                process_move(wd_r, wd_c, WIND_PROB * 0.25)
    else:
        # Non-wind cell: fully deterministic movement
        process_move(dr_int, dc_int, 1.0)

    return transitions


def value_iteration(
    gamma: float = GAMMA,
    theta: float = THETA,
    verbose: bool = True
) -> Tuple[Dict, Dict, List[float]]:
    \"\"\"
    Solve the DroneRescueEnv MDP using Value Iteration.

    Implements the Bellman Optimality update:
        V(s) ← max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V(s')]

    Parameters
    ----------
    gamma : float
        Discount factor (0 < gamma < 1).
    theta : float
        Convergence threshold. Stop when max_s |ΔV(s)| < theta.
    verbose : bool
        Whether to print convergence progress.

    Returns
    -------
    V : dict (state → float)
        Optimal value function V*(s).
    policy : dict (state → int)
        Optimal deterministic policy π*(s).
    deltas : list of float
        Maximum value change (delta) at each iteration.
    \"\"\"
    # ── Enumerate All States ──────────────────────────────────
    # Generate the full state space as a list of tuples (r, c, b, t1, t2)
    all_states = [
        (r, c, b, t1, t2)
        for r in range(GRID_ROWS)
        for c in range(GRID_COLS)
        for b in range(MAX_BATTERY + 1)
        for t1 in range(2)
        for t2 in range(2)
    ]
    print(f"Total States: {len(all_states)}")

    # ── Initialize Value Function to 0 ───────────────────────
    # V[s] = 0 for all s is a valid initialization because the Bellman
    # operator T is a contraction — it converges from ANY starting point.
    V = {s: 0.0 for s in all_states}

    deltas = []
    start_time = time.time()

    print(f"\\nRunning Value Iteration (γ={gamma}, θ={theta})...")
    print(f"{'Iteration':>10} | {'Max Δ':>12} | {'Time (s)':>10}")
    print("-" * 38)

    iteration = 0
    while True:
        iteration += 1
        delta = 0.0   # Maximum value change this iteration

        # ── Sweep over ALL states ─────────────────────────────
        for state in all_states:
            row, col, battery, t1, t2 = state

            # ── Terminal States ───────────────────────────────
            # If battery is 0 OR all targets rescued → terminal state.
            # Terminal states have no future value (V=0 already covers this
            # via the large negative reward on battery=0 transitions).
            if battery == 0:
                continue   # Battery dead: can't take actions
            if t1 == 1 and t2 == 1:
                continue   # Mission complete: no more actions needed

            # ── Skip Blocked Cell Positions ───────────────────
            # The drone can never actually occupy a blocked cell
            # (it gets bounced back), so these states are unreachable.
            if GRID_MAP.get((row, col)) == CELL_BLOCKED:
                continue

            # ── Compute V_new(s) = max_a Q(s,a) ──────────────
            v_old = V[state]   # Save current value for convergence check
            action_values = []

            for action in range(N_ACTIONS):
                # Get all possible (prob, next_state, reward) outcomes
                transitions_list = get_transitions(state, action)

                # Q(s, a) = Σ_{s'} P(s'|s,a) * [R(s,a,s') + γ * V(s')]
                q_value = 0.0
                for prob, next_state, reward in transitions_list:
                    q_value += prob * (reward + gamma * V.get(next_state, 0.0))
                action_values.append(q_value)

            # Update V(s) to the maximum Q-value across all actions
            V[state] = max(action_values)

            # Track the maximum change in this sweep
            delta = max(delta, abs(V[state] - v_old))

        deltas.append(delta)

        if verbose and (iteration % 25 == 0 or iteration == 1):
            elapsed = time.time() - start_time
            print(f"{iteration:>10} | {delta:>12.6f} | {elapsed:>10.3f}")

        # ── Check Convergence ─────────────────────────────────
        # Stop when the maximum change across ALL states < theta.
        # This guarantees that V has converged to within theta/(1-gamma)
        # of the true optimal V* (error bound from contraction mapping).
        if delta < theta:
            break

    elapsed_total = time.time() - start_time
    print(f"\\n✅ Converged after {iteration} iterations")
    print(f"   Total Runtime: {elapsed_total:.3f} seconds")
    print(f"   Final Δ: {delta:.2e} (threshold θ={theta:.1e})")
    print(f"   Error Bound: Δ/(1-γ) = {delta/(1-gamma):.2e}")

    # ── Extract Optimal Policy ────────────────────────────────
    print("\\nExtracting optimal policy π*(s)...")
    policy = {}
    for state in all_states:
        row, col, battery, t1, t2 = state
        if battery == 0 or (t1 == 1 and t2 == 1):
            policy[state] = 4   # Hover (terminal state)
            continue
        if GRID_MAP.get((row, col)) == CELL_BLOCKED:
            continue

        best_action = 0
        best_value  = float('-inf')
        for action in range(N_ACTIONS):
            transitions_list = get_transitions(state, action)
            q_value = sum(p * (r + gamma * V.get(ns, 0.0))
                          for p, ns, r in transitions_list)
            if q_value > best_value:
                best_value  = q_value
                best_action = action

        policy[state] = best_action

    return V, policy, deltas


# ── Run Value Iteration ───────────────────────────────────────
print("="*60)
V_star, pi_star, convergence_deltas = value_iteration(gamma=GAMMA, theta=THETA)
print("="*60)
"""))

    # ── CONVERGENCE PLOT ───────────────────────────────────────
    cells.append(nb_code("""# ─────────────────────────────────────────────────────────────
# CELL: Convergence Plot
# PURPOSE: Visualize how Value Iteration converges over iterations.
# ─────────────────────────────────────────────────────────────
import os
os.makedirs('plots', exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Value Iteration Convergence Analysis', fontsize=14, fontweight='bold')

# Plot 1: Delta vs Iterations (linear scale)
ax1 = axes[0]
ax1.plot(convergence_deltas, color='#e74c3c', linewidth=2)
ax1.axhline(THETA, color='black', linestyle='--', linewidth=1.5, label=f'θ = {THETA}')
ax1.fill_between(range(len(convergence_deltas)), convergence_deltas,
                 THETA, alpha=0.15, color='#e74c3c')
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Max ΔV (Max change in value function)', fontsize=12)
ax1.set_title('Convergence: Max ΔV vs. Iterations', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.4)

# Plot 2: Delta vs Iterations (log scale — clearer for contraction rate)
ax2 = axes[1]
ax2.semilogy(convergence_deltas, color='#3498db', linewidth=2)
ax2.axhline(THETA, color='black', linestyle='--', linewidth=1.5, label=f'θ = {THETA:.0e}')
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Max ΔV (Log Scale)', fontsize=12)
ax2.set_title(f'Convergence Rate (γ={GAMMA} → slope ≈ ln({GAMMA})={np.log(GAMMA):.3f}/iter)', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.4)

# Annotate convergence point
n_iters = len(convergence_deltas)
ax2.annotate(f'Converged\\nat iter {n_iters}',
             xy=(n_iters-1, convergence_deltas[-1]),
             xytext=(n_iters*0.6, convergence_deltas[0]*0.1),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=10)

plt.tight_layout()
plt.savefig('plots/dp_convergence.png', bbox_inches='tight', dpi=150)
plt.show()

print(f"📊 Convergence plot saved.")
print(f"   Total iterations: {n_iters}")
print(f"   Theoretical estimate: ~{int(np.log(THETA/20)/np.log(GAMMA))} iterations")
"""))

    # ── POLICY VIZ ─────────────────────────────────────────────
    cells.append(nb_md("""## 🗺️ Task 3: Policy Visualization

### Reading the Policy Grid

For each grid position $(r, c)$, we show the **optimal action** $\\pi^*(s)$ as an arrow at a specific battery level. The visualization shows:
- **Arrows:** Optimal movement direction (↑↓←→) or hover (●)
- **Background color:** State value $V^*(s)$ — darker = higher value
- **Special cells:** Rescue targets, charger, danger zones, blocked cells, wind zones
"""))
    cells.append(nb_code("""# ─────────────────────────────────────────────────────────────
# CELL: Task 3 — Policy Visualization
# PURPOSE: Visualize the optimal policy as a grid of directional arrows.
#          Show policy slices at different battery levels and target states.
# ─────────────────────────────────────────────────────────────

def visualize_policy(
    V: Dict, policy: Dict,
    battery_level: int,
    t1: int, t2: int,
    ax=None, title: str = ""
):
    \"\"\"
    Visualize the optimal policy and state values as a grid.

    Parameters
    ----------
    V : dict
        Optimal value function.
    policy : dict
        Optimal policy mapping state → action.
    battery_level : int
        Battery level to visualize.
    t1, t2 : int
        Target rescue status (0=not rescued, 1=rescued).
    ax : matplotlib Axes
        Axes to draw on (creates new figure if None).
    title : str
        Plot title.
    \"\"\"
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # ── Extract V-values for this slice ──────────────────────
    V_grid = np.full((GRID_ROWS, GRID_COLS), np.nan)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            state = (r, c, battery_level, t1, t2)
            if state in V:
                V_grid[r, c] = V[state]

    # ── Background: Value Heatmap ─────────────────────────────
    # Masked array to handle NaN (blocked/unreachable cells)
    masked_V = np.ma.masked_invalid(V_grid)
    cmap = plt.cm.RdYlGn
    im = ax.imshow(masked_V, cmap=cmap, aspect='equal', origin='upper',
                   vmin=np.nanmin(V_grid), vmax=np.nanmax(V_grid))

    # ── Arrow Map ─────────────────────────────────────────────
    ARROW = {0: '↑', 1: '↓', 2: '←', 3: '→', 4: '●'}
    ARROW_DX = {0: 0, 1: 0, 2: -0.35, 3: 0.35, 4: 0}   # x offset for arrow head
    ARROW_DY = {0: -0.35, 1: 0.35, 2: 0, 3: 0, 4: 0}

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cell = GRID_MAP.get((r, c), CELL_SAFE)

            # ── Cell Label (cell type overlays) ───────────────
            if cell == CELL_BLOCKED:
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1,
                                           color='#2c3e50', zorder=2))
                ax.text(c, r, 'X', ha='center', va='center',
                        fontsize=14, color='white', fontweight='bold', zorder=3)
                continue

            # Draw cell labels
            cell_labels = {
                CELL_RESCUE: ('🎯', '#27ae60'),
                CELL_CHARGER: ('⚡', '#f39c12'),
                CELL_DANGER: ('💥', '#c0392b'),
                CELL_WIND: ('🌀', '#8e44ad'),
                CELL_START: ('🏠', '#2980b9'),
            }
            if cell in cell_labels:
                emoji, color = cell_labels[cell]
                ax.text(c, r - 0.32, emoji, ha='center', va='center',
                        fontsize=12, zorder=3)

            # ── Policy Arrow ──────────────────────────────────
            state = (r, c, battery_level, t1, t2)
            if state in policy:
                act = policy[state]
                ax.text(c, r + 0.15, ARROW[act], ha='center', va='center',
                        fontsize=16, color='#2c3e50', fontweight='bold', zorder=4)

            # ── V-value label ─────────────────────────────────
            if state in V and not np.isnan(V.get(state, np.nan)):
                ax.text(c, r + 0.38, f'{V[state]:.1f}',
                        ha='center', va='center', fontsize=7,
                        color='#1a1a1a', alpha=0.8, zorder=4)

    # ── Grid Lines ────────────────────────────────────────────
    ax.set_xticks(np.arange(-0.5, GRID_COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID_ROWS, 1), minor=True)
    ax.grid(which='minor', color='#ecf0f1', linewidth=1.5)
    ax.tick_params(which='minor', size=0)
    ax.set_xticks(range(GRID_COLS))
    ax.set_yticks(range(GRID_ROWS))
    ax.set_xticklabels([f'Col {i}' for i in range(GRID_COLS)])
    ax.set_yticklabels([f'Row {i}' for i in range(GRID_ROWS)])
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)

    return im


# ── Multi-panel policy visualization ─────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 13))
fig.suptitle(
    'Optimal Policy π*(s) — Drone Rescue Grid World\\n'
    '(Arrows = optimal action | Background = state value | Numbers = V*(s))',
    fontsize=14, fontweight='bold'
)

configs = [
    (10, 0, 0, 'Battery=10, No rescues yet'),
    (10, 1, 0, 'Battery=10, R1 rescued'),
    (10, 0, 1, 'Battery=10, R2 rescued'),
    (5,  0, 0, 'Battery=5 (Low), No rescues'),
    (3,  0, 0, 'Battery=3 (Critical), No rescues'),
    (2,  0, 0, 'Battery=2 (Emergency), No rescues'),
]

ims = []
for ax, (bat, t1, t2, title) in zip(axes.flat, configs):
    im = visualize_policy(V_star, pi_star, bat, t1, t2, ax=ax, title=title)
    ims.append(im)

# Add a shared colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cb = fig.colorbar(ims[0], cax=cbar_ax)
cb.set_label('State Value V*(s)', fontsize=12)

plt.subplots_adjust(left=0.06, right=0.90, top=0.88, bottom=0.05,
                    wspace=0.35, hspace=0.35)
plt.savefig('plots/dp_policy_grid.png', bbox_inches='tight', dpi=150)
plt.show()
print("📊 Policy visualization saved.")
"""))

    # ── VALUE HEATMAP ──────────────────────────────────────────
    cells.append(nb_md("""## 🌡️ Task 4: State-Value Analysis (V* Heatmap)

### What does V*(s) tell us?

$V^*(s)$ represents the **maximum expected cumulative discounted reward** the drone can achieve starting from state $s$, following the optimal policy.

A high $V^*(s)$ means: *"Starting here, the drone will perform very well."*  
A low (negative) $V^*(s)$ means: *"Starting here is risky — low battery, bad position, etc."*

We plot heatmaps of $V^*(s)$ across all grid positions $(r,c)$ at fixed battery and target status slices.
"""))
    cells.append(nb_code("""# ─────────────────────────────────────────────────────────────
# CELL: Task 4 — State-Value Heatmap Analysis
# ─────────────────────────────────────────────────────────────
import os
os.makedirs('plots', exist_ok=True)

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(
    'State-Value Heatmap V*(r, c) at Fixed Battery & Target Status\\n'
    '(Higher = more valuable starting position for the drone)',
    fontsize=14, fontweight='bold'
)

battery_slices = [10, 7, 5, 3, 1]
target_configs = [
    (0, 0, 'No Rescues (t1=0, t2=0)'),
    (1, 0, 'R1 Rescued (t1=1, t2=0)'),
    (0, 1, 'R2 Rescued (t1=0, t2=1)'),
]

bat_levels = [10, 5, 10, 5, 3, 1]
tgt_configs = [(0,0),(0,0),(1,0),(1,0),(0,0),(0,0)]

titles_combined = [
    'Bat=10, No rescues',
    'Bat=5, No rescues',
    'Bat=10, R1 rescued',
    'Bat=5, R1 rescued',
    'Bat=3, No rescues',
    'Bat=1 (Critical), No rescues',
]

for ax, bat, (t1, t2), title in zip(axes.flat, bat_levels, tgt_configs, titles_combined):
    V_grid = np.full((GRID_ROWS, GRID_COLS), np.nan)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            state = (r, c, bat, t1, t2)
            if state in V_star:
                V_grid[r, c] = V_star[state]

    # ── Heatmap ───────────────────────────────────────────────
    masked = np.ma.masked_invalid(V_grid)
    vmin, vmax = np.nanmin(V_grid), np.nanmax(V_grid)
    im = ax.imshow(masked, cmap='RdYlGn', aspect='equal', origin='upper',
                   vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.8)

    # ── Overlay value numbers ─────────────────────────────────
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cell = GRID_MAP.get((r, c), CELL_SAFE)
            if cell == CELL_BLOCKED:
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1,
                                           color='#2c3e50', zorder=2))
                ax.text(c, r, 'X', ha='center', va='center',
                        fontsize=12, color='white', fontweight='bold', zorder=3)
                continue

            if not np.isnan(V_grid[r, c]):
                txt_color = 'white' if V_grid[r, c] < (vmin + (vmax-vmin)*0.4) else 'black'
                ax.text(c, r, f'{V_grid[r,c]:.1f}',
                        ha='center', va='center', fontsize=9,
                        color=txt_color, fontweight='bold', zorder=3)

            # Cell type annotations
            cell_symbols = {
                CELL_RESCUE: '🎯', CELL_CHARGER: '⚡',
                CELL_DANGER: '💥', CELL_WIND: '🌀', CELL_START: '🏠'
            }
            if cell in cell_symbols:
                ax.text(c, r - 0.35, cell_symbols[cell],
                        ha='center', va='center', fontsize=10, zorder=4)

    ax.set_xticks(range(GRID_COLS))
    ax.set_yticks(range(GRID_ROWS))
    ax.set_xticklabels([f'C{i}' for i in range(GRID_COLS)])
    ax.set_yticklabels([f'R{i}' for i in range(GRID_ROWS)])
    ax.set_title(title, fontsize=11, fontweight='bold')

    # Grid lines
    ax.set_xticks(np.arange(-0.5, GRID_COLS, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID_ROWS, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', size=0)

plt.tight_layout()
plt.savefig('plots/dp_value_heatmap.png', bbox_inches='tight', dpi=150)
plt.show()
print("📊 State-Value Heatmap saved.")
"""))

    # ── SCALABILITY ────────────────────────────────────────────
    cells.append(nb_md("""## 📐 Task 5: DP Scalability & Curse of Dimensionality

### The Curse of Dimensionality (Bellman, 1957)

Dynamic Programming (DP) requires iterating over the **entire state space** in each sweep. As environment complexity increases, the state space grows **exponentially** — a phenomenon called the **Curse of Dimensionality**.

#### State Space Growth for the Drone Problem:

| Scale | Grid | Targets | Battery | Wind Directions | State Count |
|-------|------|---------|---------|-----------------|-------------|
| Assignment (ours) | 5×5 | 2 | 10 | Static | **1,100** |
| Medium | 6×6 | 3 | 15 | Static | **6,048** |
| City Block | 100×100 | 10 | 100 | 4 dirs | **4 × 10¹⁰** |
| Real Drone | Continuous | Many | Continuous | ∞ | **∞** |

#### Why DP Fails at Scale:

1. **Memory:** Storing $V(s)$ for $4 \\times 10^{10}$ states requires terabytes of RAM.
2. **Computation:** Each Value Iteration sweep processes all states × all actions × all next states — quadratic to cubic complexity in the number of states.
3. **Model Requirement:** DP requires knowing $\\mathcal{P}(s'\\mid s, a)$ exactly — a complete, accurate world model. In real-world robotics, this is impossible to specify.

#### How Deep RL Overcomes This:

Instead of a tabular $V(s)$ (one float per state), Deep RL uses a **neural network** $V_\\theta(s)$ to *generalize* across states:

$$V_\\theta(s) \\approx V^*(s) \\quad \\forall s \\in \\mathcal{S}$$

Key Deep RL approaches:

| Method | Key Idea | Applicable To |
|--------|----------|---------------|
| **DQN** (Mnih et al., 2015) | CNN approximates Q*(s,a) | Discrete actions, high-dim obs |
| **PPO** (Schulman et al., 2017) | Policy gradient with clipped surrogate | Continuous action spaces |
| **SAC** (Haarnoja et al., 2018) | Off-policy, entropy-regularized | Robotic control |
| **MuZero** (Schrittwieser et al., 2020) | Learned world model + MCTS | Complex planning (Chess, Go, Atari) |

**The key trade-off:**
- DP: **Exact, guaranteed convergence** — but requires complete model and small state space.
- Deep RL: **Approximate, no convergence guarantees** — but scales to continuous, high-dimensional, real-world problems.
"""))
    cells.append(nb_code("""# ─────────────────────────────────────────────────────────────
# CELL: Task 5 — Curse of Dimensionality Visualization
# ─────────────────────────────────────────────────────────────

# State space size as environment parameters scale
grid_sizes   = [5,   6,   10,   20,   50,   100]
n_grids      = [g*g for g in grid_sizes]
n_batteries  = [10,  15,  20,   50,   100,  200]
n_targets_2  = [2**2, 2**3, 2**5, 2**8, 2**10, 2**15]   # 2^K for K targets

state_counts = [g * b * t for g, b, t in zip(n_grids, n_batteries, n_targets_2)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Curse of Dimensionality: State Space Growth', fontsize=14, fontweight='bold')

# Plot 1: State space vs grid size
ax1.semilogy(grid_sizes, state_counts, 'o-', color='#e74c3c', linewidth=2.5,
             markersize=10, markerfacecolor='white', markeredgewidth=2.5)
ax1.axhline(1100, color='#27ae60', linestyle='--', linewidth=2,
            label=f'Our assignment: {1100:,} states')
ax1.axhline(1e6,  color='#f39c12', linestyle='--', linewidth=1.5,
            label='~1M states (slow DP)')
ax1.axhline(1e9,  color='#c0392b', linestyle='--', linewidth=1.5,
            label='~1B states (infeasible DP)')
ax1.set_xlabel('Grid Side Length', fontsize=12)
ax1.set_ylabel('Total State Space Size (Log Scale)', fontsize=12)
ax1.set_title('State Space Grows Exponentially with Grid Size', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.4)

for x, y, label in zip(grid_sizes, state_counts, [f'{v:,.0f}' for v in state_counts]):
    ax1.annotate(label, (x, y), textcoords='offset points', xytext=(5, 5), fontsize=8)

# Plot 2: DP vs Deep RL scalability comparison
ax2.barh(['Tabular DP\\n(Our Assignment)', 'Tabular DP\\n(6×6 Grid)', 'DQN\\n(Atari Games)',
          'PPO\\n(Robot Control)', 'MuZero\\n(Chess/Go)'],
         [np.log10(1100), np.log10(6048), np.log10(33600), np.log10(1e10), np.log10(1e40)],
         color=['#27ae60', '#3498db', '#f39c12', '#e67e22', '#e74c3c'],
         alpha=0.85, edgecolor='white', linewidth=2)

ax2.set_xlabel('Log₁₀(State Space Size)', fontsize=12)
ax2.set_title('State Space Handled by Different Methods', fontsize=12)
ax2.axvline(6, color='black', linestyle='--', linewidth=1.5, alpha=0.5,
            label='~10⁶: DP feasibility limit')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.4, axis='x')

plt.tight_layout()
plt.savefig('plots/dp_curse_of_dimensionality.png', bbox_inches='tight', dpi=150)
plt.show()
print("📊 Curse of Dimensionality plot saved.")
"""))

    # ── FINAL SUMMARY ──────────────────────────────────────────
    cells.append(nb_code("""# ─────────────────────────────────────────────────────────────
# CELL: Final Summary
# ─────────────────────────────────────────────────────────────

print("=" * 70)
print("  PART 2 COMPLETE — DYNAMIC PROGRAMMING SUMMARY")
print("=" * 70)
print(f"  Student ID  : 2025aa05710  |  Group: 84")
print(f"  Grid        : {GRID_ROWS}×{GRID_COLS}  |  Battery: {MAX_BATTERY}  |  Wind: {WIND_PROB*100:.0f}%")
print()
print(f"  Value Iteration Results:")
print(f"    Converged in : {len(convergence_deltas)} iterations")
print(f"    Final Δ      : {convergence_deltas[-1]:.2e}")
print(f"    γ (discount) : {GAMMA}")
print()

# Show sample optimal values
sample_states = [
    (0, 0, 10, 0, 0, "Start, full battery, no rescues"),
    (1, 2, 10, 0, 0, "At R1 target, full battery"),
    (4, 0, 10, 0, 0, "At R2 target, full battery"),
    (2, 1, 10, 0, 0, "At charger, full battery"),
    (2, 2, 10, 0, 0, "At danger zone, full battery"),
    (0, 0,  2, 0, 0, "Start, critical battery"),
]

print(f"  Sample State Values:")
print(f"    {'State':<45} {'V*(s)':>8} {'π*(s)':>10}")
print("    " + "-" * 65)
for r, c, b, t1, t2, desc in sample_states:
    state = (r, c, b, t1, t2)
    val    = V_star.get(state, float('nan'))
    action = pi_star.get(state, -1)
    act_name = ACTION_NAMES.get(action, '?')
    print(f"    {desc:<45} {val:>8.2f} {act_name:>10}")

print()
print("  Key Observations:")
print("    1. Higher battery → Higher V* (more options available)")
print("    2. At rescue targets: V* is high (mission progress)")
print("    3. At danger zones: V* is lower (penalty risk)")
print("    4. Critical battery states have very low V* → policy forces charging")
print("    5. Value Iteration guarantees convergence via Bellman contraction")
print("=" * 70)
"""))

    nb.cells = cells
    return nb


# ============================================================
#  MAIN: Generate Both Notebooks
# ============================================================
if __name__ == "__main__":
    import os

    output_dir = r"."  # Current directory (Assignment 1 folder)

    # Generate MAB Notebook
    print("="*60)
    print("Generating Team_84_MAB.ipynb ...")
    mab_nb = make_mab_notebook()
    mab_path = os.path.join(output_dir, "Team_84_MAB.ipynb")
    with open(mab_path, 'w', encoding='utf-8') as f:
        nbformat.write(mab_nb, f)
    print(f"✅ Saved: {mab_path}")

    # Generate DP Notebook
    print()
    print("="*60)
    print("Generating Team_84_DP.ipynb ...")
    dp_nb = make_dp_notebook()
    dp_path = os.path.join(output_dir, "Team_84_DP.ipynb")
    with open(dp_path, 'w', encoding='utf-8') as f:
        nbformat.write(dp_nb, f)
    print(f"✅ Saved: {dp_path}")

    print()
    print("="*60)
    print("🎉 All notebooks generated successfully!")
    print(f"   - {mab_path}")
    print(f"   - {dp_path}")
    print("="*60)
    print()
    print("Next steps:")
    print("  1. Open JupyterLab in this directory")
    print("  2. Open Team_84_MAB.ipynb → Kernel → Restart & Run All")
    print("  3. Open Team_84_DP.ipynb  → Kernel → Restart & Run All")
    print("  4. Export both as PDF for submission")
