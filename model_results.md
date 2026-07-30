# 📊 Model Results Tracker

This document records the performance and progress of all pricing agents developed during the project, including baseline heuristic methods, Tabular Q-Learning, and the Deep Q-Network (DQN).

---

# 📅 Week 1

| Agent | Mean Revenue | Std Dev | Sell-through (%) | Status | Notes |
|--------|-------------:|---------:|-----------------:|--------|-------|
| Random Agent | - | - | - | ✅ Completed | Initial baseline for environment validation |

### Progress Highlights

- Designed the project evaluation framework.
- Implemented a Random Agent as the baseline policy.
- Validated the custom Gymnasium environment.
- Verified state transitions, action execution, and reward calculations.
- Established baseline metrics for future model comparisons.

---

# 📅 Week 2

| Agent | Mean Revenue | Std Dev | Sell-through (%) | Status | Notes |
|--------|-------------:|---------:|-----------------:|--------|-------|
| Fixed Price | - | - | - | ✅ Completed | Static pricing strategy |
| Time-Based Discount | - | - | - | ✅ Completed | Rule-based pricing |
| Demand-Based Pricing | - | - | - | ✅ Completed | Dynamic heuristic |
| Tabular Q-Learning | - | - | - | ✅ Completed | Reinforcement Learning baseline |

### Progress Highlights

- Implemented multiple heuristic pricing strategies.
- Developed the Tabular Q-Learning agent.
- Trained and evaluated the Q-Learning model.
- Compared heuristic approaches with RL-based pricing.
- Generated preliminary revenue and occupancy evaluation metrics.
- Prepared benchmark results for DQN comparison.

---

# 📅 Week 3

| Agent | Mean Revenue | Std Dev | Sell-through (%) | Status | Notes |
|--------|-------------:|---------:|-----------------:|--------|-------|
| Deep Q-Network (DQN) | - | - | - | 🚧 In Progress | Deep Reinforcement Learning |

### Day 1

- Designed the Deep Q-Network (DQN) architecture using PyTorch.
- Built a neural network with two hidden layers and ReLU activation.
- Configured the output layer to predict Q-values for ten discrete pricing actions.
- Verified forward propagation and action selection.

### Day 2

- Implemented the Target Network for stable learning.
- Added periodic hard synchronization using `sync_target_network()`.
- Successfully synchronized policy and target network weights.
- Reduced training instability caused by continuously changing target values.

### Upcoming Tasks

- Complete DQN training.
- Perform hyperparameter tuning.
- Compare DQN against heuristic and Q-Learning baselines.
- Generate performance plots and evaluation reports.
- Integrate the trained model into the Streamlit dashboard.
- Export the best-performing model checkpoint.

---
