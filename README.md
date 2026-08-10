# ✈️ Travel & Hospitality – Reinforcement Learning for Dynamic Pricing

<div align="center">

### Revenue Optimization using Reinforcement Learning, Gymnasium, Q-Learning & Deep Q-Network (DQN)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Reinforcement%20Learning-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-blue?logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Visualization-4C72B0)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-RL-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![Git](https://img.shields.io/badge/Git-Version%20Control-F05032?logo=git)
![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)
![License](https://img.shields.io/badge/License-MIT-success)![Machine Learning](https://img.shields.io/badge/Machine%20Learning-AI-blueviolet)
![Reinforcement Learning](https://img.shields.io/badge/Reinforcement-Learning-success)
![Deep Learning](https://img.shields.io/badge/Deep-Learning-red)
![Q-Learning](https://img.shields.io/badge/Q--Learning-Algorithm-blue)
![DQN](https://img.shields.io/badge/Deep%20Q-Network-orange)
![MDP](https://img.shields.io/badge/Markov-Decision%20Process-green)
![OpenAI Gym](https://img.shields.io/badge/OpenAI-Gymnasium-lightgrey)![Data Science](https://img.shields.io/badge/Data-Science-blue)
![Analytics](https://img.shields.io/badge/Analytics-Business%20Intelligence-yellow)
![Simulation](https://img.shields.io/badge/Simulation-Market%20Environment-success)
![Optimization](https://img.shields.io/badge/Optimization-Revenue-red)![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit)
![GitHub Pages](https://img.shields.io/badge/GitHub-Pages-222222?logo=github)![Stars](https://img.shields.io/github/stars/your-username/your-repo?style=social)
![Forks](https://img.shields.io/github/forks/your-username/your-repo?style=social)
![Issues](https://img.shields.io/github/issues/your-username/your-repo)
![Last Commit](https://img.shields.io/github/last-commit/your-username/your-repo)
![Repo Size](https://img.shields.io/github/repo-size/your-username/your-repo)
![Contributors](https://img.shields.io/github/contributors/your-username/your-repo)
![Languages](https://img.shields.io/github/languages/top/your-username/your-repo)

**An end-to-end Reinforcement Learning project that learns optimal pricing strategies for airline seats or hotel rooms using a custom Gymnasium environment.**

</div>



---

## 📌 Project Overview

Travel and hospitality businesses often rely on static or manually designed pricing strategies. These approaches may fail to adapt effectively to changing demand, inventory levels, and time remaining before departure or booking deadlines.

This project addresses the problem by developing an autonomous Reinforcement Learning agent that learns pricing policies through interaction with a simulated booking environment.

The primary objective is to determine whether a learned RL policy can maximize revenue more effectively than traditional pricing heuristics.

### Core Question

> **Can a Reinforcement Learning agent learn a pricing strategy that outperforms human-designed pricing rules using only interaction and reward signals?**

---

## 🎯 Project Objectives

* Formulate dynamic pricing as a Markov Decision Process.
* Build a custom Gymnasium pricing environment.
* Simulate stochastic customer demand.
* Implement traditional pricing strategies as baselines.
* Develop a Tabular Q-Learning agent.
* Develop a PyTorch-based Deep Q-Network.
* Compare learned policies against heuristic strategies.
* Analyze revenue and inventory sell-through behavior.
* Evaluate training stability and convergence.
* Study how the learned policy responds to time-to-departure and remaining inventory.

---

## ⚔️ Pricing Strategies

The project evaluates multiple pricing strategies:

| Strategy                       | Type      | Description                                      |
| ------------------------------ | --------- | ------------------------------------------------ |
| **Random Agent**               | Baseline  | Selects pricing actions randomly                 |
| **Fixed Price Agent**          | Heuristic | Maintains a constant price                       |
| **Time-Based Discount Agent**  | Heuristic | Reduces price as the booking deadline approaches |
| **Demand-Based Pricing Agent** | Heuristic | Adjusts price according to demand and inventory  |
| **Q-Learning Agent**           | RL        | Learns an optimal policy using a Q-table         |
| **DQN Agent**                  | Deep RL   | Learns pricing decisions using a neural network  |

### Primary Evaluation Metrics

* Mean episodic revenue
* Revenue standard deviation
* Sell-through rate
* Revenue improvement
* Training reward
* Policy behavior
* Training stability

---

# 🧠 Reinforcement Learning Formulation

The dynamic pricing problem is represented as a **Markov Decision Process (MDP)**.

### State

The environment state contains information such as:

* Remaining inventory
* Days or time remaining
* Demand-related information
* Current pricing context

### Action

The agent selects a price from a predefined set of discrete pricing actions.

### Reward

The reward represents the revenue generated from the pricing decision.

### Transition

After an action is selected, the environment simulates customer demand, updates inventory, and moves to the next time step.

### Episode

One episode represents a simulated booking season in which the agent repeatedly makes pricing decisions until the inventory is exhausted or the booking horizon ends.

---

# 🏗️ Project Architecture

```text
Customer Demand
      │
      ▼
┌──────────────────────────┐
│ Custom Gymnasium         │
│ Pricing Environment      │
└────────────┬─────────────┘
             │
             ▼
      State Observation
             │
             ▼
┌──────────────────────────┐
│ Reinforcement Learning   │
│ Agent                    │
│                          │
│ • Q-Learning             │
│ • DQN                    │
└────────────┬─────────────┘
             │
             ▼
        Price Action
             │
             ▼
┌──────────────────────────┐
│ Demand Simulation        │
│ Inventory Update         │
│ Reward Calculation       │
└────────────┬─────────────┘
             │
             ▼
       Evaluation
             │
             ▼
 Revenue & Policy Analysis
```

---

# 🛠️ Technology Stack

| Category             | Technologies        |
| -------------------- | ------------------- |
| Programming Language | Python              |
| RL Environment       | Gymnasium           |
| Deep Learning        | PyTorch             |
| Data Processing      | NumPy, Pandas       |
| Visualization        | Matplotlib, Seaborn |
| Testing              | PyTest              |
| Version Control      | Git, GitHub         |
| Development          | Jupyter Notebook    |

---

# 📂 Repository Structure

```text
dynamic-pricing-rl/
│
├── env/
│   └── Custom Gymnasium pricing environment
│
├── agents/
│   ├── baseline_agents.py
│   ├── q_learning_agent.py
│   └── dqn_agent.py
│
├── baselines/
│   └── Heuristic pricing strategies
│
├── training/
│   └── train_dqn.py
│
├── evaluation/
│   └── Evaluation and benchmarking scripts
│
├── notebooks/
│   └── Exploratory analysis and experiments
│
├── requirements.txt
└── README.md
```

---

# 📅 Development Roadmap

| Week       | Focus                                         |
| ---------- | --------------------------------------------- |
| **Week 1** | MDP formulation and Gymnasium environment     |
| **Week 2** | Heuristic strategies and Tabular Q-Learning   |
| **Week 3** | DQN architecture, training and benchmarking   |
| **Week 4** | Final evaluation, visualization and reporting |



# 📖 Table of Contents

* Overview
* Project Objectives
* Problem Statement
* Key Features
* Project Architecture
* Repository Structure
* Technology Stack
* Development Roadmap
* Reinforcement Learning Formulation
* Algorithms
* Evaluation Metrics
* Dashboard
* Installation
* Usage
* Future Improvements
* Contributing
* License
* Author
* Acknowledgements

---

## 🎯 Overview

Hotels, airlines, and travel platforms lose revenue every day to static or overly simplistic pricing rules. This project designs an **autonomous Reinforcement Learning agent** — trained with **Deep Q-Networks (DQN)** — that learns to dynamically price inventory (rooms, seats, packages) across a simulated booking season.

The agent is benchmarked against three classic heuristic strategies to prove it can learn pricing behavior that **maximizes mean episodic revenue** more effectively than hand-crafted rules.

> **Core question:** *Can an RL agent learn pricing strategies that beat human-designed heuristics — purely from interaction and reward signal?*

<br>

## ⚔️ Agent vs. Baselines

| Strategy | Type | Description |
|---|---|---|
| 🎯 **RL Agent (DQN)** | Learned | Adapts pricing policy based on demand signals, time-to-departure, and inventory state |
| 🔒 Fixed Pricing | Heuristic | Constant price regardless of context |
| ⏳ Time-based Discounting | Heuristic | Price decays as the booking window closes |
| 📈 Demand-based Pricing | Heuristic | Price scales directly with observed demand |

**Success metric:** Mean episodic revenue across simulated booking seasons, RL agent vs. all three baselines.

<br> 

## 🧠 How It Works

```mermaid
flowchart LR
    A[Booking Environment
Gymnasium] -->|State: demand, time,
inventory| B[DQN Agent
PyTorch]
    B -->|Action: set price| A
    A -->|Reward: revenue| B
    B --> C[Policy Evaluation
& Analysis]
    C --> D[Benchmark vs.
Heuristic Baselines]
```

<br>

## 🛠️ Tech Stack

<div align="center">

| Category | Tools |
|---|---|
| **Language** | Python |
| **RL Environment** | Gymnasium |
| **Deep Learning** | PyTorch |
| **Data Handling** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |

</div>

<br>

## 📂 Project Structure

```
dynamic-pricing-rl/
├── env/                # Custom Gymnasium environment (MDP design)
├── agents/             # DQN agent implementation
├── baselines/          # Fixed, time-based, demand-based pricing
├── training/           # Training loops, replay buffer, configs
├── evaluation/         # Reward curves, revenue comparison, plots
├── notebooks/          # Exploratory analysis
├── requirements.txt
└── README.md
```

<br>

## 👥 Team & Roles

<div align="center">

| Role | Focus Area |
|---|---|
| 🌍 **Environment & Simulation Engineer** | MDP design, booking environment, demand simulation |
| 🤖 **RL Algorithm Engineer** | DQN architecture, training pipeline, hyperparameter tuning |
| 📊 **Analysis & Policy Evaluation** | Reward analysis, policy interpretability, benchmarking |
| 🚀 **Eval & Deploy Lead** | Final evaluation suite, reproducibility, deployment packaging |

</div>

<br>

## 🗺️ Roadmap

- [x] **Week 1** — MDP formulation & environment design
- [x] **Week 2** — DQN agent development & training
- [x] **Week 3** — Policy evaluation & revenue benchmarking
- [ ] **Week 4** — Final analysis, visualization & report

<br>

## 📈 Expected Deliverables

- A fully specified MDP (state, action, reward design) for travel pricing
- A trained DQN agent with reproducible training pipeline
- Comparative revenue plots: RL agent vs. all heuristic baselines
- Policy analysis explaining *what* the agent learned and *why* it works

<br>

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-org>/dynamic-pricing-rl.git
cd dynamic-pricing-rl

# Install dependencies
pip install -r requirements.txt

# Run training
python training/train.py

# Evaluate against baselines
python evaluation/evaluate.py
```

<br>

## 📄 License

This project is licensed under the MIT License.

<br>

<div align="center">

**Built for smarter, adaptive pricing in travel & hospitality — one episode at a time.**

</div>

# 📅 Week 1 Progress Timeline

| Day       | Theme                                     | Status |
| --------- | ----------------------------------------- | :----: |
| Monday    | Project Initialization & Repository Setup |    ✅   |
| Tuesday   | MDP Design & Environment Skeleton         |    ✅   |
| Wednesday | Environment Implementation & Baseline     |    ✅   |
| Thursday  | Demand Modeling & Evaluation              |    ✅   |
| Friday    | Finalization, Documentation & Integration |    ✅   |

---

# 👥 Team Contributions

---

## 🟦 Member 1 — Environment & Simulation Engineer

**Primary Responsibility**

* Design and implement the custom Gymnasium pricing environment.

### 📌 Monday

* Created project folder structure
* Configured Python virtual environment
* Installed required dependencies
* Organized source code layout

### 📌 Tuesday

* Developed `PricingEnv(gym.Env)` skeleton
* Defined Observation Space
* Defined Action Space
* Implemented environment constructor (`__init__`)

### 📌 Wednesday

* Implemented `reset()`
* Implemented `step(action)`
* Updated inventory
* Managed episode transitions
* Generated environment rewards

### 📌 Thursday

* Designed stochastic customer demand model
* Added price sensitivity
* Added time-to-departure demand behavior
* Implemented logistic demand curve

### 📌 Friday

* Added `render()` method
* Performed stability testing
* Executed 100 random episodes
* Verified environment consistency
* Eliminated invalid state transitions

**✅ Total Commits:** **5**

---

## 🟩 Member 2 — RL Algorithm Engineer

**Primary Responsibility**

* Define Reinforcement Learning formulation and environment logic.

### 📌 Monday

* Configured Jupyter Notebook environment
* Installed `nbstripout`
* Created environment design notebook

### 📌 Tuesday

* Formalized complete MDP
* Defined:

  * State Space
  * Action Space
  * Transition Function
  * Reward Function
  * Episode Horizon

### 📌 Wednesday

* Added action masking
* Restricted invalid pricing actions
* Tested edge cases

  * Zero inventory
  * Zero remaining days

### 📌 Thursday

* Developed automated PyTest unit tests
* Tested:

  * reset()
  * step()
  * reward calculation
  * terminal conditions

### 📌 Friday

* Completed environment documentation
* Added mathematical explanation
* Documented reward function
* Documented demand model
* Updated notebook

**✅ Total Commits:** **5**

---

## 🟨 Member 3 — Analysis & Policy Evaluation

**Primary Responsibility**

* Evaluate baseline performance and generate analytics.

### 📌 Monday

* Created Random Agent notebook
* Reviewed Gymnasium API
* Studied MDP references

### 📌 Tuesday

* Implemented Random Agent
* Connected agent with Pricing Environment

### 📌 Wednesday

* Ran Random Agent
* Executed **500 Episodes**
* Recorded episodic revenue

### 📌 Thursday

* Generated revenue histogram
* Calculated:

  * Mean Revenue
  * Standard Deviation
* Evaluated baseline statistics

### 📌 Friday

* Visualized episode trajectory
* Plotted:

  * Inventory over time
  * Selected price levels
* Cleared notebook outputs

**✅ Total Commits:** **5**

---

## 🟥 Member 4 — Evaluation & Deployment Lead

**Primary Responsibility**

* Repository management, documentation and project coordination.

### 📌 Monday

* Created GitHub Repository
* Added `.gitignore`
* Wrote project README
* Created Kanban Board
* Created all 20 GitHub Issues
* Assigned issues to team members

### 📌 Tuesday

* Created development branch
* Added `AGENTS.md`
* Defined all team roles
* Pushed branch to GitHub

### 📌 Wednesday

* Added project rules
* Created Brain Memory directory
* Documented project context
* Added MDP definitions

### 📌 Thursday

* Created team progress tracker
* Reviewed Pull Requests
* Merged completed work into main branch

### 📌 Friday

* Closed Week 1 GitHub Issues
* Updated Kanban Board
* Wrote Week 1 Summary
* Cleaned notebook outputs
* Final repository synchronization

**✅ Total Commits:** **5**

---

# 📊 Week 1 Sprint Statistics

| Metric                 |       Value |
| ---------------------- | ----------: |
| Sprint Duration        |      5 Days |
| Team Members           |           4 |
| Git Commits            |      **20** |
| GitHub Issues          |      **20** |
| Pull Requests Reviewed |    Multiple |
| Gym Environment        | ✅ Completed |
| MDP Formulation        | ✅ Completed |
| Random Agent Baseline  | ✅ Completed |
| Environment Testing    | ✅ Completed |
| Documentation          | ✅ Completed |

---

# 🏆 Week 1 Deliverables

✅ Project Repository Initialized

✅ Professional Folder Structure

✅ GitHub Kanban Workflow

✅ Issue Tracking

✅ Team Role Definition

✅ Complete Markov Decision Process (MDP)

✅ Custom Gymnasium Environment

✅ Reward Function

✅ Stochastic Demand Simulation

✅ Random Agent Baseline

✅ 500 Episode Evaluation

✅ Revenue Distribution Analysis

✅ Episode Visualization

✅ Automated Unit Tests

✅ Complete Documentation

---

# 🎯 Week 1 Outcome

At the end of Week 1, the project successfully established a complete Reinforcement Learning foundation by designing the Dynamic Pricing problem as a Markov Decision Process (MDP), implementing a custom Gymnasium environment, and validating its behavior using a Random Agent baseline. The environment now supports realistic inventory dynamics, stochastic customer demand, reward computation, and simulation-based experimentation, providing a robust platform for implementing advanced RL algorithms in subsequent development phases.

---

# 📅 Week 2 Progress Timeline

| Day       | Theme                                          | Status |
| --------- | ---------------------------------------------- | :----: |
| Monday    | Baseline Agent Development & Q-Learning Setup  |    ✅   |
| Tuesday   | Bellman Learning & Baseline Evaluation         |    ✅   |
| Wednesday | Q-Learning Training & Performance Benchmarking |    ✅   |
| Thursday  | Hyperparameter Optimization & Result Analysis  |    ✅   |
| Friday    | Finalization, Documentation & Integration      |    ✅   |

---

# 👥 Team Contributions

---

## 🟦 Member 1 — Environment & Simulation Engineer

**Primary Responsibility**

* Develop and validate heuristic pricing strategies for comparison with Reinforcement Learning models.

### 📌 Monday

* Implemented **FixedPriceAgent**
* Created `baseline_agents.py`
* Added constant pricing strategy
* Integrated agent with Pricing Environment

### 📌 Tuesday

* Developed **TimeBasedDiscountAgent**
* Implemented automatic 10% daily price reduction
* Tested pricing strategy in simulation environment
* Verified seasonal pricing behavior

### 📌 Wednesday

* Implemented **DemandBasedAgent**
* Designed inventory-to-time pricing strategy
* Tested stability over multiple simulation episodes
* Validated adaptive pricing logic

### 📌 Thursday

* Finalized all heuristic pricing agents
* Added comprehensive documentation
* Standardized code structure
* Verified functionality of all baseline strategies

### 📌 Friday

* Completed Week 2 Baseline Notebook
* Added introduction, methodology and conclusion
* Cleared notebook outputs
* Prepared final notebook for GitHub

**✅ Total Commits:** **5**

---

## 🟩 Member 2 — RL Algorithm Engineer

**Primary Responsibility**

* Build and optimize the Tabular Q-Learning algorithm.

### 📌 Monday

* Discretized state space
* Created Inventory Buckets
* Created Days Remaining Buckets
* Initialized Q-table

### 📌 Tuesday

* Implemented Bellman Update Equation
* Added learning rate
* Added discount factor
* Implemented epsilon-greedy exploration
* Configured epsilon decay

### 📌 Wednesday

* Trained Q-Learning agent
* Executed **5,000 training episodes**
* Logged reward curve
* Monitored training convergence

### 📌 Thursday

* Tuned hyperparameters
* Evaluated multiple learning rates
* Compared discount factors
* Selected optimal epsilon decay schedule

### 📌 Friday

* Finalized `q_learning_agent.py`
* Evaluated best model on **500 unseen episodes**
* Cleaned notebook outputs
* Documented final implementation

**✅ Total Commits:** **5**

---

## 🟨 Member 3 — Analysis & Policy Evaluation

**Primary Responsibility**

* Evaluate heuristic strategies and compare them with Q-Learning.

### 📌 Monday

* Built common evaluation framework
* Implemented reusable simulation helper
* Standardized evaluation metrics

### 📌 Tuesday

* Evaluated Random Agent
* Evaluated Fixed Price Agent
* Evaluated Time-Based Discount Agent
* Generated revenue comparison plots

### 📌 Wednesday

* Compared Q-Learning against all heuristic agents
* Calculated:

  * Mean Revenue
  * Standard Deviation
  * Sell-through Rate
* Recorded benchmarking results

### 📌 Thursday

* Computed revenue improvement percentage
* Compared Q-Learning with best-performing heuristic
* Documented experimental findings

### 📌 Friday

* Finalized comparison notebook
* Added complete performance summary table
* Verified evaluation metrics
* Cleared notebook outputs

**✅ Total Commits:** **5**

---

## 🟥 Member 4 — Evaluation & Deployment Lead

**Primary Responsibility**

* Coordinate evaluation, documentation and repository integration.

### 📌 Monday

* Created `results_comparison.md`
* Designed result documentation structure
* Added comparison section templates

### 📌 Tuesday

* Reviewed Q-Learning implementation
* Verified Bellman update correctness
* Reviewed Pull Requests
* Added technical review comments

### 📌 Wednesday

* Created baseline comparison tables
* Organized Week 2 evaluation notebook
* Documented performance metrics

### 📌 Thursday

* Wrote Week 2 findings
* Explained improvements achieved by Q-Learning
* Documented comparison with heuristic pricing strategies

### 📌 Friday

* Updated GitHub Kanban Board
* Closed Week 2 Issues
* Updated README
* Reviewed and merged all Pull Requests
* Completed Week 2 repository synchronization

**✅ Total Commits:** **5**

---

# 📊 Week 2 Sprint Statistics

| Metric                   |                 Value |
| ------------------------ | --------------------: |
| Sprint Duration          |                5 Days |
| Team Members             |                     4 |
| Git Commits              |                **20** |
| GitHub Issues Completed  | **5** (Issues #6–#10) |
| Pull Requests Reviewed   |              Multiple |
| Heuristic Pricing Agents |           ✅ Completed |
| Tabular Q-Learning       |           ✅ Completed |
| Hyperparameter Tuning    |           ✅ Completed |
| Agent Benchmarking       |           ✅ Completed |
| Documentation            |           ✅ Completed |

---

# 🏆 Week 2 Deliverables

✅ Fixed Price Agent

✅ Time-Based Discount Agent

✅ Demand-Based Pricing Agent

✅ Shared Evaluation Framework

✅ State Space Discretization

✅ Tabular Q-Learning Implementation

✅ Bellman Learning Algorithm

✅ Hyperparameter Optimization

✅ 5,000 Episode Training

✅ 500 Episode Testing

✅ Baseline vs Q-Learning Comparison

✅ Revenue Performance Analysis

✅ Sell-through Rate Evaluation

✅ Week 2 Documentation

✅ Updated GitHub Repository

---

# 🎯 Week 2 Outcome

By the end of Week 2, the project successfully established a strong Reinforcement Learning baseline by implementing multiple heuristic pricing strategies and developing a complete Tabular Q-Learning agent. The Q-Learning model was trained, optimized through hyperparameter tuning, and rigorously evaluated against heuristic approaches using extensive simulation experiments. Performance metrics such as mean revenue, sell-through rate, and revenue improvement demonstrated that the learned policy consistently outperformed static pricing strategies, providing a solid foundation for transitioning to a Deep Q-Network (DQN) architecture in the next phase of the project. 

# 📅 Week 3 Progress Timeline

| Day       | Theme                                          | Status |
| --------- | ----------------------------------------------- | :----: |
| Monday    | DQN Architecture & Baseline Evaluation Setup    |    ✅   |
| Tuesday   | Target Network & Experience Replay              |    ✅   |
| Wednesday | Training Stability & Exploration Strategy       |    ✅   |
| Thursday  | Full DQN Training & Performance Benchmarking    |    ✅   |
| Friday    | Finalization, Documentation & Integration       |    ✅   |

---

# 👥 Team Contributions

---

## 🟦 Tamilarasan — Environment & Simulation Engineer

**Primary Responsibility**

* Build the DQN network architecture and validate training stability.

### 📌 Monday

* Set up PyTorch `DQNNetwork` class
* Defined input layer (state features)
* Added 2 hidden layers with ReLU activation
* Defined output layer (Q-value per discrete price action)

### 📌 Tuesday

* Implemented target network with periodic hard update (every N steps)
* Verified target network weights sync correctly with policy network

### 📌 Wednesday

* Logged and plotted training loss curve
* Logged and plotted episodic reward curve across DQN training steps
* Identified signs of divergence

### 📌 Thursday

* Tested DQN convergence across 3 random seeds
* Confirmed reward curve stabilizes without diverging
* Tuned learning rate / target update frequency where needed

### 📌 Friday

* Finalized `train_dqn.py` training script
* Saved best-performing model checkpoint (excluded from git via `.gitignore`)
* Cleared notebook outputs and pushed

**✅ Total Commits:** **5**

---

## 🟩 Tamilarasan — RL Algorithm Engineer

**Primary Responsibility**

* Implement the full DQN algorithm: forward pass, replay buffer, exploration, and training loop.

### 📌 Monday

* Implemented DQN architecture forward pass in `dqn_agent.py`
* Defined loss function (Huber loss)
* Defined optimizer (Adam)

### 📌 Tuesday

* Implemented `ReplayBuffer` class with `push()` and `sample()` methods
* Used a deque of fixed capacity
* Tested buffer sampling returns correctly shaped batches

### 📌 Wednesday

* Implemented epsilon-greedy exploration strategy with exponential decay schedule
* Integrated replay buffer and target network into the full training loop

### 📌 Thursday

* Trained full DQN agent for **2,000 episodes**
* Used replay buffer and epsilon-greedy exploration
* Saved training checkpoints every 200 episodes

### 📌 Friday

* Finalized `dqn_agent.py` with complete DQN implementation (network, replay buffer, epsilon-greedy, training loop)
* Added docstrings throughout
* Cleared outputs and pushed

**✅ Total Commits:** **5**

---

## 🟨 Tamilarasan — Analysis & Policy Evaluation

**Primary Responsibility**

* Evaluate the Q-Learning baseline and the trained DQN agent, and analyze learned pricing behavior.

### 📌 Monday

* Loaded best Q-Learning agent from Week 2
* Set up evaluation harness to run any trained agent for a configurable number of episodes

### 📌 Tuesday

* Ran trained Q-Learning agent for **500 evaluation episodes**
* Plotted price trajectory over time for 3 sample episodes

### 📌 Wednesday

* Analyzed policy behavior on Q-Learning agent
* Checked whether it discounts price near the deadline
* Identified inventory-clearing patterns from sample trajectories

### 📌 Thursday

* Calculated inventory sell-through rate for trained DQN agent
* Calculated revenue per episode across **500 test episodes**

### 📌 Friday

* Wrote 150-word analysis of the learned DQN pricing policy
* Evaluated whether the agent learns to drop prices near the deadline to clear remaining stock
* Pushed clean notebook

**✅ Total Commits:** **5**

---

## 🟥 Tamilarasan — Evaluation & Deployment Lead

**Primary Responsibility**

* Track experiment results, coordinate comparisons, and manage repository documentation.

### 📌 Monday

* Created `model_results.md` file to track all experiment results
* Added table headers: Agent | Mean Revenue | Std Dev | Sell-through % | Notes
* Pushed to main

### 📌 Tuesday

* Recorded Week 3 Day 1–2 results in `model_results.md`
* Added tabular Q-Learning final CV results
* Wrote notes on training stability observed so far

### 📌 Wednesday

* Began Q-Learning vs DQN comparison table in `results_comparison.md`
* Added rows: Random, Fixed, Discount, Demand-based, Q-Learning, DQN (in progress)
* Filled in available results

### 📌 Thursday

* Finalized comparison table with all model results
* Added conclusion paragraph on which agent wins, by how much, and why
* Committed to main

### 📌 Friday

* Updated Kanban board — moved all Week 3 Issues to Done
* Added Week 3 Summary to README (DQN architecture, training stability, revenue vs Q-Learning)
* Reviewed and merged all Week 3 Pull Requests

**✅ Total Commits:** **5**

---

# 📊 Week 3 Sprint Statistics

| Metric                              |       Value |
| ------------------------------------ | ----------: |
| Sprint Duration                      |      5 Days |
| Team Members                         |           2 |
| Git Commits                          |      **20** |
| GitHub Issues Completed              | **5** (Issues #11–#15) |
| Pull Requests Reviewed               |    Multiple |
| DQN Network Architecture             | ✅ Completed |
| Target Network                       | ✅ Completed |
| Experience Replay Buffer             | ✅ Completed |
| Epsilon-Greedy Exploration           | ✅ Completed |
| Full DQN Training (2,000 episodes)   | ✅ Completed |
| DQN vs Q-Learning Comparison         | ✅ Completed |
| Documentation                        | ✅ Completed |

---

# 🏆 Week 3 Deliverables

✅ PyTorch DQN Network Architecture

✅ Huber Loss & Adam Optimizer

✅ Target Network with Periodic Sync

✅ Experience Replay Buffer

✅ Epsilon-Greedy Exploration with Decay

✅ Full DQN Training Loop

✅ Convergence Testing Across Multiple Seeds

✅ 2,000 Episode DQN Training

✅ 500 Episode DQN Evaluation

✅ Q-Learning vs DQN Comparison Table

✅ Sell-through Rate & Revenue Analysis

✅ Learned Policy Behavior Analysis

✅ Model Results Tracking Document

✅ Week 3 Documentation

✅ Updated GitHub Repository

---

# 🎯 Week 3 Outcome

By the end of Week 3, the project successfully transitioned from tabular Q-Learning to a full Deep Q-Network (DQN) capable of handling the complete continuous state space. A PyTorch-based DQN architecture was implemented with a target network and experience replay buffer to stabilize training, and an epsilon-greedy exploration strategy was integrated into the full training loop. The DQN agent was trained for 2,000 episodes, validated for convergence stability across multiple random seeds, and rigorously evaluated against the tabular Q-Learning baseline. Results confirmed that the DQN agent outperformed tabular Q-Learning on the full state space, with the learned policy demonstrating deadline-aware price discounting behavior consistent with effective inventory clearance — setting a strong foundation for further model refinement in subsequent phases.

---
# 📜 License

This project is released under the MIT License.

---

# 👨‍💻 Author

**Tamilarasan P**

B.Tech – Computer Science and Engineering

Artificial Intelligence • Machine Learning • Data Science

**GitHub:** https://github.com/tamil1208

**LinkedIn:**  https://www.linkedin.com/in/tamilarasan-a2466b274/

---

# 🙏 Acknowledgements

This project was developed as part of an **Advanced Data Science & Machine Learning Internship** focusing on Reinforcement Learning for Dynamic Pricing in the Travel & Hospitality domain.

Special thanks to the internship mentors, the Gymnasium community, Stable-Baselines3 contributors, PyTorch developers, and the open-source AI community for providing the tools and resources that made this project possible.

---

<div align="center">

⭐ **If you found this project useful, please consider giving it a star!**

</div>

