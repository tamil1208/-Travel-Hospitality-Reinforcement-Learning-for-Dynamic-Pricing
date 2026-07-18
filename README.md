# ✈️ Travel & Hospitality – Reinforcement Learning for Dynamic Pricing

<div align="center">

### Revenue Optimization using Reinforcement Learning, Gymnasium, Q-Learning & Deep Q-Network (DQN)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Gymnasium](https://img.shields.io/badge/Gymnasium-Reinforcement%20Learning-green)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-RL-purple)
![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-blue?logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![Git](https://img.shields.io/badge/Git-Version%20Control-F05032?logo=git)
![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)
![License](https://img.shields.io/badge/License-MIT-success)

**An end-to-end Reinforcement Learning project that learns optimal pricing strategies for airline seats or hotel rooms using a custom Gymnasium environment.**

</div>

---

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

# 📌 Overview

Dynamic pricing plays a critical role in industries with limited inventory, such as airlines, hotels, and travel platforms. Prices that are too low reduce revenue, while prices that are too high can leave inventory unsold.

This project develops an intelligent Reinforcement Learning (RL) agent that learns pricing strategies by interacting with a simulated booking environment. Instead of relying on fixed business rules, the agent continuously improves its pricing policy to maximize cumulative revenue.

---

# 🎯 Project Objectives

* Maximize total revenue
* Improve inventory utilization
* Minimize unsold inventory
* Learn adaptive pricing policies
* Compare RL models with traditional pricing strategies
* Visualize learning performance using interactive dashboards

---

# 🧠 Problem Statement

The pricing decision depends on multiple factors:

* Remaining inventory
* Days until departure
* Current ticket or room price
* Customer demand

The objective is to determine the optimal price at every decision step to maximize long-term cumulative reward.

---

# ✨ Key Features

* Custom Gymnasium Environment
* Stochastic Demand Simulation
* Markov Decision Process (MDP)
* Fixed Price Baseline
* Time-Based Pricing
* Tabular Q-Learning
* Deep Q-Network (DQN)
* Experience Replay
* Target Network
* Epsilon-Greedy Exploration
* Performance Evaluation
* Streamlit Dashboard

---

# 🏗 Project Architecture

```text
                    Customer Demand
                           │
                           ▼
              Custom Gymnasium Environment
                           │
                           ▼
          Current State (Inventory, Time)
                           │
                           ▼
             Reinforcement Learning Agent
            (Q-Learning / Deep Q-Network)
                           │
                           ▼
                  Select Ticket Price
                           │
                           ▼
              Simulated Customer Purchase
                           │
                           ▼
                Reward (Revenue Earned)
                           │
                           ▼
                   Policy Optimization
```

---

# 📂 Repository Structure

```text
travel-hospitality-dynamic-pricing-rl/

├── environment/
│   └── pricing_env.py
│
├── agents/
│   ├── baseline.py
│   ├── qlearning.py
│   └── dqn.py
│
├── dashboard/
│   └── app.py
│
├── notebooks/
│
├── scripts/
│   ├── train.py
│   └── evaluate.py
│
├── configs/
├── utils/
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🛠 Technology Stack

| Category               | Technologies                 |
| ---------------------- | ---------------------------- |
| Programming Language   | Python 3.11                  |
| Reinforcement Learning | Gymnasium, Stable-Baselines3 |
| Deep Learning          | PyTorch                      |
| Data Processing        | NumPy, Pandas                |
| Visualization          | Matplotlib, Seaborn          |
| Dashboard              | Streamlit                    |
| Development Tools      | Git, GitHub, VS Code         |
| Notebook               | Jupyter Notebook             |

---

# 📅 Development Roadmap

## Week 1 – Environment Development

* Design the Markov Decision Process
* Build a custom Gymnasium environment
* Define State, Action and Reward spaces
* Simulate customer demand

## Week 2 – Classical Reinforcement Learning

* Fixed Pricing Strategy
* Time-Based Discount Strategy
* Implement Q-Learning
* Train Q-Table
* Compare baseline performance

## Week 3 – Deep Reinforcement Learning

* Deep Q-Network (DQN)
* Experience Replay
* Target Network
* Neural Network Policy
* Hyperparameter Tuning

## Week 4 – Evaluation & Deployment

* Revenue Comparison
* Policy Evaluation
* Dashboard Development
* Documentation
* GitHub Repository

---

# 🤖 Reinforcement Learning Formulation

## State Space

```text
(Remaining Inventory,
 Days Until Departure)
```

Example

```text
(75, 12)
```

---

## Action Space

| Action | Price |
| ------ | ----- |
| 0      | ₹4000 |
| 1      | ₹5000 |
| 2      | ₹6000 |
| 3      | ₹7000 |
| 4      | ₹8000 |

---

## Reward Function

```text
Reward = Selling Price (if sold)

Reward = 0 (if no purchase)
```

---

# 🧠 Algorithms

## Baseline Models

* Fixed Pricing
* Time-Based Discount Pricing

## Reinforcement Learning Models

* Tabular Q-Learning
* Deep Q-Network (DQN)

---

# 📈 Evaluation Metrics

The trained agents are evaluated using:

* Total Revenue
* Average Reward
* Revenue Per Episode
* Inventory Utilization
* Unsold Inventory
* Learning Curve
* Reward Convergence
* Policy Stability

---

# 📊 Dashboard

The Streamlit dashboard includes:

* Revenue Overview
* Reward Curve
* Price Trajectory
* Inventory Trend
* Episode Statistics
* Baseline Comparison
* RL Agent Performance

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/travel-hospitality-dynamic-pricing-rl.git
```

Navigate to the project:

```bash
cd travel-hospitality-dynamic-pricing-rl
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶ Usage

Train the RL agent:

```bash
python scripts/train.py
```

Evaluate the trained model:

```bash
python scripts/evaluate.py
```

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

---

# 📷 Results

The trained Reinforcement Learning agent is expected to:

* Learn adaptive pricing strategies
* Increase cumulative revenue
* Reduce unsold inventory
* Outperform fixed and rule-based pricing approaches

> Add your reward curves, revenue plots, and dashboard screenshots here after training.

---

# 📌 Future Improvements

* Proximal Policy Optimization (PPO)
* Soft Actor-Critic (SAC)
* Continuous Action Spaces
* Multi-Agent Reinforcement Learning
* Competitor Pricing Simulation
* Seasonal Demand Forecasting
* Hyperparameter Optimization
* Cloud Deployment

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Commit your changes.
4. Push your branch.
5. Open a Pull Request.

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

