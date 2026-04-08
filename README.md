<div align="center">

# 🚦 Coordinating Chaos: Multi-Agent RL for Traffic Bottlenecks

**Understanding & Mitigating Urban Congestion Through Artificial Intelligence**

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Gymnasium](https://img.shields.io/badge/Environment-Gymnasium-orange.svg)](https://gymnasium.farama.org/)
[![Stable Baselines3](https://img.shields.io/badge/RL-Stable_Baselines3-purple.svg)](https://stable-baselines3.readthedocs.io/en/master/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

*A research initiative demonstrating that urban traffic bottlenecks are a behavioral coordination problem, not just a capacity issue.*

</div>

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [The Simulation Explained](#-the-simulation-explained)
- [The Core Problem & Hypothesis](#-the-core-problem--hypothesis)
- [Methodology & Architecture](#%EF%B8%8F-methodology--architecture)
- [Key Findings](#-key-findings)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)

---

## 🌍 Project Overview

Modern traffic management often operates on the assumption that more cars inherently cause more traffic. **This project challenges that narrative.** 

By developing a custom high-fidelity traffic simulation environment (`BottleneckEnv`), we investigate how congestion is an emergent property of decentralized selfishness (aggressive driving). Through the deployment of a mere 20% population of Reinforcement Learning (RL) agents trained with a mixed-reward structure (optimizing individual speed + collective flow), we successfully demonstrate a measurable improvement in overall throughput and a massive reduction in conflict rates. 

This repository houses the simulation environment, behavioral profiles, PPO training pipelines, and experiment runners used to validate this hypothesis.

---

## 🎮 The Simulation Explained

> **Note on Visuals:** *Please insert your live simulation screenshot below (`visualization_screenshot.png`) to showcase the PyGame rendering.*

<div align="center">
  <img src="https://github.com/Tripathi-Nishant/KnowledgeQuarry-Outliers/blob/main/Screenshot%202026-04-07%20114738.png" alt="Live Traffic Simulation Representation" width="100%" />
</div>

The custom simulation (`pygame_renderer.py`) serves as our "Virtual Brain" to visualize and measure throughput in real-time. Here is how to interpret the rendering:

### The Environment
- **The 3 Lanes**: Represent the initial highway segment.
- **The Red Vertical Line**: The structural bottleneck. Past this critical transition point, the outer lanes are blocked. Every vehicle must successfully merge into the center lane.
- **The Red Pulsing Halo**: Appears when a vehicle is "stuck" or experiences a forced braking/conflict due to poor merging.

### Agent Behavioral Profiles (Color-Coded)
The environment consists of distinct driver archetypes:
*   🔵 **Blue (Cooperative)**: Standard IDM-lite physics; rule-following drivers who merge early and maintain safe distances.
*   🔴 **Red (Aggressive)**: Selfish drivers who wait until the very last second to merge, ignoring yield rules and creating diagonal trajectory conflicts.
*   🟢 **Green (RL Agents)**: The trained Artificial Intelligence. These agents act as "flow dampeners," learning to navigate chaos, buffer selfish drivers, and maintain smooth momentum for the collective.

### HUD Metrics
*   **Throughput (v/s)**: Vehicles per Second exiting the bottleneck (Primary metric).
*   **Conflicts**: Total number of collision/braking events (Lower is better).
*   **Avg Wait (s)**: Measure of bottleneck severity (congestion penalty).
*   **Aggressive Ratio (%)**: The percentage of "Red" selfish drivers.

---

## 🧠 The Core Problem & Hypothesis

### The Behavioral Tipping Point
We hypothesized that congestion goes through a non-linear phase transition. 
Through our "Aggression Sweep" experiment, we discovered the **Tipping Point**: at exactly **30% aggressive drivers**, the bottleneck throughput absolutely collapses by ~40%, regardless of the total baseline volume. 

The aggressive trajectory blocks multiple lanes simultaneously, cascading braking waves backward through the system.

---

## ⚙️ Methodology & Architecture

Our architecture is segmented into 4 primary layers, designed to foster rigorous testing:

1. **Custom Gym Environment (`envs/bottleneck_env.py`)** 
   - A normalized 1000-unit continuous space supporting multi-agent states.
   - Designed around a **12-Sensor Suite** for RL observation (Longitude/Latitude, Speed, Leader Info, Lane Occupancy, Global Queue Density).
2. **Behavioral Engine (`agents/behavior_profiles.py`)**
   - Heuristics governing the deterministic Blue and Red agent physics.
3. **RL Pipeline (`training/train_ppo.py`)**
   - Utilizes **Proximal Policy Optimization (PPO)** for high stability.
   - Incorporates a **4-Stage Curriculum Learning** approach:
     - *Stage 1*: Low Traffic (3 v/s), Low Aggression (10%).
     - *Stage 2*: Moderate Density (4 v/s), Mixed Behavior (20%).
     - *Stage 3*: High Density (5 v/s), Realism (30%).
     - *Stage 4*: Stress Testing (6 v/s), High Aggression (40%).
4. **Experimental Analytics (`experiments/run_experiments.py`)**
   - Head-to-head performance benchmarks extracting live metrics to pandas/matplotlib for final validation.

---

## 📊 Key Findings

By introducing a mixed-autonomy system containing **20% RL agents** into a collapsing population (30% aggressive drivers), we secured the following quantifiable gains:

| Scenario | Throughput (v/ep) | Avg Speed (u/s) | Conflict Rate |
| :--- | :--- | :--- | :--- |
| **Mixed Population (Baseline)** | 2548.5 | 2.13 | High |
| **RL-Intervention (20%)** | **2698.6** | **22.99** | **Dramatically Reduced** |

* **Throughput Gain**: ~5.88% increase in capacity.
* **Velocity Gain**: ~7.5% increase in average system speed.
* **Conclusion**: AI agents successfully learned to create buffer gaps ahead of selfish drivers, effectively smoothing the flow. We proved that mixed-autonomy traffic can increase local infrastructure capacity without needing physical concrete expansion.

---

## 🚀 Getting Started

Ensure you have Python 3.9+ installed.

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/yourusername/knowledge_querry_hackathon.git
cd knowledge_querry_hackathon
pip install -r requirements.txt
```

### 2. Verify Environment Array
Ensure the gym environment resolves correctly before running graphics:
```bash
python verify_env.py
```

### 3. Launch Live Simulation
To watch the trained agents interact with cooperative and aggressive drivers in real-time:
```bash
python visualization/pygame_renderer.py
```

### 4. Data Analysis & Training
To run the evaluation analytics matrix:
```bash
jupyter notebook notebooks/analysis.ipynb  # Note: Create this using create_notebook.py if it is not generated yet
```

---

## 📁 Project Structure

```text
├── agents/                  # Deterministic vehicle logic (Aggressive / Cooperative)
├── envs/                    # Custom Gymnasium Bottleneck environment details
├── experiments/             # Scripts to run benchmark testing & sweeping
├── models/                  # Saved artifacts (.zip) of trained Stable-Baselines models
├── training/                # PPO curriculum training scripts 
├── visualization/           # Renderers (PyGame real-time view & Matplotlib)
├── verify_env.py            # Environment unit-test 
├── README.md                # Project abstract and setup
└── requirements.txt         # Core dependencies
```

---
<div align="center">
<i>Built to coordinate the chaos of tomorrow's roads. </i> <br>
<b>ML Research Submission</b>
</div>
