# Adaptive Traffic Signal Control using Reinforcement Learning

**Student:** [Your Name]  
**Date:** March 2026  
**Subject:** [Your Subject / Course]

---

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Proposed Solution](#2-proposed-solution)
3. [Background Theory](#3-background-theory)
4. [System Architecture](#4-system-architecture)
5. [Algorithm Design](#5-algorithm-design)
6. [Implementation](#6-implementation)
7. [Results](#7-results)
8. [Conclusion](#8-conclusion)

---

## 1. Problem Statement

### The Problem with Fixed-Time Traffic Signals

Traditional traffic lights operate on a fixed timer — they give each direction a set amount of green time regardless of actual traffic conditions. This leads to:

- **Wasted green time:** A direction gets green even when no cars are waiting
- **Unnecessary delays:** Cars sit at red while the opposite direction is empty
- **Poor scalability:** Fixed timing cannot respond to rush hours, accidents, or irregular traffic patterns

> **Example:** During off-peak hours, a fixed 30-second cycle still forces cars to wait 30 seconds even if only 1 car is approaching from one direction.

---

### 📸 Screenshot Area — Problem Illustration
> *Insert screenshot here: SUMO-GUI showing fixed-time controller with cars queuing on one side while the other side is empty*

---

### Goal
Design an intelligent agent that **observes real-time traffic conditions** and decides which direction should receive a green light — minimising total vehicle waiting time and maximising throughput.

---

## 2. Proposed Solution

### Reinforcement Learning (Q-Learning)

Train an AI agent to control traffic signals by learning from experience. The agent:
- **Observes** queue lengths at the intersection
- **Decides** which direction gets the green light
- **Receives a reward** based on how well it managed traffic
- **Improves** its decisions over time through trial and error

The simulation environment used is **SUMO (Simulation of Urban Mobility)**, a professional open-source traffic simulator used by researchers worldwide.

---

## 3. Background Theory

### 3.1 Reinforcement Learning

Reinforcement Learning (RL) is a type of machine learning where an **agent** learns to make decisions by interacting with an **environment**. The agent receives **rewards** for good decisions and **penalties** for bad ones.

| Term | Definition | This Project |
|------|-----------|--------------|
| **Agent** | The learner / decision maker | Traffic signal controller |
| **Environment** | The world the agent acts in | SUMO traffic simulation |
| **State** | Current situation | Queue lengths + current phase |
| **Action** | Decision made | NS Green or EW Green |
| **Reward** | Feedback on the action | -waiting time -queue length +throughput |

---

### 3.2 Q-Learning

Q-Learning is a specific RL algorithm that builds a **Q-Table** — a lookup table mapping every (state, action) pair to an expected future reward value called a **Q-value**.

#### The Q-Value Update Rule (Bellman Equation)

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

| Symbol | Name | Value Used | Meaning |
|--------|------|-----------|---------|
| $\alpha$ | Learning rate | 0.1 | How fast to update beliefs |
| $\gamma$ | Discount factor | 0.95 | How much to value future rewards |
| $r$ | Reward | variable | Immediate feedback |
| $s'$ | Next state | variable | Where the agent ended up |

---

### 3.3 Exploration vs Exploitation (Epsilon-Greedy)

A key challenge in RL is balancing:
- **Exploration:** Trying random actions to discover new knowledge
- **Exploitation:** Using learned knowledge to get the best reward

This project uses **Epsilon-Greedy** strategy where epsilon (ε) controls the probability of taking a random action:

| Epsilon Value | Behaviour |
|--------------|-----------|
| ε = 1.0 (start) | 100% random — full exploration |
| ε = 0.5 | 50% random, 50% learned |
| ε = 0.05 (trained) | 5% random, 95% optimal decisions |

Epsilon decays gradually from 1.0 → 0.05 over training (decay rate = 0.9995 per decision).

---

### 📸 Screenshot Area — Training Progress
> *Insert screenshot here: Terminal output showing epsilon decreasing from Episode 1 to Episode 200*

---

## 4. System Architecture

### Components

```
┌─────────────────────────────────────────────────┐
│                   main_sumo.py                  │
│           (Training / Demo entry point)         │
└──────────────────┬──────────────────────────────┘
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
┌─────────────┐         ┌──────────────┐
│   agent.py  │         │  sumo_env.py │
│ Q-Learning  │◄───────►│  SUMO Bridge │
│   Agent     │         │  (TraCI API) │
└─────────────┘         └──────┬───────┘
                               │
                        ┌──────▼───────┐
                        │  SUMO-GUI    │
                        │ (Simulation) │
                        └──────────────┘
```

### Files

| File | Role |
|------|------|
| `agent.py` | Q-Learning algorithm, save/load model |
| `sumo_env.py` | Interface between Python and SUMO via TraCI |
| `config_sumo.py` | All configurable parameters |
| `main_sumo.py` | Training loop and demo runner |
| `compare_demo.py` | Side-by-side comparison with fixed-time controller |
| `trained_agent.pkl` | Saved Q-Table (binary) |
| `trained_agent.json` | Saved Q-Table (human readable) |
| `sumo_files/` | SUMO network definition files (roads, routes, signals) |

---

## 5. Algorithm Design

### 5.1 State Representation

The state captures the current traffic situation as a 3-element tuple:

```
State = (NS_queue_bucket, EW_queue_bucket, current_phase)
```

Queue lengths are **bucketed** (discretised) to keep the state space manageable:

| Bucket | Cars Waiting |
|--------|-------------|
| 0 | 0 (empty) |
| 1 | 1–3 cars (light) |
| 2 | 4–7 cars (medium) |
| 3 | 8+ cars (heavy) |

This gives **4 × 4 × 2 = 32 possible states**. The trained agent encountered **30 of these** during training.

---

### 5.2 Action Space

| Action | Effect |
|--------|--------|
| 0 | Give green to North-South direction |
| 1 | Give green to East-West direction |

---

### 5.3 Reward Function

$$\text{reward} = -\text{waiting\_time} \times 0.1 - \text{queue\_length} \times 0.5 + \text{vehicles\_passed} \times 2$$

| Component | Weight | Purpose |
|-----------|--------|---------|
| Total waiting time | −0.1 | Penalise idle vehicles |
| Total queue length | −0.5 | Penalise traffic build-up |
| Vehicles that completed trip | +2.0 | Reward throughput |

The agent's goal is to make rewards **as close to zero as possible** (less negative = better).

---

### 5.4 Training Loop

```
For each episode (1 hour of simulated traffic = 3600 seconds):
│
├── Reset simulation
├── Set step counter = 0
│
└── Every 5 seconds:
    ├── 1. Get current state (queue lengths, current phase)
    ├── 2. Choose action (epsilon-greedy)
    │       if random() < epsilon → random action (explore)
    │       else                  → best Q-table action (exploit)
    ├── 3. Apply action to traffic light (via TraCI)
    ├── 4. Simulate traffic for 5 seconds
    ├── 5. Calculate reward
    ├── 6. Update Q-Table using Bellman equation
    ├── 7. Decay epsilon by × 0.9995
    └── 8. Repeat until 3600s elapsed

After episode → save model if best reward achieved
```

---

### 📸 Screenshot Area — Training Terminal Output
> *Insert screenshot here: Terminal showing episode numbers, rewards, epsilon values*

---

### 5.5 Realistic Signal Constraints

To make the simulation realistic (as in real traffic engineering):

| Constraint | Value | Reason |
|-----------|-------|--------|
| Minimum green time | 15 seconds | Prevents rapid flickering |
| Maximum green time | 90 seconds | Prevents indefinite blocking |
| Yellow phase | 4 seconds | Driver reaction time |
| Decision interval | Every 5 seconds | Reasonable response time |

---

## 6. Implementation

### 6.1 Training Configuration

| Parameter | Value |
|-----------|-------|
| Episodes | 200 |
| Episode duration | 3600 seconds (1 hour) |
| Total decisions made | 51,120 |
| States learned | 30 |
| Traffic density | Medium (~520 vehicles/hour) |
| Simulation step | 0.1 seconds (smooth animation) |

---

### 6.2 The Q-Table (Trained Model)

After training, the agent stores learned Q-values for each state-action pair. Example entries:

```
State (3, 0, 1) = NS has 8+ cars, EW empty, EW currently green
  → Action 0 (switch to NS): Q = -43.46   ← AGENT CHOOSES THIS
  → Action 1 (keep EW):      Q = -48.35

State (0, 3, 0) = NS empty, EW has 8+ cars, NS currently green
  → Action 0 (keep NS):      Q = -43.83
  → Action 1 (switch to EW): Q = -41.55   ← AGENT CHOOSES THIS
```

The agent correctly learned: **give green to whichever direction has more cars waiting.**

---

### 📸 Screenshot Area — Q-Table JSON
> *Insert screenshot here: Open trained_agent.json showing the Q-values*

---

### 6.3 SUMO Simulation

The simulation uses a **4-way intersection** with:
- 3 lanes per approach (left turn, straight, right turn)
- Coloured vehicles (white, black, red, blue, silver)
- Left-hand traffic (Indian road rules)
- Continuous vehicle flow for 27+ hours (realistic long runs)

---

### 📸 Screenshot Area — SUMO-GUI (RL Agent)
> *Insert screenshot here: SUMO-GUI showing the intersection with vehicles and green signal*

---

### 📸 Screenshot Area — SUMO-GUI (Fixed-Time Comparison)
> *Insert screenshot here: SUMO-GUI showing fixed-time controller with large queues on one side*

---

## 7. Results

### 7.1 Quantitative Comparison

Comparison over a 3-minute simulation run:

| Metric | Fixed-Time (30s cycles) | RL Agent | Improvement |
|--------|------------------------|----------|-------------|
| Average Queue Length | 2.9 vehicles | 2.1 vehicles | **+26.5%** |
| Average Wait Time | 26.2 seconds | 14.2 seconds | **+45.6%** |

---

### 📸 Screenshot Area — Comparison Results
> *Insert screenshot here: Terminal output of compare_demo.py showing the results table*

---

### 7.2 Qualitative Observations

- The RL agent **switches early** when queues begin to build, preventing large backlogs
- The fixed-time controller **ignores empty queues**, wasting green time
- The RL agent shows **adaptive behaviour** — longer green for busy directions
- Both controllers show similar performance during very **light traffic** (queues rarely exceed bucket 1)

---

### 📸 Screenshot Area — Side by Side Comparison
> *Insert screenshot here: Two SUMO windows side-by-side showing RL vs Fixed at the same moment*

---

## 8. Conclusion

### What Was Achieved
- Successfully implemented a Q-Learning agent to control traffic signals
- Trained over 200 episodes (200 simulated hours of traffic)
- Demonstrated measurable improvement over traditional fixed-time control:
  - **26.5% reduction** in average queue length
  - **45.6% reduction** in average wait time

### Limitations
- State space is simplified (bucket-based, not exact counts)
- Only 2 phases (NS/EW) — real intersections can have 4–8 phases
- Trained on uniform traffic density — real traffic varies throughout the day

### Future Improvements
- Use **Deep Q-Network (DQN)** to handle larger, continuous state spaces
- Add **more phases** (left-turn only, pedestrian crossing)
- Train with **varying traffic patterns** (rush hour, night, weekends)
- Deploy on **multiple connected intersections** (corridor optimisation)

---

### 📸 Screenshot Area — Final Demo
> *Insert screenshot here: SUMO-GUI running the final trained agent demo*

---

## References

1. SUMO - Simulation of Urban Mobility: https://sumo.dlr.de
2. Watkins, C.J.C.H. (1989). *Learning from Delayed Rewards*. PhD Thesis, Cambridge University.
3. Sutton, R.S. & Barto, A.G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
4. Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning*. Nature.

---

*Report generated for traffic signal RL project — trafficLightQlearning*
