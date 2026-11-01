"""
Q-Learning agent for traffic signal control.
Supports saving/loading trained models.
"""

import os
import json
import random
import pickle
import numpy as np
from collections import defaultdict

# Try to import from config, fallback to config_sumo
try:
    from config import PHASES, ALPHA, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY
except ImportError:
    from config_sumo import PHASES, ALPHA, GAMMA, EPSILON_START, EPSILON_END, EPSILON_DECAY


class QLearningAgent:
    """
    Tabular Q-Learning agent.
    State: (N_bucket, S_bucket, E_bucket, W_bucket, phase_index)
    Actions: 0 = NS_GREEN, 1 = EW_GREEN
    """

    def __init__(self):
        self.q_table = defaultdict(lambda: np.zeros(len(PHASES)))
        self.alpha = ALPHA
        self.gamma = GAMMA
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_END
        self.epsilon_decay = EPSILON_DECAY
        self.total_actions = 0
        self.random_actions = 0

        # Training phase tracking
        self.phase_name = "Observe"
        self.phase_description = "The system does random actions to learn their effect"

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        self.total_actions += 1

        if random.random() < self.epsilon:
            self.random_actions += 1
            action = random.randint(0, len(PHASES) - 1)
        else:
            action = int(np.argmax(self.q_table[state]))

        # Update phase description
        if self.epsilon > 0.7:
            self.phase_name = "Observe"
            self.phase_description = "The system does random actions\nto learn their effect"
        elif self.epsilon > 0.3:
            self.phase_name = "Explore"
            self.phase_description = "The rate of random actions is\nreduced slowly, and the system\ntransitions to training"
        elif self.epsilon > 0.1:
            self.phase_name = "Training"
            self.phase_description = "The system chooses the option\nwith the highest reward, and\nassesses the outcome"
        else:
            self.phase_name = "Trained Model"
            self.phase_description = "Adapts to unexpected situations"

        return action

    def learn(self, state, action, reward, next_state):
        """Q-learning update."""
        current_q = self.q_table[state][action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_stats(self):
        return {
            "epsilon": self.epsilon,
            "states_visited": len(self.q_table),
            "total_actions": self.total_actions,
            "phase_name": self.phase_name,
            "phase_desc": self.phase_description,
        }

    def save(self, filepath="trained_agent.pkl"):
        """Save the trained agent to a file."""
        save_data = {
            "q_table": dict(self.q_table),  # Convert defaultdict to regular dict
            "epsilon": self.epsilon,
            "total_actions": self.total_actions,
            "random_actions": self.random_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"✓ Agent saved to {filepath}")
        print(f"  States learned: {len(self.q_table)}")
        print(f"  Epsilon: {self.epsilon:.4f}")
        
        # Also save human-readable JSON version
        json_path = filepath.replace('.pkl', '.json')
        json_data = {
            "epsilon": self.epsilon,
            "total_actions": self.total_actions,
            "states_learned": len(self.q_table),
            "q_table": {str(k): v.tolist() for k, v in self.q_table.items()}
        }
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"  Human-readable version: {json_path}")

    def load(self, filepath="trained_agent.pkl"):
        """Load a trained agent from a file."""
        if not os.path.exists(filepath):
            print(f"✗ No saved agent found at {filepath}")
            return False
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Restore Q-table as defaultdict
        self.q_table = defaultdict(lambda: np.zeros(len(PHASES)))
        for state, values in save_data["q_table"].items():
            self.q_table[state] = np.array(values)
        
        self.epsilon = save_data.get("epsilon", EPSILON_END)
        self.total_actions = save_data.get("total_actions", 0)
        self.random_actions = save_data.get("random_actions", 0)
        self.alpha = save_data.get("alpha", ALPHA)
        self.gamma = save_data.get("gamma", GAMMA)
        
        # Set to trained mode
        self.phase_name = "Trained Model"
        self.phase_description = "Loaded pre-trained agent\nAdapts to unexpected situations"
        
        print(f"✓ Agent loaded from {filepath}")
        print(f"  States learned: {len(self.q_table)}")
        print(f"  Epsilon: {self.epsilon:.4f}")
        
        return True

    def set_evaluation_mode(self):
        """Set agent to pure exploitation (no exploration)."""
        self.epsilon = 0.0
        self.phase_name = "Evaluation Mode"
        self.phase_description = "Running trained model\n(No exploration)"