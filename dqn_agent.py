"""
Deep Q-Network (DQN) Agent for Multi-Intersection Traffic Control.

This agent uses a neural network to approximate the Q-function, allowing it
to handle more complex, continuous state spaces than a tabular Q-learning
agent. It also uses an experience replay buffer to learn more efficiently.
"""

import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

class DQNAgent:
    def __init__(self, state_size, action_size, agent_id="", hidden_units=(64, 64)):
        self.state_size = state_size
        self.action_size = action_size
        self.agent_id = agent_id
        self.hidden_units = tuple(hidden_units)

        # Experience replay buffer
        self.memory = deque(maxlen=10000)

        # Hyperparameters
        self.gamma = 0.95    # Discount rate
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.998
        self.learning_rate = 0.0005
        self.target_update_every = 200
        self.train_steps = 0
        
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self):
        """Build a neural network for the Q-function."""
        model = Sequential()
        model.add(Input(shape=(self.state_size,)))
        model.add(Dense(self.hidden_units[0], activation='relu'))
        model.add(Dense(self.hidden_units[1], activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss=Huber(), optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def _rebuild_models(self, hidden_units):
        self.hidden_units = tuple(hidden_units)
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())

    @staticmethod
    def _normalize_weights_path(path):
        return path if path.endswith(".weights.h5") else f"{path}.weights.h5"

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Reshape state for prediction
        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        """Train the network on a batch of experiences from memory."""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)

        states = np.vstack([np.reshape(s, [1, self.state_size]) for s, _, _, _, _ in minibatch])
        next_states = np.vstack([np.reshape(ns, [1, self.state_size]) for _, _, _, ns, _ in minibatch])

        current_q = self.model.predict(states, verbose=0)
        next_q_online = self.model.predict(next_states, verbose=0)
        next_q_target = self.target_model.predict(next_states, verbose=0)

        for i, (_, action, reward, _, done) in enumerate(minibatch):
            target = reward
            if not done:
                best_next_action = int(np.argmax(next_q_online[i]))
                target = reward + self.gamma * next_q_target[i][best_next_action]
            current_q[i][action] = target

        self.model.fit(states, current_q, epochs=1, verbose=0)
        self.train_steps += 1

        if self.train_steps % self.target_update_every == 0:
            self.target_model.set_weights(self.model.get_weights())
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    def load(self, name):
        """Load a trained model."""
        normalized = self._normalize_weights_path(name)

        def _try_load_current_arch():
            self.model.load_weights(normalized)
            self.target_model.set_weights(self.model.get_weights())

        try:
            _try_load_current_arch()
            print(f"Agent {self.agent_id} weights loaded successfully from {normalized}")
            return True
        except Exception:
            for candidate_arch in ((24, 24), (64, 64)):
                if tuple(candidate_arch) == tuple(self.hidden_units):
                    continue
                try:
                    self._rebuild_models(candidate_arch)
                    _try_load_current_arch()
                    print(
                        f"Agent {self.agent_id} weights loaded successfully from {normalized} "
                        f"using architecture {candidate_arch}"
                    )
                    return True
                except Exception:
                    continue

            print(f"Could not load weights for agent {self.agent_id} from {name}")
            return False

    def save(self, name):
        """Save the trained model."""
        try:
            normalized = self._normalize_weights_path(name)
            self.model.save_weights(normalized)
            print(f"Agent {self.agent_id} weights saved successfully to {normalized}")
        except Exception as e:
            print(f"Could not save weights for agent {self.agent_id} to {name}: {e}")

    def set_evaluation_mode(self):
        """Set epsilon to 0 for pure exploitation."""
        self.epsilon = 0.0
