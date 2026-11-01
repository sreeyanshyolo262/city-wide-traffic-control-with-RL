"""
Configuration for SUMO-based Traffic Signal RL system.
"""

# ─── SUMO Settings ───────────────────────────────────────────────
SUMO_STEP_LENGTH = 0.1  # Simulation step in seconds (smooth animation)
SUMO_YELLOW_TIME = 4    # Yellow phase duration in seconds (realistic)
SUMO_MIN_GREEN = 15     # Minimum green time in seconds (real: 15-30s)
SUMO_MAX_GREEN = 90     # Maximum green time in seconds (real: 60-120s)

# ─── RL Decision Interval ────────────────────────────────────────
DECISION_INTERVAL = 5   # Make RL decision every N seconds

# ─── RL Parameters ───────────────────────────────────────────────
ALPHA = 0.1             # Learning rate
GAMMA = 0.95            # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9995  # Per-decision decay

# ─── Training Settings ───────────────────────────────────────────
TRAINING_EPISODES = 200
STEPS_PER_EPISODE = 3600  # 1 hour of simulation

# ─── Signal Phases ───────────────────────────────────────────────
PHASES = ["NS_GREEN", "EW_GREEN"]
DIRECTIONS = ["North", "South", "East", "West"]

# ─── Dashboard Colors ────────────────────────────────────────────
COLOR_DASH_BG = (25, 25, 30)
COLOR_DASH_TEXT = (220, 220, 220)
COLOR_DASH_ACCENT = (80, 180, 255)
COLOR_DASH_GREEN = (0, 200, 80)
COLOR_DASH_RED = (220, 50, 50)
COLOR_DASH_YELLOW = (220, 200, 60)
COLOR_DASH_GRID = (45, 45, 55)

# ─── Window (for optional Pygame dashboard) ──────────────────────
DASH_WIDTH = 400
DASH_HEIGHT = 600
