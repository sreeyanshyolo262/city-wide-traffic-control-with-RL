"""
Configuration constants for the Traffic Signal RL system.
"""

# ─── Window Layout ───────────────────────────────────────────────
WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 800
SIM_WIDTH = 850          # Left panel: intersection simulation
DASH_WIDTH = 550         # Right panel: dashboard
FPS = 30

# ─── Intersection Geometry ───────────────────────────────────────
# Center of the intersection in the sim panel
CENTER_X = SIM_WIDTH // 2
CENTER_Y = WINDOW_HEIGHT // 2

ROAD_WIDTH = 80          # Width of each road (pixels)
LANE_WIDTH = 40          # Each lane is half the road
MEDIAN_WIDTH = 4         # Center divider

# ─── Vehicle ─────────────────────────────────────────────────────
CAR_LENGTH = 22
CAR_WIDTH = 12
CAR_GAP = 6              # Gap between queued cars
CAR_SPEED = 4            # Pixels per frame when moving (increased for smoother flow)

# Spawn/Despawn distances from stop line
SPAWN_DISTANCE = 350     # Distance from stop line where cars spawn
DESPAWN_DISTANCE = 400   # Distance past intersection where cars are removed

# ─── Colors (Realistic palette) ──────────────────────────────────
COLOR_GRASS = (74, 105, 60)
COLOR_ROAD = (62, 62, 62)
COLOR_ROAD_DARK = (50, 50, 50)
COLOR_SIDEWALK = (140, 140, 130)
COLOR_LANE_MARKING = (200, 200, 200)
COLOR_LANE_MARKING_YELLOW = (210, 190, 80)
COLOR_STOP_LINE = (220, 220, 220)
COLOR_CROSSWALK = (200, 200, 200)

COLOR_GREEN_LIGHT = (0, 220, 0)
COLOR_RED_LIGHT = (220, 0, 0)
COLOR_YELLOW_LIGHT = (220, 200, 0)
COLOR_LIGHT_OFF = (60, 60, 60)

# Car colors — realistic vehicle palette
CAR_COLORS = [
    (220, 220, 225),   # White / Silver
    (30, 30, 30),      # Black
    (45, 45, 48),      # Dark gray
    (140, 140, 145),   # Gray
    (22, 56, 130),     # Dark blue
    (180, 30, 30),     # Red
    (30, 90, 50),      # Dark green
    (200, 170, 80),    # Gold / tan
    (80, 30, 20),      # Maroon
    (100, 130, 180),   # Light blue
]

# ─── Dashboard Colors ────────────────────────────────────────────
COLOR_DASH_BG = (25, 25, 30)
COLOR_DASH_TEXT = (220, 220, 220)
COLOR_DASH_ACCENT = (80, 180, 255)
COLOR_DASH_GREEN = (0, 200, 80)
COLOR_DASH_RED = (220, 50, 50)
COLOR_DASH_YELLOW = (220, 200, 60)
COLOR_DASH_GRID = (45, 45, 55)

# ─── RL Parameters ───────────────────────────────────────────────
ALPHA = 0.15             # Learning rate (increased for faster learning)
GAMMA = 0.95             # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995    # Faster decay for quicker training

# ─── Simulation ──────────────────────────────────────────────────
PHASE_DURATION = 60      # Frames per signal phase (2 seconds at 30fps)
YELLOW_DURATION = 20     # Yellow transition frames (shorter)

DIRECTIONS = ["North", "South", "East", "West"]
PHASES = ["NS_GREEN", "EW_GREEN"]  # Two phases

# Arrival rates (probability per frame of a new car appearing)
ARRIVAL_RATE_LOW = 0.015
ARRIVAL_RATE_MEDIUM = 0.04
ARRIVAL_RATE_HIGH = 0.08