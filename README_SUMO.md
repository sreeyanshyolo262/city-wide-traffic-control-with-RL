# Traffic Signal Control with Reinforcement Learning (SUMO Version)

A Q-Learning agent that learns to control traffic signals at a 4-way intersection using SUMO (Simulation of Urban Mobility).

## Features
- **Realistic Traffic Simulation**: Uses SUMO for accurate traffic modeling
- **Turning Movements**: Left, straight, and right turns for all approaches
- **Save/Load Models**: Train once, demo anytime
- **Visual Feedback**: SUMO-GUI shows real-time traffic flow

## Installation

### 1. Install SUMO
Download and install SUMO from: https://sumo.dlr.de/docs/Downloads.php

For Windows:
- Download the installer from the SUMO website
- Run the installer (typically installs to `C:\Program Files (x86)\Eclipse\Sumo`)

### 2. Set SUMO_HOME Environment Variable

**Windows (Command Prompt):**
```cmd
set SUMO_HOME=C:\Program Files (x86)\Eclipse\Sumo
```

**Windows (PowerShell):**
```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
```

**Permanent (Windows):**
1. Open System Properties → Advanced → Environment Variables
2. Add new System Variable:
   - Name: `SUMO_HOME`
   - Value: `C:\Program Files (x86)\Eclipse\Sumo`

### 3. Install Python Dependencies
```bash
pip install numpy traci sumolib
```

## Usage

### Build the Network (First Time Only)
```bash
python main_sumo.py --build
```

### Train the Agent
```bash
# With SUMO GUI (slower but visual)
python main_sumo.py

# Without GUI (faster training)
python main_sumo.py --no-gui --episodes 100
```

### Demo the Trained Agent
```bash
# Load and run the trained agent
python main_sumo.py --load
```

### Evaluate Performance
```bash
python main_sumo.py --eval
```

## Files

### Python Files
- `main_sumo.py` - Main entry point for SUMO version
- `sumo_env.py` - SUMO environment wrapper
- `agent.py` - Q-Learning agent with save/load
- `config_sumo.py` - Configuration parameters

### SUMO Network Files (in `sumo_files/`)
- `intersection.nod.xml` - Node definitions
- `intersection.edg.xml` - Edge definitions
- `intersection.con.xml` - Connection definitions (turning movements)
- `intersection.rou.xml` - Route and flow definitions
- `intersection.tll.xml` - Traffic light logic
- `intersection.net.xml` - Compiled network (auto-generated)
- `intersection.sumocfg` - SUMO configuration

### Saved Models
- `trained_agent.pkl` - Trained Q-table (binary)
- `trained_agent.json` - Human-readable version

## Demonstrating to Your Teacher

1. **Train the model** (do this before demo):
   ```bash
   python main_sumo.py --no-gui --episodes 100
   ```

2. **Show the trained model**:
   ```bash
   python main_sumo.py --load
   ```
   This will open SUMO-GUI showing the trained agent controlling signals!

## Keyboard Shortcuts (Original Pygame Version)

The original `main.py` still works if you prefer the Pygame visualization:
- `1/2/3` - Set traffic density (Low/Medium/High)
- `Space` - Pause/Resume
- `S` - Save trained agent
- `L` - Load trained agent
- `R` - Reset simulation
- `ESC` - Quit

## How It Works

### State Space
- Queue lengths (bucketed: 0, 1-3, 4-7, 8+) for each direction
- Current signal phase (NS or EW green)

### Action Space
- 0: Give green to North-South
- 1: Give green to East-West

### Reward Function
- Negative reward for waiting time and queue length
- Positive reward for vehicles completing trips

### Learning Algorithm
Q-Learning with epsilon-greedy exploration that decays over time.
