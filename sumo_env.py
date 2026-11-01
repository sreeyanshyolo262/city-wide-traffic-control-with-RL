"""
SUMO Traffic Environment for RL-based signal control.
Uses TraCI to interface with SUMO simulator.
Handles turning movements (left, straight, right) for all approaches.
"""

import os
import sys
import numpy as np
from collections import defaultdict

# SUMO imports
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
import sumolib

from config_sumo import *


class SumoEnvironment:
    """
    SUMO-based traffic environment with TraCI control.
    Supports 4-way intersection with all turning movements.
    """

    def __init__(self, gui=True, config_file=None):
        self.gui = gui
        self.config_file = config_file or os.path.join(
            os.path.dirname(__file__), 'sumo_files', '4x4', 'intersection.sumocfg')
        
        # Traffic light ID (from SUMO network)
        self.tl_id = "center"
        
        # Incoming edges (where vehicles approach from)
        self.incoming_edges = [
            "south_to_center",  # Northbound
            "north_to_center",  # Southbound  
            "west_to_center",   # Eastbound
            "east_to_center"    # Westbound
        ]
        
        # Phase definitions
        # Phase 0: NS Green, Phase 1: NS Yellow, Phase 2: EW Green, Phase 3: EW Yellow
        self.current_phase = 0
        self.phase_timer = 0
        self.yellow_active = False
        self.time_since_phase_change = 0  # Track time since last phase switch
        
        # Metrics
        self.total_waiting = 0
        self.total_passed = 0
        self.cumulative_wait = 0
        self.step_count = 0
        
        # History for graphs
        self.avg_delay_history = []
        self.queue_history = []
        self.reward_history = []
        
        # Signal timing tracking
        self.last_green_duration = 0
        self.last_red_duration = 0
        self.green_accumulator = 0
        self.red_accumulator = 0
        
        # Traffic density mode
        self.density_mode = "medium"
        
        self._sumo_running = False

    def start(self):
        """Start SUMO simulation."""
        if self._sumo_running:
            self.close()
        
        sumo_binary = "sumo-gui" if self.gui else "sumo"
        
        sumo_cmd = [
            sumo_binary,
            "-c", self.config_file,
            "--start",
            "--quit-on-end",
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1",
            "--no-warnings",
            "--step-length", "0.1",  # Smaller steps for smooth animation
            "--delay", "50" if self.gui else "0",  # 50ms delay for GUI, 0 for training
        ]
        
        traci.start(sumo_cmd)
        self._sumo_running = True
        
        # Get initial traffic light info
        self._init_tl_info()
        
    def _init_tl_info(self):
        """Initialize traffic light information."""
        # Get all traffic light IDs
        self.tl_ids = traci.trafficlight.getIDList()
        if self.tl_ids:
            self.tl_id = self.tl_ids[0]
        
    def close(self):
        """Close SUMO simulation."""
        if self._sumo_running:
            traci.close()
            self._sumo_running = False

    def reset(self):
        """Reset the simulation."""
        self.close()
        self.step_count = 0
        self.total_waiting = 0
        self.total_passed = 0
        self.cumulative_wait = 0
        self.avg_delay_history = []
        self.queue_history = []
        self.reward_history = []
        self.green_accumulator = 0
        self.red_accumulator = 0
        self.time_since_phase_change = SUMO_MIN_GREEN  # Start ready to switch
        self.start()
        return self.get_state()

    def get_state(self):
        """
        Returns discretized state: (ns_queue_bucket, ew_queue_bucket, phase_index)
        Queues are bucketed: 0, 1-3, 4-7, 8+
        """
        def bucket(n):
            if n == 0: return 0
            if n <= 3: return 1
            if n <= 7: return 2
            return 3
        
        # Get queue lengths for NS and EW
        ns_queue = self._get_queue_length("south_to_center") + self._get_queue_length("north_to_center")
        ew_queue = self._get_queue_length("west_to_center") + self._get_queue_length("east_to_center")
        
        # Current phase (0 = NS_GREEN, 1 = EW_GREEN)
        current_tl_phase = traci.trafficlight.getPhase(self.tl_id) if self._sumo_running else 0
        phase_idx = 0 if current_tl_phase in [0, 1] else 1  # 0,1 = NS phases, 2,3 = EW phases
        
        return (bucket(ns_queue), bucket(ew_queue), phase_idx)

    def _get_queue_length(self, edge_id):
        """Get number of halted vehicles on an edge."""
        try:
            return traci.edge.getLastStepHaltingNumber(edge_id)
        except:
            return 0

    def _get_waiting_time(self, edge_id):
        """Get total waiting time on an edge."""
        try:
            return traci.edge.getWaitingTime(edge_id)
        except:
            return 0

    def get_queue_lengths(self):
        """Raw queue lengths per direction."""
        if not self._sumo_running:
            return {"North": 0, "South": 0, "East": 0, "West": 0}
        
        return {
            "North": self._get_queue_length("south_to_center"),  # Vehicles going north
            "South": self._get_queue_length("north_to_center"),  # Vehicles going south
            "East": self._get_queue_length("west_to_center"),    # Vehicles going east
            "West": self._get_queue_length("east_to_center")     # Vehicles going west
        }

    def step(self, action=None):
        """
        Advance simulation by one step.
        action: 0 = NS_GREEN, 1 = EW_GREEN (or None to keep current)
        Returns: reward
        """
        if not self._sumo_running:
            return 0
            
        self.step_count += 1
        
        # Handle action (phase switch)
        if action is not None:
            self._set_phase(action)
        
        # Advance SUMO simulation
        traci.simulationStep()
        
        # Update phase timer
        self.time_since_phase_change += SUMO_STEP_LENGTH
        
        # Calculate metrics
        reward = self._calculate_reward()
        
        # Update history
        self._update_history()
        
        # Update timing accumulators
        current_phase = traci.trafficlight.getPhase(self.tl_id)
        if current_phase in [0, 1]:  # NS phase
            self.green_accumulator += SUMO_STEP_LENGTH
        else:
            self.red_accumulator += SUMO_STEP_LENGTH
        
        return reward

    def _set_phase(self, action):
        """Set traffic light phase based on action with minimum green time enforcement."""
        current_phase = traci.trafficlight.getPhase(self.tl_id)
        
        # Determine if we're requesting a phase change
        current_action = 0 if current_phase in [0, 1] else 1  # NS phases = action 0, EW phases = action 1
        
        # Only allow phase change if minimum green time has passed
        if action != current_action:
            if self.time_since_phase_change < SUMO_MIN_GREEN:
                # Ignore switch request - minimum green not met
                return
        
        # Phase 0: NS Green, Phase 1: NS Yellow, Phase 2: EW Green, Phase 3: EW Yellow
        if action == 0:  # NS_GREEN
            if current_phase not in [0]:  # Not already NS Green
                traci.trafficlight.setPhase(self.tl_id, 0)  # Set to NS Green
                self.last_red_duration = self.red_accumulator
                self.red_accumulator = 0
                self.time_since_phase_change = 0  # Reset timer on phase change
        elif action == 1:  # EW_GREEN
            if current_phase not in [2]:  # Not already EW Green
                traci.trafficlight.setPhase(self.tl_id, 2)  # Set to EW Green
                self.last_green_duration = self.green_accumulator
                self.green_accumulator = 0
                self.time_since_phase_change = 0  # Reset timer on phase change

    def _calculate_reward(self):
        """Calculate reward based on queue lengths and waiting times."""
        total_waiting_time = 0
        total_queue = 0
        
        for edge in self.incoming_edges:
            total_waiting_time += self._get_waiting_time(edge)
            total_queue += self._get_queue_length(edge)
        
        # Negative reward for waiting (encourage minimizing delay)
        reward = -total_waiting_time * 0.1 - total_queue * 0.5
        
        # Bonus for throughput
        try:
            departed = traci.simulation.getDepartedNumber()
            arrived = traci.simulation.getArrivedNumber()
            reward += arrived * 2  # Reward for vehicles completing their trips
        except:
            pass
        
        self.reward_history.append(reward)
        
        return reward

    def _update_history(self):
        """Update metrics history for dashboard."""
        queues = self.get_queue_lengths()
        total_queue = sum(queues.values())
        self.queue_history.append(total_queue)
        
        # Calculate average delay
        total_waiting = sum(self._get_waiting_time(e) for e in self.incoming_edges)
        
        # Get number of vehicles in simulation
        try:
            num_vehicles = traci.vehicle.getIDCount()
            avg_delay = total_waiting / max(num_vehicles, 1)
        except:
            avg_delay = 0
        
        self.avg_delay_history.append(avg_delay)
        
        # Keep only last N entries
        max_history = 500
        if len(self.avg_delay_history) > max_history:
            self.avg_delay_history = self.avg_delay_history[-max_history:]
        if len(self.queue_history) > max_history:
            self.queue_history = self.queue_history[-max_history:]
        if len(self.reward_history) > max_history:
            self.reward_history = self.reward_history[-max_history:]

    def get_signal_state(self):
        """Get current signal state for each direction."""
        if not self._sumo_running:
            return {"North": "red", "South": "red", "East": "red", "West": "red"}
        
        try:
            phase = traci.trafficlight.getPhase(self.tl_id)
        except:
            phase = 0
        
        if phase == 0:  # NS Green
            return {"North": "green", "South": "green", "East": "red", "West": "red"}
        elif phase == 1:  # NS Yellow
            return {"North": "yellow", "South": "yellow", "East": "red", "West": "red"}
        elif phase == 2:  # EW Green
            return {"North": "red", "South": "red", "East": "green", "West": "green"}
        else:  # EW Yellow
            return {"North": "red", "South": "red", "East": "yellow", "West": "yellow"}

    def set_traffic_density(self, level):
        """Set traffic density level: 'low', 'medium', 'high'."""
        self.density_mode = level
        # Note: Density is controlled via route file probabilities
        # For dynamic control, would need to modify vehicle insertion rates via TraCI

    def get_simulation_time(self):
        """Get current simulation time in seconds."""
        try:
            return traci.simulation.getTime()
        except:
            return 0


def build_network():
    """Build SUMO network from node/edge files."""
    sumo_files_dir = os.path.join(os.path.dirname(__file__), 'sumo_files')
    
    netconvert_cmd = [
        "netconvert",
        "--node-files", os.path.join(sumo_files_dir, "intersection.nod.xml"),
        "--edge-files", os.path.join(sumo_files_dir, "intersection.edg.xml"),
        "--connection-files", os.path.join(sumo_files_dir, "intersection.con.xml"),
        "--tllogic-files", os.path.join(sumo_files_dir, "intersection.tll.xml"),
        "--output-file", os.path.join(sumo_files_dir, "intersection.net.xml"),
        "--no-turnarounds",
        "--lefthand",  # Left-hand traffic (India/UK style)
    ]
    
    import subprocess
    result = subprocess.run(netconvert_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error building network: {result.stderr}")
        return False
    
    print("Network built successfully!")
    return True


if __name__ == "__main__":
    # Build network when run directly
    build_network()
