"""
SUMO Environment for Multi-Agent RL Traffic Control.

This module provides a wrapper around the SUMO simulation to handle
multiple traffic light agents in a grid network.
"""

import os
import sys
import numpy as np

# Check SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please set SUMO_HOME environment variable")

import traci
from config_sumo import *

class MultiAgentSumoEnv:
    def __init__(self, gui=True, traffic_scale=1.0):
        self.gui = gui
        self.traffic_scale = float(traffic_scale)
        self.config_file = os.path.join(
            os.path.dirname(__file__), 'sumo_files', '4x4', 'intersection.sumocfg')
        self._running = False
        self.tl_ids = []
        self.port = 8813 # Port for TraCI connection
        self.agent_data = {}
        self.sumo_proc = None

    def start(self):
        """Start SUMO simulation."""
        if self._running:
            return
            
        sumo_binary = "sumo-gui" if self.gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", self.config_file,
            "--start",
            "--quit-on-end",
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1",
            "--no-warnings",
            "--step-length", str(SUMO_STEP_LENGTH),
            "--scale", str(self.traffic_scale),
        ]
        
        traci.start(sumo_cmd)
        self._running = True
        
        self.tl_ids = traci.trafficlight.getIDList()
        print(f"Found {len(self.tl_ids)} traffic lights: {self.tl_ids}")

        for tl_id in self.tl_ids:
            incoming = self._get_incoming_lanes(tl_id)
            outgoing = self._get_outgoing_lanes(tl_id)
            if incoming and outgoing:
                self.agent_data[tl_id] = {
                    "incoming_lanes": incoming,
                    "outgoing_lanes": outgoing,
                }

        # Only train on intersections with valid controlled lanes.
        self.tl_ids = sorted(self.agent_data.keys())
        print(f"Using {len(self.tl_ids)} controllable intersections: {self.tl_ids}")

    def close(self):
        """Close SUMO."""
        if self._running:
            traci.close()
            self._running = False

    def reset(self):
        """Reset simulation."""
        self.close()
        self.start()
        return self.get_all_states()

    def step(self, actions=None):
        """Advance simulation. Returns rewards for all agents."""
        if not self._running:
            return {}

        if actions:
            for tl_id, action in actions.items():
                # Assuming 2 phases: 0 for NS green, 1 for EW green
                # This needs to be mapped to specific phase indices in SUMO
                # For a 4-way intersection, phases are often:
                # 0: NS green, 2: EW green
                phase_to_set = action * 2 
                if traci.trafficlight.getPhase(tl_id) != phase_to_set:
                    traci.trafficlight.setPhase(tl_id, phase_to_set)

        traci.simulationStep()
        
        rewards = {tl_id: self._calculate_pressure_reward(tl_id) for tl_id in self.tl_ids}
        return rewards

    def get_all_states(self):
        """Get states for all agents."""
        return {tl_id: self.get_state(tl_id) for tl_id in self.tl_ids}

    def get_state(self, tl_id):
        """
        Get state for a single agent.
        State: [current_phase, norm_queue_in_1, norm_queue_in_2, ..., 
                norm_queue_out_1, norm_queue_out_2, ...]
        """
        state = []
        
        # Use binary phase group (NS-like vs EW-like) instead of raw 0..3 phase index.
        current_phase = traci.trafficlight.getPhase(tl_id)
        phase_group = 0 if current_phase in (0, 1) else 1
        state.append(float(phase_group))
        
        # Normalized queue on incoming lanes
        incoming_lanes = self.agent_data[tl_id]["incoming_lanes"]
        for lane in incoming_lanes:
            queue = traci.lane.getLastStepHaltingNumber(lane)
            state.append(queue / 50.0) # Normalize by a reasonable max queue

        # Normalized queue on outgoing lanes
        outgoing_lanes = self.agent_data[tl_id]["outgoing_lanes"]
        for lane in outgoing_lanes:
            queue = traci.lane.getLastStepHaltingNumber(lane)
            state.append(queue / 50.0)

        return np.array(state)

    def _calculate_pressure_reward(self, tl_id):
        """
        Calculate reward based on 'pressure'.
        Pressure = (sum of queues on incoming lanes) - (sum of queues on outgoing lanes)
        Reward = -abs(Pressure)
        """
        incoming_lanes = self.agent_data[tl_id]["incoming_lanes"]
        outgoing_lanes = self.agent_data[tl_id]["outgoing_lanes"]

        in_queue = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in incoming_lanes)
        out_queue = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in outgoing_lanes)
        in_wait = sum(traci.lane.getWaitingTime(lane) for lane in incoming_lanes)
        in_vehicles = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in incoming_lanes)
        avg_wait = in_wait / max(1.0, float(in_vehicles))
        
        pressure = in_queue - out_queue
        # Reward optimized for the same metrics used in evaluation.
        return -(1.0 * in_queue + 0.5 * abs(pressure) + 0.2 * avg_wait)

    def _get_incoming_lanes(self, tl_id):
        """Get all incoming lanes for a traffic light."""
        lanes = set()
        for link in traci.trafficlight.getControlledLinks(tl_id):
            lanes.add(link[0][0])
        return sorted(lanes)

    def _get_outgoing_lanes(self, tl_id):
        """Get all outgoing lanes for a traffic light."""
        outgoing_lanes = set()
        controlled_links = traci.trafficlight.getControlledLinks(tl_id)
        
        # A controlled link is a list of connections
        for link_group in controlled_links:
            # Each connection is a tuple (fromLane, toLane, viaLane)
            if len(link_group) > 0 and len(link_group[0]) > 1:
                out_lane = link_group[0][1]
                outgoing_lanes.add(out_lane)
            
        return sorted(outgoing_lanes)

    def get_simulation_time(self):
        return traci.simulation.getTime()
