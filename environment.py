"""
Traffic junction simulation environment for RL.
Simulates vehicle arrivals, queues, and signal-controlled flow.
"""

import random
from config import *


class Vehicle:
    """Represents a single vehicle in the simulation."""

    def __init__(self, direction, lane_offset, spawn_distance):
        self.direction = direction
        self.lane_offset = lane_offset  # Lateral offset in lane
        self.position = spawn_distance  # Distance from stop line (positive = before stop line)
        self.color = random.choice(CAR_COLORS)
        self.waiting = False            # Waiting at stop line
        self.approaching = True         # Driving toward intersection
        self.passed = False             # Has passed through intersection
        self.wait_time = 0              # Frames spent waiting
        self.speed = CAR_SPEED          # Start moving immediately

    def get_rect_coords(self):
        """Return (x, y, width, height, angle) for rendering."""
        # Position is distance from stop line (positive = before, negative = past)
        stop_line_offset = ROAD_WIDTH // 2 + 5
        
        if self.direction == "North":
            # Coming from bottom, going up
            x = CENTER_X + MEDIAN_WIDTH // 2 + self.lane_offset
            y = CENTER_Y + stop_line_offset + self.position
            return (x, y, CAR_WIDTH, CAR_LENGTH, 0)
        elif self.direction == "South":
            # Coming from top, going down
            x = CENTER_X - MEDIAN_WIDTH // 2 - CAR_WIDTH - self.lane_offset
            y = CENTER_Y - stop_line_offset - self.position - CAR_LENGTH
            return (x, y, CAR_WIDTH, CAR_LENGTH, 0)
        elif self.direction == "East":
            # Coming from left, going right
            x = CENTER_X - stop_line_offset - self.position - CAR_LENGTH
            y = CENTER_Y + MEDIAN_WIDTH // 2 + self.lane_offset
            return (x, y, CAR_LENGTH, CAR_WIDTH, 0)
        elif self.direction == "West":
            # Coming from right, going left
            x = CENTER_X + stop_line_offset + self.position
            y = CENTER_Y - MEDIAN_WIDTH // 2 - CAR_WIDTH - self.lane_offset
            return (x, y, CAR_LENGTH, CAR_WIDTH, 0)


class TrafficEnvironment:
    """
    Simulates a 4-way intersection with vehicle arrivals and signal control.
    """

    def __init__(self):
        self.vehicles = {d: [] for d in DIRECTIONS}
        self.current_phase = "NS_GREEN"  # Start with North-South green
        self.phase_timer = PHASE_DURATION
        self.yellow_active = False
        self.yellow_timer = 0

        self.arrival_rate = ARRIVAL_RATE_MEDIUM  # Default
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

    def get_state(self):
        """
        Returns discretized state: (n_queue, s_queue, e_queue, w_queue, phase_index)
        Queues are bucketed: 0, 1-3, 4-7, 8+
        """
        def bucket(n):
            if n == 0: return 0
            elif n <= 3: return 1
            elif n <= 7: return 2
            else: return 3

        queues = [len([v for v in self.vehicles[d] if v.waiting]) for d in DIRECTIONS]
        phase_idx = PHASES.index(self.current_phase)
        return (bucket(queues[0]), bucket(queues[1]),
                bucket(queues[2]), bucket(queues[3]), phase_idx)

    def get_queue_lengths(self):
        """Raw queue lengths per direction."""
        return {d: len([v for v in self.vehicles[d] if v.waiting]) for d in DIRECTIONS}

    def step(self, action=None):
        """
        Advance simulation by one frame.
        action: 0 = NS_GREEN, 1 = EW_GREEN (or None to keep current)
        Returns: reward
        """
        self.step_count += 1

        # ─── Handle action (phase switch) ───
        if action is not None and not self.yellow_active:
            desired_phase = PHASES[action]
            if desired_phase != self.current_phase:
                self.yellow_active = True
                self.yellow_timer = YELLOW_DURATION
                self._next_phase = desired_phase
                # Record durations
                self.last_green_duration = self.green_accumulator / FPS
                self.green_accumulator = 0

        # ─── Yellow transition ───
        if self.yellow_active:
            self.yellow_timer -= 1
            self.red_accumulator += 1
            if self.yellow_timer <= 0:
                self.yellow_active = False
                self.current_phase = self._next_phase
                self.last_red_duration = self.red_accumulator / FPS
                self.red_accumulator = 0
        else:
            self.green_accumulator += 1

        # ─── Determine which directions are green ───
        if self.yellow_active:
            green_dirs = []  # All red during yellow
        elif self.current_phase == "NS_GREEN":
            green_dirs = ["North", "South"]
        else:
            green_dirs = ["East", "West"]

        # ─── Spawn new vehicles at screen edges ───
        for d in DIRECTIONS:
            if random.random() < self.arrival_rate:
                lane_off = random.randint(2, LANE_WIDTH - CAR_WIDTH - 2)
                # Spawn at edge of screen
                spawn_distance = SPAWN_DISTANCE + random.randint(0, 50)
                self.vehicles[d].append(Vehicle(d, lane_off, spawn_distance))

        # ─── Update vehicles ───
        total_wait = 0
        for d in DIRECTIONS:
            is_green = d in green_dirs
            to_remove = []
            
            # Sort vehicles by position (closest to intersection first)
            sorted_vehicles = sorted(self.vehicles[d], key=lambda v: v.position)
            
            for i, v in enumerate(sorted_vehicles):
                if v.passed:
                    # Move through and past intersection
                    v.position -= CAR_SPEED * 2
                    if v.position < -DESPAWN_DISTANCE:
                        to_remove.append(v)
                        
                elif v.approaching:
                    # Find the vehicle ahead (if any)
                    vehicle_ahead = None
                    for other in sorted_vehicles:
                        if other is not v and other.position < v.position and not other.passed:
                            if vehicle_ahead is None or other.position > vehicle_ahead.position:
                                vehicle_ahead = other
                    
                    # Calculate stopping distance
                    if vehicle_ahead:
                        safe_distance = vehicle_ahead.position + CAR_LENGTH + CAR_GAP
                    else:
                        safe_distance = 0  # Stop at the stop line
                    
                    # Check if we should stop
                    if v.position <= safe_distance + CAR_SPEED:
                        # We're at the queue
                        if is_green and v.position <= CAR_SPEED:
                            # Green light and at stop line - go through
                            v.approaching = False
                            v.passed = True
                            v.speed = CAR_SPEED * 2
                            self.total_passed += 1
                            self.cumulative_wait += v.wait_time
                        elif is_green and vehicle_ahead and vehicle_ahead.passed:
                            # Vehicle ahead is moving through, follow it
                            v.position -= CAR_SPEED
                        elif v.position > safe_distance:
                            # Move up to safe distance
                            v.position = max(safe_distance, v.position - CAR_SPEED)
                        else:
                            # Waiting in queue
                            v.waiting = True
                            v.wait_time += 1
                            total_wait += 1
                    else:
                        # Keep approaching
                        v.position -= CAR_SPEED
                        v.waiting = False
                        
                elif v.waiting:
                    # Find the vehicle ahead
                    vehicle_ahead = None
                    for other in sorted_vehicles:
                        if other is not v and other.position < v.position and not other.passed:
                            if vehicle_ahead is None or other.position > vehicle_ahead.position:
                                vehicle_ahead = other
                    
                    if is_green:
                        if vehicle_ahead is None or vehicle_ahead.passed:
                            # Clear to go
                            v.waiting = False
                            v.approaching = False
                            v.passed = True
                            v.speed = CAR_SPEED * 2
                            self.total_passed += 1
                            self.cumulative_wait += v.wait_time
                        elif vehicle_ahead.position < v.position - CAR_LENGTH - CAR_GAP:
                            # Vehicle ahead moved, follow it
                            v.position -= CAR_SPEED
                    else:
                        v.wait_time += 1
                        total_wait += 1

            for v in to_remove:
                self.vehicles[d].remove(v)

        self.total_waiting = total_wait

        # ─── Compute reward ───
        reward = -total_wait  # Negative of total waiting vehicles

        # ─── Record history (every ~1 simulated second) ───
        if self.step_count % FPS == 0:
            avg_delay = 0
            if self.total_passed > 0:
                avg_delay = (self.cumulative_wait / self.total_passed) / FPS  # In seconds
            self.avg_delay_history.append(avg_delay)
            queues = self.get_queue_lengths()
            self.queue_history.append(queues)
            self.reward_history.append(reward)

        return reward

    def set_arrival_rate(self, rate):
        self.arrival_rate = rate

    def get_signal_state(self):
        """Returns (phase, is_yellow, green_dirs)"""
        if self.yellow_active:
            return self.current_phase, True, []
        elif self.current_phase == "NS_GREEN":
            return self.current_phase, False, ["North", "South"]
        else:
            return self.current_phase, False, ["East", "West"]