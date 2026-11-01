"""
Traffic Signal Control with Reinforcement Learning
Hybrid Mode: SUMO simulation + Pygame stats dashboard

SUMO shows the realistic traffic view
Pygame shows the dashboard with graphs and training stats

Controls:
  1 / 2 / 3 — Set traffic density (Low / Medium / High)
  SPACE      — Pause / Resume
  S          — Save trained agent
  L          — Load trained agent
  R          — Reset simulation
  ESC        — Quit
"""

import os
import sys
import time
import pygame
import argparse
import numpy as np
from collections import deque

# Check SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please set SUMO_HOME environment variable")

import traci

from config_sumo import *
from agent import QLearningAgent


class SumoSimulation:
    """Manages SUMO traffic simulation via TraCI."""
    
    # Realistic timing constraints (in seconds)
    MIN_GREEN_TIME = 15.0   # Minimum green before switching (real: 15-30s)
    MAX_GREEN_TIME = 90.0   # Maximum green time (real: 60-120s)
    YELLOW_TIME = 4.0       # Yellow phase duration
    STEP_LENGTH = 1.0       # Simulation step in seconds
    
    def __init__(self, gui=True):
        self.gui = gui
        self.config_file = os.path.join(
            os.path.dirname(__file__), 'sumo_files', '4x4', 'intersection.sumocfg')
        self.tl_id = "center"
        self.incoming_edges = [
            "south_to_center", "north_to_center",
            "west_to_center", "east_to_center"
        ]
        self._running = False
        
        # Metrics history
        self.delay_history = deque(maxlen=200)
        self.queue_history = deque(maxlen=200)
        self.reward_history = deque(maxlen=200)
        
        # Timing - realistic tracking
        self.green_time = 0           # Current NS green accumulator
        self.red_time = 0             # Current EW green accumulator  
        self.last_green_duration = 0
        self.last_red_duration = 0
        self.current_phase = 0
        self.current_phase_time = 0   # Time in current phase
        self.phase_just_changed = False
        
        # Counters
        self.step_count = 0
        self.sim_hours = 0
        
    def start(self):
        """Start SUMO simulation."""
        if self._running:
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
            "--step-length", str(self.STEP_LENGTH),
            "--delay", "100",  # 100ms delay between steps - slower visualization
        ]
        
        traci.start(sumo_cmd)
        self._running = True
        
        # Get TL ID
        tl_ids = traci.trafficlight.getIDList()
        if tl_ids:
            self.tl_id = tl_ids[0]
            
    def close(self):
        """Close SUMO."""
        if self._running:
            try:
                traci.close()
            except:
                pass
            self._running = False
            
    def reset(self):
        """Reset simulation."""
        self.close()
        self.step_count = 0
        self.sim_hours = 0
        self.current_phase_time = 0
        self.green_time = 0
        self.red_time = 0
        self.delay_history.clear()
        self.queue_history.clear()
        self.reward_history.clear()
        self.green_time = 0
        self.red_time = 0
        time.sleep(0.5)
        self.start()
        return self.get_state()
        
    def get_state(self):
        """Get discretized state for RL."""
        def bucket(n):
            if n == 0: return 0
            if n <= 3: return 1
            if n <= 7: return 2
            return 3
            
        ns_queue = self._get_queue("south_to_center") + self._get_queue("north_to_center")
        ew_queue = self._get_queue("west_to_center") + self._get_queue("east_to_center")
        
        try:
            phase = traci.trafficlight.getPhase(self.tl_id)
            phase_idx = 0 if phase in [0, 1] else 1
        except:
            phase_idx = 0
            
        return (bucket(ns_queue), bucket(ew_queue), phase_idx)
        
    def _get_queue(self, edge):
        try:
            return traci.edge.getLastStepHaltingNumber(edge)
        except:
            return 0
            
    def _get_wait_time(self, edge):
        try:
            return traci.edge.getWaitingTime(edge)
        except:
            return 0
            
    def get_queues(self):
        """Get queue lengths per direction."""
        return {
            "NB": self._get_queue("south_to_center"),
            "SB": self._get_queue("north_to_center"),
            "EB": self._get_queue("west_to_center"),
            "WB": self._get_queue("east_to_center"),
        }
        
    def step(self, action=None):
        """Advance simulation. Returns reward."""
        if not self._running:
            return 0
            
        self.step_count += 1
        self.phase_just_changed = False
        
        # Get current phase info
        try:
            current = traci.trafficlight.getPhase(self.tl_id)
        except:
            current = 0
        
        # Apply action ONLY if minimum green time has passed
        # Phase 0: NS Green, Phase 1: NS Yellow, Phase 2: EW Green, Phase 3: EW Yellow
        if action is not None:
            try:
                # Check if we can switch (minimum green time elapsed)
                can_switch = self.current_phase_time >= self.MIN_GREEN_TIME
                
                # Also force switch if max green time exceeded
                force_switch = self.current_phase_time >= self.MAX_GREEN_TIME
                
                if action == 0 and current in [2, 3]:  # Want NS Green, currently EW
                    if can_switch or force_switch:
                        # Go through yellow first if currently green
                        if current == 2:
                            traci.trafficlight.setPhase(self.tl_id, 3)  # EW Yellow
                        else:
                            traci.trafficlight.setPhase(self.tl_id, 0)  # NS Green
                        self.last_red_duration = self.red_time
                        self.red_time = 0
                        self.current_phase_time = 0
                        self.phase_just_changed = True
                        
                elif action == 1 and current in [0, 1]:  # Want EW Green, currently NS
                    if can_switch or force_switch:
                        if current == 0:
                            traci.trafficlight.setPhase(self.tl_id, 1)  # NS Yellow
                        else:
                            traci.trafficlight.setPhase(self.tl_id, 2)  # EW Green
                        self.last_green_duration = self.green_time
                        self.green_time = 0
                        self.current_phase_time = 0
                        self.phase_just_changed = True
            except:
                pass
                
        # Simulate
        try:
            traci.simulationStep()
        except Exception as e:
            self._running = False
            raise e
            
        # Update timing
        try:
            phase = traci.trafficlight.getPhase(self.tl_id)
            self.current_phase_time += self.STEP_LENGTH
            
            if phase in [0, 1]:  # NS active
                self.green_time += self.STEP_LENGTH
            else:  # EW active
                self.red_time += self.STEP_LENGTH
            self.current_phase = phase
        except:
            pass
            
        # Calculate reward
        reward = self._calc_reward()
        self._update_history()
        
        # Update sim time
        try:
            self.sim_hours = traci.simulation.getTime() / 3600.0
        except:
            pass
            
        return reward
        
    def _calc_reward(self):
        """Calculate reward."""
        total_wait = sum(self._get_wait_time(e) for e in self.incoming_edges)
        total_queue = sum(self._get_queue(e) for e in self.incoming_edges)
        
        try:
            arrived = traci.simulation.getArrivedNumber()
        except:
            arrived = 0
            
        reward = -total_wait * 0.1 - total_queue * 0.5 + arrived * 2
        return reward
        
    def _update_history(self):
        """Update history for graphs."""
        queues = self.get_queues()
        total_queue = sum(queues.values())
        self.queue_history.append(total_queue)
        
        total_wait = sum(self._get_wait_time(e) for e in self.incoming_edges)
        try:
            num_veh = traci.vehicle.getIDCount()
            avg_delay = (total_wait / max(num_veh, 1)) / 60.0  # Convert to minutes
        except:
            avg_delay = 0
        self.delay_history.append(avg_delay)
        
    def is_running(self):
        return self._running


class StatsDashboard:
    """Pygame window showing real-time stats."""
    
    def __init__(self, width=500, height=700):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Traffic RL Dashboard - Stats & Training Progress")
        
        # Fonts
        self.font_title = pygame.font.SysFont("Arial", 28, bold=True)
        self.font_large = pygame.font.SysFont("Arial", 48, bold=True)
        self.font_med = pygame.font.SysFont("Arial", 20)
        self.font_small = pygame.font.SysFont("Arial", 16)
        
        # Colors
        self.bg = (25, 25, 30)
        self.text = (220, 220, 220)
        self.accent = (80, 180, 255)
        self.green = (0, 200, 80)
        self.red = (220, 50, 50)
        self.yellow = (220, 200, 60)
        self.grid = (45, 45, 55)
        
    def draw(self, sim, agent):
        """Draw the dashboard."""
        self.screen.fill(self.bg)
        y = 20
        
        # Title
        title = self.font_title.render("Corridor Signal Timing", True, self.text)
        self.screen.blit(title, (self.width//2 - title.get_width()//2, y))
        y += 45
        
        # Timing bar
        y = self._draw_timing_bar(sim, y)
        y += 20
        
        # Last green/red durations
        green_txt = self.font_med.render(f"Corridor Last Green: {sim.last_green_duration:.1f}", True, self.text)
        self.screen.blit(green_txt, (self.width//2 - green_txt.get_width()//2, y))
        y += 25
        
        red_txt = self.font_med.render(f"Corridor Last Red: {sim.last_red_duration:.1f}", True, self.text)
        self.screen.blit(red_txt, (self.width//2 - red_txt.get_width()//2, y))
        y += 40
        
        # Average Delay Graph
        title2 = self.font_title.render("System Average Delay", True, self.text)
        self.screen.blit(title2, (self.width//2 - title2.get_width()//2, y))
        y += 35
        
        y = self._draw_delay_graph(sim, y)
        y += 20
        
        # Queue stats
        queues = sim.get_queues()
        for direction, label in [("EB", "EB Queue"), ("WB", "WB Queue"), 
                                  ("NB", "NB Queue"), ("SB", "SB Queue")]:
            txt = self.font_med.render(f"{label}: {queues[direction]}", True, self.text)
            self.screen.blit(txt, (self.width//2 - txt.get_width()//2, y))
            y += 28
            
        y += 20
        
        # Training phase indicator
        stats = agent.get_stats()
        
        # Phase name (large)
        phase_color = self.green if stats["phase_name"] == "Trained Model" else self.accent
        phase_txt = self.font_large.render(stats["phase_name"], True, phase_color)
        self.screen.blit(phase_txt, (self.width//2 - phase_txt.get_width()//2, y))
        y += 60
        
        # Phase description
        for line in stats["phase_desc"].split("\n"):
            desc = self.font_small.render(line, True, (180, 180, 180))
            self.screen.blit(desc, (self.width//2 - desc.get_width()//2, y))
            y += 22
            
        y += 20
        
        # Progress bar (epsilon)
        self._draw_progress_bar(agent, y)
        y += 40
        
        # Stats
        eps_txt = self.font_small.render(f"Epsilon: {stats['epsilon']:.4f}  |  States: {stats['states_visited']}", True, (150, 150, 150))
        self.screen.blit(eps_txt, (self.width//2 - eps_txt.get_width()//2, y))
        
        pygame.display.flip()
        
    def _draw_timing_bar(self, sim, y):
        """Draw signal timing bar."""
        bar_x = 30
        bar_w = self.width - 60
        bar_h = 25
        
        # Background
        pygame.draw.rect(self.screen, self.grid, (bar_x, y, bar_w, bar_h))
        
        # Calculate proportions
        total = sim.green_time + sim.red_time + 0.001
        green_w = int((sim.green_time / max(total, 60)) * bar_w)
        red_w = int((sim.red_time / max(total, 60)) * bar_w)
        
        # Draw segments
        pygame.draw.rect(self.screen, self.green, (bar_x, y, green_w, bar_h))
        pygame.draw.rect(self.screen, self.red, (bar_x + green_w, y, red_w, bar_h))
        
        # Border
        pygame.draw.rect(self.screen, self.text, (bar_x, y, bar_w, bar_h), 1)
        
        return y + bar_h
        
    def _draw_delay_graph(self, sim, y):
        """Draw delay history graph."""
        graph_x = 40
        graph_w = self.width - 80
        graph_h = 150
        
        # Background
        pygame.draw.rect(self.screen, (35, 35, 40), (graph_x, y, graph_w, graph_h))
        
        # Grid lines
        for i in range(5):
            gy = y + int(graph_h * i / 4)
            pygame.draw.line(self.screen, self.grid, (graph_x, gy), (graph_x + graph_w, gy))
            
        # Border
        pygame.draw.rect(self.screen, (80, 80, 90), (graph_x, y, graph_w, graph_h), 1)
        
        # Draw data
        if len(sim.delay_history) > 1:
            max_val = max(max(sim.delay_history), 1)
            points = []
            for i, val in enumerate(sim.delay_history):
                px = graph_x + int(i * graph_w / len(sim.delay_history))
                py = y + graph_h - int((val / max_val) * (graph_h - 10))
                points.append((px, py))
                
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.accent, False, points, 2)
                
            # Draw points
            for p in points[-20:]:  # Last 20 points
                pygame.draw.circle(self.screen, self.accent, p, 3)
                
        # Y-axis label
        label = self.font_small.render("Avg Delay (mins)", True, (150, 150, 150))
        label = pygame.transform.rotate(label, 90)
        self.screen.blit(label, (graph_x - 25, y + graph_h//2 - label.get_height()//2))
        
        # X-axis label
        if sim.sim_hours > 0:
            x_label = self.font_small.render(f"Hours: {sim.sim_hours:.1f}", True, (150, 150, 150))
            self.screen.blit(x_label, (graph_x + graph_w//2 - x_label.get_width()//2, y + graph_h + 5))
            
        return y + graph_h + 25
        
    def _draw_progress_bar(self, agent, y):
        """Draw training progress bar."""
        bar_x = 50
        bar_w = self.width - 100
        bar_h = 8
        
        progress = 1.0 - (agent.epsilon - agent.epsilon_min) / (EPSILON_START - agent.epsilon_min)
        progress = max(0, min(1, progress))
        
        # Background
        pygame.draw.rect(self.screen, self.grid, (bar_x, y, bar_w, bar_h))
        
        # Progress
        prog_w = int(bar_w * progress)
        r = int(220 * (1 - progress))
        g = int(200 * progress)
        pygame.draw.rect(self.screen, (r, g, 60), (bar_x, y, prog_w, bar_h))
        
        # End dot
        pygame.draw.circle(self.screen, (r, g, 60), (bar_x + prog_w, y + bar_h//2), 6)
        
    def handle_events(self):
        """Handle pygame events. Returns dict of actions."""
        actions = {"quit": False, "pause": False, "save": False, "load": False, 
                   "reset": False, "density": None}
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                actions["quit"] = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    actions["quit"] = True
                elif event.key == pygame.K_SPACE:
                    actions["pause"] = True
                elif event.key == pygame.K_s:
                    actions["save"] = True
                elif event.key == pygame.K_l:
                    actions["load"] = True
                elif event.key == pygame.K_r:
                    actions["reset"] = True
                elif event.key == pygame.K_1:
                    actions["density"] = "low"
                elif event.key == pygame.K_2:
                    actions["density"] = "medium"
                elif event.key == pygame.K_3:
                    actions["density"] = "high"
                    
        return actions


def main():
    parser = argparse.ArgumentParser(description="Traffic Signal RL - SUMO + Dashboard")
    parser.add_argument("--load", action="store_true", help="Load trained agent")
    parser.add_argument("--no-gui", action="store_true", help="Run SUMO without GUI")
    parser.add_argument("--model", type=str, default="trained_agent.pkl", help="Model path")
    args = parser.parse_args()
    
    print("="*60)
    print("  TRAFFIC SIGNAL RL - SUMO + PYGAME DASHBOARD")
    print("="*60)
    print("\nControls:")
    print("  1/2/3  - Traffic density (Low/Med/High)")
    print("  SPACE  - Pause/Resume")
    print("  S      - Save agent")
    print("  L      - Load agent")
    print("  R      - Reset simulation")
    print("  ESC    - Quit")
    print("\n" + "="*60)
    
    # Create components
    use_gui = not args.no_gui
    sim = SumoSimulation(gui=use_gui)
    agent = QLearningAgent()
    dashboard = StatsDashboard()
    
    # Load agent if requested
    if args.load:
        if agent.load(args.model):
            agent.set_evaluation_mode()
        else:
            print("No trained agent found, starting fresh.")
    
    # Start simulation
    sim.start()
    state = sim.get_state()
    
    paused = False
    decision_counter = 0
    # Decision every 5 seconds (5 steps at 1s/step)
    decision_interval = int(DECISION_INTERVAL / sim.STEP_LENGTH)
    
    prev_state = state
    prev_action = 0
    
    clock = pygame.time.Clock()
    
    print("\nRunning... SUMO window should open.")
    print("Dashboard window shows training stats.\n")
    
    running = True
    while running:
        # Handle events
        actions = dashboard.handle_events()
        
        if actions["quit"]:
            running = False
            break
            
        if actions["pause"]:
            paused = not paused
            print("PAUSED" if paused else "RESUMED")
            
        if actions["save"]:
            agent.save(args.model)
            
        if actions["load"]:
            agent.load(args.model)
            
        if actions["reset"]:
            print("Resetting simulation...")
            state = sim.reset()
            prev_state = state
            decision_counter = 0
            
        if not paused and sim.is_running():
            # RL Decision
            decision_counter += 1
            
            if decision_counter >= decision_interval:
                decision_counter = 0
                state = sim.get_state()
                action = agent.choose_action(state)
                
                try:
                    reward = sim.step(action)
                    next_state = sim.get_state()
                    agent.learn(prev_state, prev_action, reward, next_state)
                    prev_state = state
                    prev_action = action
                except Exception as e:
                    print(f"Simulation ended: {e}")
                    print("Restarting...")
                    time.sleep(1)
                    state = sim.reset()
                    prev_state = state
            else:
                try:
                    sim.step()
                except Exception as e:
                    print(f"Simulation step error: {e}")
                    time.sleep(1)
                    state = sim.reset()
                    prev_state = state
        
        # Draw dashboard
        dashboard.draw(sim, agent)
        
        clock.tick(30)  # 30 FPS for dashboard
        
    # Cleanup
    sim.close()
    pygame.quit()
    
    # Auto-save on exit
    agent.save(args.model)
    print("\nAgent saved. Goodbye!")


if __name__ == "__main__":
    main()
