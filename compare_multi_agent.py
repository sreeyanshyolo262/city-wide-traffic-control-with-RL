"""
Side-by-side comparison of a trained multi-agent DQN system
and a fixed-time controller in the 4x4 SUMO grid.
"""

import os
import sys
import argparse
import numpy as np
from collections import defaultdict, deque
import pygame

# Check SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please set SUMO_HOME environment variable")

import traci
from dqn_agent import DQNAgent

class SumoRunner:
    """Manages a single SUMO instance."""
    def __init__(self, config_file, label, gui=True, is_fixed_time=False, traffic_scale=1.0):
        self.label = label
        self.is_fixed_time = is_fixed_time
        self.gui = gui
        self.traffic_scale = float(traffic_scale)
        self.config_file = config_file
        self._running = False
        self.tl_ids = []
        self.incoming_lanes = {}
        self.outgoing_lanes = {}

    def _switch(self):
        traci.switch(self.label)

    def start(self):
        sumo_binary = "sumo-gui" if self.gui else "sumo"
        gui_settings = os.path.join(os.path.dirname(self.config_file), "gui_settings.xml")
        sumo_cmd = [
            sumo_binary,
            "-c", self.config_file,
            "--start",
            "--quit-on-end",
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1",
            "--no-warnings",
            "--step-length", "1.0",
            "--scale", str(self.traffic_scale),
        ]
        if self.gui and os.path.exists(gui_settings):
            sumo_cmd.extend(["--gui-settings-file", gui_settings])
        traci.start(sumo_cmd, label=self.label)
        self._switch()
        self._running = True
        self.tl_ids = traci.trafficlight.getIDList()
        for tl_id in self.tl_ids:
            self.incoming_lanes[tl_id] = self._get_incoming_lanes(tl_id)
            self.outgoing_lanes[tl_id] = self._get_outgoing_lanes(tl_id)
        
        if self.is_fixed_time:
            # Set fixed time program for all traffic lights
            for tl_id in self.tl_ids:
                traci.trafficlight.setProgram(tl_id, "0")

    def step(self, actions=None):
        if not self._running:
            return
        self._switch()
        
        if not self.is_fixed_time and actions:
            for tl_id, action in actions.items():
                phase_to_set = action * 2
                if traci.trafficlight.getPhase(tl_id) != phase_to_set:
                    traci.trafficlight.setPhase(tl_id, phase_to_set)
        
        traci.simulationStep()

    def get_stats(self):
        self._switch()
        stats = defaultdict(float)
        for tl_id in self.tl_ids:
            for lane in self.incoming_lanes[tl_id]:
                stats['total_queue'] += traci.lane.getLastStepHaltingNumber(lane)
                stats['total_wait_time'] += traci.lane.getWaitingTime(lane)
        
        stats['total_vehicles'] = traci.vehicle.getIDCount()
        stats['arrived_step'] = traci.simulation.getArrivedNumber()
        if stats['total_vehicles'] > 0:
            stats['avg_wait_time'] = stats['total_wait_time'] / stats['total_vehicles']
        return stats

    def get_all_states(self):
        self._switch()
        states = {}
        for tl_id in self.tl_ids:
            state = []
            current_phase = traci.trafficlight.getPhase(tl_id)
            phase_group = 0 if current_phase in (0, 1) else 1
            state.append(float(phase_group))
            for lane in self.incoming_lanes[tl_id]:
                state.append(traci.lane.getLastStepHaltingNumber(lane) / 50.0)
            for lane in self.outgoing_lanes[tl_id]:
                state.append(traci.lane.getLastStepHaltingNumber(lane) / 50.0)
            states[tl_id] = np.array(state)
        return states

    def _get_incoming_lanes(self, tl_id):
        self._switch()
        lanes = set()
        for link in traci.trafficlight.getControlledLinks(tl_id):
            lanes.add(link[0][0])
        return sorted(lanes)

    def _get_outgoing_lanes(self, tl_id):
        self._switch()
        lanes = set()
        for link_group in traci.trafficlight.getControlledLinks(tl_id):
            if len(link_group) > 0 and len(link_group[0]) > 1:
                lanes.add(link_group[0][1])
        return sorted(lanes)

    def close(self):
        if self._running:
            try:
                self._switch()
                traci.close()
            except Exception:
                pass
            self._running = False


def rolling_mean(values, window=20):
    if not values:
        return 0.0
    arr = np.array(values[-window:], dtype=float)
    return float(np.mean(arr))


def smooth_series(values, window=15):
    if not values:
        return []
    smoothed = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        smoothed.append(float(np.mean(values[start:i + 1])))
    return smoothed


def draw_line_chart(screen, rect, series_a, series_b, color_a, color_b, title, font):
    x, y, w, h = rect
    pygame.draw.rect(screen, (35, 35, 45), rect, border_radius=6)
    pygame.draw.rect(screen, (80, 80, 95), rect, width=1, border_radius=6)

    title_surf = font.render(title, True, (220, 220, 220))
    screen.blit(title_surf, (x + 8, y + 6))

    if len(series_a) < 2 and len(series_b) < 2:
        return

    vmax = max(max(series_a) if series_a else 1.0, max(series_b) if series_b else 1.0, 1e-6)
    vmin = min(min(series_a) if series_a else 0.0, min(series_b) if series_b else 0.0)
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1.0

    def to_points(series):
        points = []
        n = len(series)
        if n == 1:
            points.append((x + 10, y + h - 10))
            return points
        for i, value in enumerate(series):
            px = x + 10 + (w - 20) * (i / (n - 1))
            py = y + h - 10 - (h - 20) * ((value - vmin) / (vmax - vmin))
            points.append((px, py))
        return points

    pts_a = to_points(series_a)
    pts_b = to_points(series_b)
    if len(pts_a) >= 2:
        pygame.draw.lines(screen, color_a, False, pts_a, 2)
    if len(pts_b) >= 2:
        pygame.draw.lines(screen, color_b, False, pts_b, 2)

    # Max/min labels to make chart values interpretable.
    max_text = font.render(f"max {vmax:.2f}", True, (170, 170, 170))
    min_text = font.render(f"min {vmin:.2f}", True, (170, 170, 170))
    screen.blit(max_text, (x + w - 95, y + 8))
    screen.blit(min_text, (x + w - 95, y + h - 22))

def run_comparison(duration=1800, traffic_scale=1.0, decision_interval_steps=20):
    """Run side-by-side comparison."""
    config_path = os.path.join(os.path.dirname(__file__), 'sumo_files', '4x4', 'intersection.sumocfg')

    # --- Setup Instances ---
    rl_runner = SumoRunner(config_path, label="rl", gui=True, is_fixed_time=False, traffic_scale=traffic_scale)
    fixed_runner = SumoRunner(config_path, label="fixed", gui=True, is_fixed_time=True, traffic_scale=traffic_scale)

    print("Opening SUMO windows...")
    print("  Window 1 (opened first): DQN controller")
    print("  Window 2 (opened second): Fixed-time controller")

    rl_runner.start()
    fixed_runner.start()

    # --- Load RL Agents ---
    agents = {}
    initial_states = rl_runner.get_all_states()
    for tl_id, state in initial_states.items():
        agent = DQNAgent(len(state), 2, agent_id=tl_id)
        if agent.load(f"trained_agent_{tl_id}.weights.h5"):
            agent.set_evaluation_mode()
            agents[tl_id] = agent
        else:
            print(f"Could not load agent for {tl_id}. Exiting.")
            rl_runner.close()
            fixed_runner.close()
            return

    # --- Pygame for Stats Display ---
    pygame.init()
    screen = pygame.display.set_mode((1200, 700))
    pygame.display.set_caption("Comparison: RL vs Fixed-Time")
    font = pygame.font.SysFont("Arial", 24)
    font_bold = pygame.font.SysFont("Arial", 28, bold=True)
    font_small = pygame.font.SysFont("Arial", 18)

    # --- Main Loop ---
    rl_wait_sum = 0.0
    rl_queue_sum = 0.0
    rl_vehicle_sum = 0.0
    rl_arrived_sum = 0.0
    fixed_wait_sum = 0.0
    fixed_queue_sum = 0.0
    fixed_vehicle_sum = 0.0
    fixed_arrived_sum = 0.0

    hist_len = min(max(120, duration), 500)
    rl_wait_hist = deque(maxlen=hist_len)
    fx_wait_hist = deque(maxlen=hist_len)
    rl_queue_hist = deque(maxlen=hist_len)
    fx_queue_hist = deque(maxlen=hist_len)
    rl_arrived_hist = deque(maxlen=hist_len)
    fx_arrived_hist = deque(maxlen=hist_len)

    rl_arrived_cum = 0.0
    fx_arrived_cum = 0.0

    try:
        for step in range(duration):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            # RL Runner Step
            rl_states = rl_runner.get_all_states()
            if step % decision_interval_steps == 0:
                rl_actions = {tl_id: agents[tl_id].choose_action(np.reshape(s, [1, len(s)])) for tl_id, s in rl_states.items()}
            else:
                rl_actions = None
            rl_runner.step(rl_actions)

            # Fixed-Time Runner Step
            fixed_runner.step()

            # Get Stats
            rl_stats = rl_runner.get_stats()
            fixed_stats = fixed_runner.get_stats()

            rl_wait_sum += rl_stats['avg_wait_time']
            rl_queue_sum += rl_stats['total_queue']
            rl_vehicle_sum += rl_stats['total_vehicles']
            rl_arrived_sum += rl_stats['arrived_step']
            fixed_wait_sum += fixed_stats['avg_wait_time']
            fixed_queue_sum += fixed_stats['total_queue']
            fixed_vehicle_sum += fixed_stats['total_vehicles']
            fixed_arrived_sum += fixed_stats['arrived_step']

            rl_arrived_cum += rl_stats['arrived_step']
            fx_arrived_cum += fixed_stats['arrived_step']

            rl_wait_hist.append(float(rl_stats['avg_wait_time']))
            fx_wait_hist.append(float(fixed_stats['avg_wait_time']))
            rl_queue_hist.append(float(rl_stats['total_queue']))
            fx_queue_hist.append(float(fixed_stats['total_queue']))
            rl_arrived_hist.append(float(rl_arrived_cum))
            fx_arrived_hist.append(float(fx_arrived_cum))

            # Display Stats
            screen.fill((25, 25, 30))
            
            # Titles
            rl_title = font_bold.render("DQN Multi-Agent", True, (80, 180, 255))
            fixed_title = font_bold.render("Fixed-Time Control", True, (220, 50, 50))
            screen.blit(rl_title, (60, 18))
            screen.blit(fixed_title, (650, 18))

            # Stats display
            def draw_stats(x_offset, title, stats):
                y_offset = 70
                labels = ["Avg Wait Time", "Total Queue", "Total Vehicles"]
                values = [f"{stats['avg_wait_time']:.2f} s", f"{stats['total_queue']:.0f}", f"{stats['total_vehicles']:.0f}"]
                
                for label, value in zip(labels, values):
                    label_surf = font.render(label, True, (220, 220, 220))
                    value_surf = font.render(value, True, (255, 255, 255))
                    screen.blit(label_surf, (x_offset, y_offset))
                    screen.blit(value_surf, (x_offset + 200, y_offset))
                    y_offset += 40

            draw_stats(50, "RL", rl_stats)
            draw_stats(640, "Fixed", fixed_stats)

            legend_rl = font_small.render("DQN", True, (80, 180, 255))
            legend_fx = font_small.render("Fixed", True, (220, 70, 70))
            screen.blit(legend_rl, (1030, 24))
            screen.blit(legend_fx, (1080, 24))

            sm_rl_wait = smooth_series(list(rl_wait_hist), window=20)
            sm_fx_wait = smooth_series(list(fx_wait_hist), window=20)
            sm_rl_queue = smooth_series(list(rl_queue_hist), window=20)
            sm_fx_queue = smooth_series(list(fx_queue_hist), window=20)

            draw_line_chart(
                screen,
                rect=(20, 220, 370, 220),
                series_a=sm_rl_wait,
                series_b=sm_fx_wait,
                color_a=(80, 180, 255),
                color_b=(220, 70, 70),
                title="Avg Wait Time (rolling)",
                font=font_small,
            )
            draw_line_chart(
                screen,
                rect=(415, 220, 370, 220),
                series_a=sm_rl_queue,
                series_b=sm_fx_queue,
                color_a=(80, 180, 255),
                color_b=(220, 70, 70),
                title="Total Queue (rolling)",
                font=font_small,
            )
            draw_line_chart(
                screen,
                rect=(810, 220, 370, 220),
                series_a=list(rl_arrived_hist),
                series_b=list(fx_arrived_hist),
                color_a=(80, 180, 255),
                color_b=(220, 70, 70),
                title="Cumulative Throughput",
                font=font_small,
            )

            wait_adv_now = float(fixed_stats['avg_wait_time'] - rl_stats['avg_wait_time'])
            queue_adv_now = float(fixed_stats['total_queue'] - rl_stats['total_queue'])
            wait_adv_roll = rolling_mean([f - r for f, r in zip(fx_wait_hist, rl_wait_hist)], window=30)
            queue_adv_roll = rolling_mean([f - r for f, r in zip(fx_queue_hist, rl_queue_hist)], window=30)
            adv_text_1 = font_small.render(
                f"Wait advantage (Fixed-DQN): now {wait_adv_now:+.2f}s | roll30 {wait_adv_roll:+.2f}s",
                True,
                (230, 230, 230),
            )
            adv_text_2 = font_small.render(
                f"Queue advantage (Fixed-DQN): now {queue_adv_now:+.2f} | roll30 {queue_adv_roll:+.2f}",
                True,
                (230, 230, 230),
            )
            screen.blit(adv_text_1, (20, 470))
            screen.blit(adv_text_2, (20, 495))
            
            pygame.display.flip()
            
            print(f"\rStep {step}/{duration}", end="")

        denom = float(max(1, duration))
        rl_wait_avg = rl_wait_sum / denom
        rl_queue_avg = rl_queue_sum / denom
        rl_veh_avg = rl_vehicle_sum / denom
        rl_arrived_avg = rl_arrived_sum / denom
        fx_wait_avg = fixed_wait_sum / denom
        fx_queue_avg = fixed_queue_sum / denom
        fx_veh_avg = fixed_vehicle_sum / denom
        fx_arrived_avg = fixed_arrived_sum / denom

        wait_improvement = ((fx_wait_avg - rl_wait_avg) / max(1e-6, fx_wait_avg)) * 100.0
        queue_improvement = ((fx_queue_avg - rl_queue_avg) / max(1e-6, fx_queue_avg)) * 100.0

        print("\n\n=== Comparison Summary ===")
        print(f"DQN   avg wait: {rl_wait_avg:.2f}s | avg queue: {rl_queue_avg:.2f} | avg vehicles: {rl_veh_avg:.2f} | throughput/step: {rl_arrived_avg:.2f}")
        print(f"Fixed avg wait: {fx_wait_avg:.2f}s | avg queue: {fx_queue_avg:.2f} | avg vehicles: {fx_veh_avg:.2f} | throughput/step: {fx_arrived_avg:.2f}")
        print(f"Wait improvement vs fixed: {wait_improvement:+.2f}%")
        print(f"Queue improvement vs fixed: {queue_improvement:+.2f}%")

    except KeyboardInterrupt:
        print("\nComparison stopped by user.")
    finally:
        rl_runner.close()
        fixed_runner.close()
        pygame.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare DQN vs fixed-time on 4x4 SUMO grid")
    parser.add_argument("--duration", type=int, default=1800, help="Simulation steps for comparison")
    parser.add_argument("--traffic-scale", type=float, default=1.0, help="SUMO demand scale multiplier (e.g., 5 or 6)")
    parser.add_argument("--decision-interval-steps", type=int, default=20, help="Choose new DQN action every N steps")
    args = parser.parse_args()

    run_comparison(
        duration=args.duration,
        traffic_scale=args.traffic_scale,
        decision_interval_steps=args.decision_interval_steps,
    )
