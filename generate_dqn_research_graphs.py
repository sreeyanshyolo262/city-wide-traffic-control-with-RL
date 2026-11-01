"""
Generate research-focused analytics for multi-agent DQN traffic signal control.

This script evaluates DQN against fixed-time control, saves raw step metrics,
and creates publication-friendly graphs and matrix heatmaps.

Usage examples:
  python generate_dqn_research_graphs.py --duration 1800 --traffic-scale 4 --decision-interval-steps 20
  python generate_dqn_research_graphs.py --duration 1200 --grid-study
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# Check SUMO_HOME
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please set SUMO_HOME environment variable")

import traci

from dqn_agent import DQNAgent


class SumoRunner:
    """Manages one SUMO instance for evaluation."""

    def __init__(self, config_file, label, gui=False, is_fixed_time=False, traffic_scale=1.0):
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
        sumo_cmd = [
            sumo_binary,
            "-c",
            self.config_file,
            "--start",
            "--quit-on-end",
            "--waiting-time-memory",
            "1000",
            "--time-to-teleport",
            "-1",
            "--no-warnings",
            "--step-length",
            "1.0",
            "--scale",
            str(self.traffic_scale),
        ]
        traci.start(sumo_cmd, label=self.label)
        self._switch()
        self._running = True
        self.tl_ids = traci.trafficlight.getIDList()
        for tl_id in self.tl_ids:
            self.incoming_lanes[tl_id] = self._get_incoming_lanes(tl_id)
            self.outgoing_lanes[tl_id] = self._get_outgoing_lanes(tl_id)

        if self.is_fixed_time:
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
                stats["total_queue"] += traci.lane.getLastStepHaltingNumber(lane)
                stats["total_wait_time"] += traci.lane.getWaitingTime(lane)

        stats["total_vehicles"] = traci.vehicle.getIDCount()
        stats["arrived_step"] = traci.simulation.getArrivedNumber()
        if stats["total_vehicles"] > 0:
            stats["avg_wait_time"] = stats["total_wait_time"] / stats["total_vehicles"]
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


def _moving_average(values, window=20):
    if len(values) == 0:
        return []
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(float(np.mean(values[start : i + 1])))
    return out


def run_single_eval(duration, traffic_scale, decision_interval_steps, use_gui=False):
    config_path = os.path.join(os.path.dirname(__file__), "sumo_files", "4x4", "intersection.sumocfg")

    rl_runner = SumoRunner(config_path, label="rl_eval", gui=use_gui, is_fixed_time=False, traffic_scale=traffic_scale)
    fixed_runner = SumoRunner(config_path, label="fixed_eval", gui=use_gui, is_fixed_time=True, traffic_scale=traffic_scale)

    rl_runner.start()
    fixed_runner.start()

    agents = {}
    initial_states = rl_runner.get_all_states()
    for tl_id, state in initial_states.items():
        agent = DQNAgent(len(state), 2, agent_id=tl_id)
        if not agent.load(f"trained_agent_{tl_id}.weights.h5"):
            rl_runner.close()
            fixed_runner.close()
            raise RuntimeError(f"Missing or incompatible checkpoint for {tl_id}")
        agent.set_evaluation_mode()
        agents[tl_id] = agent

    action_counts = {tl_id: [0, 0] for tl_id in agents}

    metrics = {
        "step": [],
        "rl_wait": [],
        "fx_wait": [],
        "rl_queue": [],
        "fx_queue": [],
        "rl_vehicles": [],
        "fx_vehicles": [],
        "rl_arrived_cum": [],
        "fx_arrived_cum": [],
    }

    rl_arrived_cum = 0.0
    fx_arrived_cum = 0.0

    try:
        for step in range(duration):
            rl_states = rl_runner.get_all_states()
            if step % decision_interval_steps == 0:
                rl_actions = {}
                for tl_id, s in rl_states.items():
                    action = agents[tl_id].choose_action(np.reshape(s, [1, len(s)]))
                    rl_actions[tl_id] = action
                    action_counts[tl_id][action] += 1
            else:
                rl_actions = None

            rl_runner.step(rl_actions)
            fixed_runner.step()

            rl_stats = rl_runner.get_stats()
            fx_stats = fixed_runner.get_stats()

            rl_arrived_cum += rl_stats["arrived_step"]
            fx_arrived_cum += fx_stats["arrived_step"]

            metrics["step"].append(step)
            metrics["rl_wait"].append(float(rl_stats["avg_wait_time"]))
            metrics["fx_wait"].append(float(fx_stats["avg_wait_time"]))
            metrics["rl_queue"].append(float(rl_stats["total_queue"]))
            metrics["fx_queue"].append(float(fx_stats["total_queue"]))
            metrics["rl_vehicles"].append(float(rl_stats["total_vehicles"]))
            metrics["fx_vehicles"].append(float(fx_stats["total_vehicles"]))
            metrics["rl_arrived_cum"].append(float(rl_arrived_cum))
            metrics["fx_arrived_cum"].append(float(fx_arrived_cum))

    finally:
        rl_runner.close()
        fixed_runner.close()

    return metrics, action_counts


def compute_summary(metrics):
    rl_wait_avg = float(np.mean(metrics["rl_wait"]))
    fx_wait_avg = float(np.mean(metrics["fx_wait"]))
    rl_queue_avg = float(np.mean(metrics["rl_queue"]))
    fx_queue_avg = float(np.mean(metrics["fx_queue"]))
    rl_throughput = float(metrics["rl_arrived_cum"][-1] / max(1, len(metrics["step"])))
    fx_throughput = float(metrics["fx_arrived_cum"][-1] / max(1, len(metrics["step"])))

    wait_impr = ((fx_wait_avg - rl_wait_avg) / max(1e-6, fx_wait_avg)) * 100.0
    queue_impr = ((fx_queue_avg - rl_queue_avg) / max(1e-6, fx_queue_avg)) * 100.0
    throughput_impr = ((rl_throughput - fx_throughput) / max(1e-6, fx_throughput)) * 100.0

    return {
        "rl_wait_avg": rl_wait_avg,
        "fx_wait_avg": fx_wait_avg,
        "rl_queue_avg": rl_queue_avg,
        "fx_queue_avg": fx_queue_avg,
        "rl_throughput_per_step": rl_throughput,
        "fx_throughput_per_step": fx_throughput,
        "wait_improvement_pct": wait_impr,
        "queue_improvement_pct": queue_impr,
        "throughput_improvement_pct": throughput_impr,
    }


def save_step_csv(metrics, out_csv):
    header = [
        "step",
        "rl_wait",
        "fx_wait",
        "rl_queue",
        "fx_queue",
        "rl_vehicles",
        "fx_vehicles",
        "rl_arrived_cum",
        "fx_arrived_cum",
    ]
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for i in range(len(metrics["step"])):
            row = [
                metrics["step"][i],
                metrics["rl_wait"][i],
                metrics["fx_wait"][i],
                metrics["rl_queue"][i],
                metrics["fx_queue"][i],
                metrics["rl_vehicles"][i],
                metrics["fx_vehicles"][i],
                metrics["rl_arrived_cum"][i],
                metrics["fx_arrived_cum"][i],
            ]
            f.write(",".join(str(x) for x in row) + "\n")


def plot_time_series(metrics, out_png):
    x = metrics["step"]

    rl_wait = _moving_average(metrics["rl_wait"], window=30)
    fx_wait = _moving_average(metrics["fx_wait"], window=30)
    rl_queue = _moving_average(metrics["rl_queue"], window=30)
    fx_queue = _moving_average(metrics["fx_queue"], window=30)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(x, rl_wait, label="DQN", color="#2E86DE")
    axes[0].plot(x, fx_wait, label="Fixed", color="#E74C3C")
    axes[0].set_ylabel("Avg Wait (s)")
    axes[0].set_title("Rolling Average Waiting Time")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(x, rl_queue, label="DQN", color="#2E86DE")
    axes[1].plot(x, fx_queue, label="Fixed", color="#E74C3C")
    axes[1].set_ylabel("Total Queue")
    axes[1].set_title("Rolling Total Queue")
    axes[1].grid(alpha=0.25)

    axes[2].plot(x, metrics["rl_arrived_cum"], label="DQN", color="#2E86DE")
    axes[2].plot(x, metrics["fx_arrived_cum"], label="Fixed", color="#E74C3C")
    axes[2].set_ylabel("Cumulative Arrivals")
    axes[2].set_xlabel("Simulation Step (1 step = 1s)")
    axes[2].set_title("Cumulative Throughput")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_summary_bars(summary, out_png):
    labels = ["Avg Wait", "Avg Queue", "Throughput/step"]
    dqn_vals = [summary["rl_wait_avg"], summary["rl_queue_avg"], summary["rl_throughput_per_step"]]
    fixed_vals = [summary["fx_wait_avg"], summary["fx_queue_avg"], summary["fx_throughput_per_step"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, dqn_vals, width, label="DQN", color="#2E86DE")
    ax.bar(x + width / 2, fixed_vals, width, label="Fixed", color="#E74C3C")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("DQN vs Fixed: Aggregate Performance")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    text = (
        f"Wait improvement: {summary['wait_improvement_pct']:+.2f}%\n"
        f"Queue improvement: {summary['queue_improvement_pct']:+.2f}%\n"
        f"Throughput improvement: {summary['throughput_improvement_pct']:+.2f}%"
    )
    ax.text(1.02, 0.98, text, transform=ax.transAxes, va="top", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_action_matrix(action_counts, out_png):
    tls = sorted(action_counts.keys())
    mat = np.zeros((len(tls), 2), dtype=float)

    for i, tl in enumerate(tls):
        ns_count, ew_count = action_counts[tl]
        total = max(1, ns_count + ew_count)
        mat[i, 0] = ns_count / total
        mat[i, 1] = ew_count / total

    fig, ax = plt.subplots(figsize=(6, max(3, len(tls) * 0.7)))
    im = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Action 0 (NS)", "Action 1 (EW)"])
    ax.set_yticks(np.arange(len(tls)))
    ax.set_yticklabels(tls)
    ax.set_title("DQN Action Allocation Matrix (Per Intersection)")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, label="Action Share")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_grid_heatmap(matrix, x_labels, y_labels, title, out_png, cmap="RdYlGn"):
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap=cmap)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Decision Interval (steps)")
    ax.set_ylabel("Traffic Scale")
    ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:+.1f}", ha="center", va="center", color="black")

    fig.colorbar(im, ax=ax, label="Improvement (%)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def run_grid_study(base_duration, scales, intervals):
    wait_matrix = np.zeros((len(scales), len(intervals)), dtype=float)
    queue_matrix = np.zeros((len(scales), len(intervals)), dtype=float)

    for i, scale in enumerate(scales):
        for j, interval in enumerate(intervals):
            metrics, _ = run_single_eval(
                duration=base_duration,
                traffic_scale=scale,
                decision_interval_steps=interval,
                use_gui=False,
            )
            summary = compute_summary(metrics)
            wait_matrix[i, j] = summary["wait_improvement_pct"]
            queue_matrix[i, j] = summary["queue_improvement_pct"]

    return wait_matrix, queue_matrix


def main():
    parser = argparse.ArgumentParser(description="Generate research analytics for DQN traffic control")
    parser.add_argument("--duration", type=int, default=1200, help="Evaluation duration in steps")
    parser.add_argument("--traffic-scale", type=float, default=4.0, help="Demand scaling factor")
    parser.add_argument("--decision-interval-steps", type=int, default=20, help="Action decision interval")
    parser.add_argument("--out-dir", type=str, default="research_outputs", help="Output folder")
    parser.add_argument("--gui", action="store_true", help="Run SUMO-GUI during data capture")
    parser.add_argument("--grid-study", action="store_true", help="Run matrix study across scales and intervals")
    parser.add_argument(
        "--grid-scales",
        type=str,
        default="2,4,6",
        help="Comma-separated traffic scales for grid study",
    )
    parser.add_argument(
        "--grid-intervals",
        type=str,
        default="10,20,30",
        help="Comma-separated decision intervals for grid study",
    )
    parser.add_argument(
        "--use-manual-summary",
        action="store_true",
        help="Use manually provided aggregate values to generate summary/bar chart without running SUMO",
    )
    parser.add_argument("--rl-wait", type=float, default=None, help="Manual DQN average wait")
    parser.add_argument("--fx-wait", type=float, default=None, help="Manual Fixed average wait")
    parser.add_argument("--rl-queue", type=float, default=None, help="Manual DQN average queue")
    parser.add_argument("--fx-queue", type=float, default=None, help="Manual Fixed average queue")
    parser.add_argument("--rl-throughput", type=float, default=None, help="Manual DQN throughput per step")
    parser.add_argument("--fx-throughput", type=float, default=None, help="Manual Fixed throughput per step")
    parser.add_argument("--wait-improvement-pct", type=float, default=None, help="Optional manual wait improvement percentage")
    parser.add_argument("--queue-improvement-pct", type=float, default=None, help="Optional manual queue improvement percentage")
    parser.add_argument("--throughput-improvement-pct", type=float, default=None, help="Optional manual throughput improvement percentage")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    summary_path = os.path.join(args.out_dir, "summary.json")
    bar_path = os.path.join(args.out_dir, "aggregate_comparison_bars.png")

    if args.use_manual_summary:
        required = [args.rl_wait, args.fx_wait, args.rl_queue, args.fx_queue, args.rl_throughput, args.fx_throughput]
        if any(v is None for v in required):
            raise ValueError(
                "Manual summary mode requires: --rl-wait --fx-wait --rl-queue --fx-queue --rl-throughput --fx-throughput"
            )

        wait_impr = ((args.fx_wait - args.rl_wait) / max(1e-6, args.fx_wait)) * 100.0
        queue_impr = ((args.fx_queue - args.rl_queue) / max(1e-6, args.fx_queue)) * 100.0
        throughput_impr = ((args.rl_throughput - args.fx_throughput) / max(1e-6, args.fx_throughput)) * 100.0

        if args.wait_improvement_pct is not None:
            wait_impr = float(args.wait_improvement_pct)
        if args.queue_improvement_pct is not None:
            queue_impr = float(args.queue_improvement_pct)
        if args.throughput_improvement_pct is not None:
            throughput_impr = float(args.throughput_improvement_pct)

        summary = {
            "rl_wait_avg": float(args.rl_wait),
            "fx_wait_avg": float(args.fx_wait),
            "rl_queue_avg": float(args.rl_queue),
            "fx_queue_avg": float(args.fx_queue),
            "rl_throughput_per_step": float(args.rl_throughput),
            "fx_throughput_per_step": float(args.fx_throughput),
            "wait_improvement_pct": float(wait_impr),
            "queue_improvement_pct": float(queue_impr),
            "throughput_improvement_pct": float(throughput_impr),
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        plot_summary_bars(summary, bar_path)

        print("\nManual summary outputs generated:")
        print(f"  - {summary_path}")
        print(f"  - {bar_path}")
        print("\nSummary:")
        for key, value in summary.items():
            print(f"  {key}: {value:.4f}")
        return

    metrics, action_counts = run_single_eval(
        duration=args.duration,
        traffic_scale=args.traffic_scale,
        decision_interval_steps=args.decision_interval_steps,
        use_gui=args.gui,
    )

    summary = compute_summary(metrics)

    csv_path = os.path.join(args.out_dir, "step_metrics.csv")
    ts_path = os.path.join(args.out_dir, "time_series_wait_queue_throughput.png")
    action_path = os.path.join(args.out_dir, "action_allocation_matrix.png")

    save_step_csv(metrics, csv_path)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_time_series(metrics, ts_path)
    plot_summary_bars(summary, bar_path)
    plot_action_matrix(action_counts, action_path)

    if args.grid_study:
        scales = [float(x.strip()) for x in args.grid_scales.split(",") if x.strip()]
        intervals = [int(x.strip()) for x in args.grid_intervals.split(",") if x.strip()]

        wait_matrix, queue_matrix = run_grid_study(args.duration, scales, intervals)
        np.save(os.path.join(args.out_dir, "wait_improvement_matrix.npy"), wait_matrix)
        np.save(os.path.join(args.out_dir, "queue_improvement_matrix.npy"), queue_matrix)

        plot_grid_heatmap(
            wait_matrix,
            [str(v) for v in intervals],
            [str(v) for v in scales],
            "Wait Improvement Heatmap (DQN vs Fixed)",
            os.path.join(args.out_dir, "wait_improvement_heatmap.png"),
            cmap="RdYlGn",
        )
        plot_grid_heatmap(
            queue_matrix,
            [str(v) for v in intervals],
            [str(v) for v in scales],
            "Queue Improvement Heatmap (DQN vs Fixed)",
            os.path.join(args.out_dir, "queue_improvement_heatmap.png"),
            cmap="RdYlGn",
        )

    print("\nResearch outputs generated:")
    print(f"  - {csv_path}")
    print(f"  - {summary_path}")
    print(f"  - {ts_path}")
    print(f"  - {bar_path}")
    print(f"  - {action_path}")
    if args.grid_study:
        print("  - grid heatmaps and matrices")

    print("\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
