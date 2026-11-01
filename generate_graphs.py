"""
Generates all graphs needed for the project report.
Run this script to create PNG images you can paste into your document.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─── Load Q-Table ────────────────────────────────────────────────
with open("trained_agent.json", "r") as f:
    data = json.load(f)

q_table = data["q_table"]

# ─── Graph 1: Q-Values Heatmap (What the agent learned) ──────────
def plot_q_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Learned Q-Values: Agent Decision Map", fontsize=16, fontweight='bold')

    for phase_idx, phase_name in enumerate(["Phase 0: NS Green", "Phase 1: EW Green"]):
        matrix = np.full((4, 4), np.nan)
        decision = np.full((4, 4), "", dtype=object)

        for ns in range(4):
            for ew in range(4):
                key = f"({ns}, {ew}, {phase_idx})"
                if key in q_table:
                    vals = q_table[key]
                    if len(vals) == 2:
                        best = np.argmax(vals)
                        matrix[ns][ew] = vals[best]
                        decision[ns][ew] = "NS→" if best == 0 else "EW→"

        im = axes[phase_idx].imshow(matrix, cmap="RdYlGn", aspect="auto",
                                     vmin=-500, vmax=0)
        axes[phase_idx].set_title(phase_name, fontsize=13)
        axes[phase_idx].set_xlabel("EW Queue (0=empty, 3=heavy)", fontsize=11)
        axes[phase_idx].set_ylabel("NS Queue (0=empty, 3=heavy)", fontsize=11)
        axes[phase_idx].set_xticks([0, 1, 2, 3])
        axes[phase_idx].set_yticks([0, 1, 2, 3])
        axes[phase_idx].set_xticklabels(["0\n(empty)", "1\n(1-3)", "2\n(4-7)", "3\n(8+)"])
        axes[phase_idx].set_yticklabels(["0\n(empty)", "1\n(1-3)", "2\n(4-7)", "3\n(8+)"])

        for ns in range(4):
            for ew in range(4):
                if decision[ns][ew]:
                    axes[phase_idx].text(ew, ns, decision[ns][ew],
                                          ha='center', va='center',
                                          fontsize=10, fontweight='bold', color='black')

        plt.colorbar(im, ax=axes[phase_idx], label="Best Q-Value")

    plt.tight_layout()
    plt.savefig("graph_q_heatmap.png", dpi=150, bbox_inches='tight')
    print("✓ Saved: graph_q_heatmap.png")
    plt.show()


# ─── Graph 2: Comparison Bar Chart ───────────────────────────────
def plot_comparison():
    metrics = ['Avg Waiting\nTime (s)', 'Avg Queue\nLength (vehicles)']
    fixed_values = [26.2, 2.9]
    rl_values = [14.2, 2.1]
    improvements = [45.6, 27.6]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, fixed_values, width, label='Fixed-Time (30s cycles)',
                   color='#e74c3c', alpha=0.85, edgecolor='black')
    bars2 = ax.bar(x + width/2, rl_values, width, label='RL Agent (Q-Learning)',
                   color='#2ecc71', alpha=0.85, edgecolor='black')

    # Add value labels on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{bar.get_height()}', ha='center', va='bottom', fontweight='bold')
    for bar, imp in zip(bars2, improvements):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{bar.get_height()}\n(-{imp}%)', ha='center', va='bottom',
                fontweight='bold', color='#27ae60')

    ax.set_title('Fixed-Time vs RL Agent: Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 35)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("graph_comparison.png", dpi=150, bbox_inches='tight')
    print("✓ Saved: graph_comparison.png")
    plt.show()


# ─── Graph 3: Epsilon Decay Curve ────────────────────────────────
def plot_epsilon_decay():
    episodes = np.arange(0, 201)
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 0.9995
    decisions_per_episode = 720

    epsilons = []
    eps = epsilon_start
    for ep in episodes:
        epsilons.append(eps)
        for _ in range(decisions_per_episode):
            eps = max(epsilon_end, eps * epsilon_decay)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, epsilons, color='#3498db', linewidth=2.5)
    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Min epsilon (0.05)')
    ax.axvline(x=50, color='green', linestyle='--', alpha=0.7, label='Stabilized (~ep 50)')

    ax.fill_between(episodes, epsilons, 0.05, alpha=0.15, color='#3498db')
    ax.annotate('Exploration Phase\n(Random decisions)', xy=(25, 0.5),
                fontsize=10, ha='center', color='#2980b9')
    ax.annotate('Exploitation Phase\n(Smart decisions)', xy=(130, 0.12),
                fontsize=10, ha='center', color='#27ae60')

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Epsilon (ε)', fontsize=12)
    ax.set_title('Epsilon Decay: Exploration → Exploitation Over Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("graph_epsilon_decay.png", dpi=150, bbox_inches='tight')
    print("✓ Saved: graph_epsilon_decay.png")
    plt.show()


# ─── Graph 4: Q-Value Distribution ───────────────────────────────
def plot_q_distribution():
    all_q_vals = []
    for vals in q_table.values():
        all_q_vals.extend(vals)

    # Separate normal and gridlock states
    normal = [v for v in all_q_vals if v > -1000]
    gridlock = [v for v in all_q_vals if v <= -1000]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Q-Value Distribution Across Learned States", fontsize=14, fontweight='bold')

    # Normal states
    axes[0].hist(normal, bins=15, color='#3498db', edgecolor='black', alpha=0.8)
    axes[0].set_title('Normal Traffic States\n(30 states)', fontsize=12)
    axes[0].set_xlabel('Q-Value', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].grid(alpha=0.3)

    # Gridlock states
    axes[1].bar(['(3,3,0)\nBoth Heavy\nNS Green', '(3,3,1)\nBoth Heavy\nEW Green'],
                [min(q_table.get("(3, 3, 0)", [0])), min(q_table.get("(3, 3, 1)", [0]))],
                color=['#e74c3c', '#c0392b'], edgecolor='black', alpha=0.85)
    axes[1].set_title('Gridlock States\n(Agent learned to AVOID)', fontsize=12)
    axes[1].set_ylabel('Q-Value (lower = worse)', fontsize=11)
    axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig("graph_q_distribution.png", dpi=150, bbox_inches='tight')
    print("✓ Saved: graph_q_distribution.png")
    plt.show()


# ─── Run All ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nGenerating all graphs for project report...\n")
    plot_comparison()
    plot_epsilon_decay()
    plot_q_heatmap()
    plot_q_distribution()
    print("\n✓ All graphs saved! Look for PNG files in your project folder.")
    print("\nPaste these into your report:")
    print("  graph_comparison.png     → Section 5.2 (Results)")
    print("  graph_epsilon_decay.png  → Section 3 (Algorithm / Training)")
    print("  graph_q_heatmap.png      → Section 2 (Algorithm / Q-Table)")
    print("  graph_q_distribution.png → Section 5 (Results / Analysis)")