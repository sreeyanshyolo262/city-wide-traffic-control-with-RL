"""
Dashboard panel: live signal timing bar, dynamic average delay graph,
queue stats, and RL metrics. Rendered on the right side of the window.
"""

import pygame
import math
from config import *


class Dashboard:
    """Draws the right-side analytics dashboard."""

    def __init__(self, surface, x_offset):
        self.surface = surface
        self.x = x_offset
        self.w = DASH_WIDTH
        self.h = WINDOW_HEIGHT

        # Fonts
        pygame.font.init()
        self.font_title = pygame.font.SysFont("Arial", 22, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 17, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 15)
        self.font_large = pygame.font.SysFont("Arial", 28, bold=True)
        self.font_value = pygame.font.SysFont("Consolas", 19, bold=True)

        # Graph buffer
        self.graph_surface = pygame.Surface((480, 250))

    def draw(self, env, agent):
        """Render the full dashboard."""
        # Background
        pygame.draw.rect(self.surface, COLOR_DASH_BG,
                         (self.x, 0, self.w, self.h))
        # Subtle left border
        pygame.draw.line(self.surface, (50, 50, 60),
                         (self.x, 0), (self.x, self.h), 2)

        y_cursor = 15
        cx = self.x + self.w // 2  # Center x of dashboard

        # ─── Title: Corridor Signal Timing ───
        title = self.font_title.render("Corridor Signal Timing", True, COLOR_DASH_TEXT)
        self.surface.blit(title, (cx - title.get_width() // 2, y_cursor))
        y_cursor += 35

        # ─── Signal Timing Bar ───
        self._draw_timing_bar(env, y_cursor)
        y_cursor += 35

        # ─── Last Green / Red ───
        green_text = self.font_medium.render(
            f"Corridor Last Green: {env.last_green_duration:.1f}", True, COLOR_DASH_TEXT)
        self.surface.blit(green_text, (cx - green_text.get_width() // 2, y_cursor))
        y_cursor += 22
        red_text = self.font_medium.render(
            f"Corridor Last Red: {env.last_red_duration:.1f}", True, COLOR_DASH_TEXT)
        self.surface.blit(red_text, (cx - red_text.get_width() // 2, y_cursor))
        y_cursor += 35

        # ─── System Average Delay Graph ───
        graph_title = self.font_title.render("System Average Delay", True, COLOR_DASH_TEXT)
        self.surface.blit(graph_title, (cx - graph_title.get_width() // 2, y_cursor))
        y_cursor += 28

        self._draw_graph(env, y_cursor)
        y_cursor += 260

        # ─── Queue Lengths ───
        queues = env.get_queue_lengths()
        y_cursor += 5
        for d_label, d_key in [("NB Queue:", "North"), ("SB Queue:", "South"),
                                ("EB Queue:", "East"), ("WB Queue:", "West")]:
            val = queues[d_key]
            label = self.font_medium.render(d_label, True, COLOR_DASH_TEXT)
            value = self.font_value.render(str(val), True,
                                           COLOR_DASH_GREEN if val < 4 else
                                           COLOR_DASH_YELLOW if val < 8 else COLOR_DASH_RED)
            self.surface.blit(label, (cx - 50, y_cursor))
            self.surface.blit(value, (cx + 65, y_cursor))
            y_cursor += 26

        y_cursor += 10

        # ─── RL Stats ───
        stats = agent.get_stats()
        pygame.draw.line(self.surface, (60, 60, 70),
                         (self.x + 20, y_cursor), (self.x + self.w - 20, y_cursor), 1)
        y_cursor += 10

        rl_title = self.font_medium.render("RL Agent Status", True, COLOR_DASH_ACCENT)
        self.surface.blit(rl_title, (cx - rl_title.get_width() // 2, y_cursor))
        y_cursor += 25

        # Epsilon bar
        eps_label = self.font_small.render(
            f"Exploration (ε): {stats['epsilon']:.3f}", True, COLOR_DASH_TEXT)
        self.surface.blit(eps_label, (self.x + 30, y_cursor))
        y_cursor += 20

        bar_x = self.x + 30
        bar_w = self.w - 60
        bar_h = 12
        pygame.draw.rect(self.surface, (50, 50, 60), (bar_x, y_cursor, bar_w, bar_h),
                         border_radius=4)
        fill_w = int(bar_w * stats['epsilon'])
        bar_color = COLOR_DASH_ACCENT if stats['epsilon'] > 0.3 else COLOR_DASH_GREEN
        pygame.draw.rect(self.surface, bar_color, (bar_x, y_cursor, fill_w, bar_h),
                         border_radius=4)
        y_cursor += 22

        info_lines = [
            f"States visited: {stats['states_visited']}",
            f"Total decisions: {stats['total_actions']}",
        ]
        for line in info_lines:
            txt = self.font_small.render(line, True, (170, 170, 180))
            self.surface.blit(txt, (self.x + 30, y_cursor))
            y_cursor += 20

        y_cursor += 8

        # ─── Traffic Density Controls ───
        pygame.draw.line(self.surface, (60, 60, 70),
                         (self.x + 20, y_cursor), (self.x + self.w - 20, y_cursor), 1)
        y_cursor += 10
        ctrl_title = self.font_medium.render("Traffic Density (Keys 1-3)", True,
                                              COLOR_DASH_ACCENT)
        self.surface.blit(ctrl_title, (cx - ctrl_title.get_width() // 2, y_cursor))
        y_cursor += 22

        densities = [
            ("1 - Low", ARRIVAL_RATE_LOW),
            ("2 - Medium", ARRIVAL_RATE_MEDIUM),
            ("3 - High", ARRIVAL_RATE_HIGH),
        ]
        for label, rate in densities:
            is_active = abs(env.arrival_rate - rate) < 0.001
            color = COLOR_DASH_GREEN if is_active else (120, 120, 130)
            marker = "► " if is_active else "  "
            txt = self.font_small.render(f"{marker}{label}", True, color)
            self.surface.blit(txt, (self.x + 40, y_cursor))
            y_cursor += 18

    def _draw_timing_bar(self, env, y):
        """Draw the signal timing bar (green/red/black)."""
        bar_x = self.x + 30
        bar_w = self.w - 60
        bar_h = 22

        # Background
        pygame.draw.rect(self.surface, (15, 15, 15), (bar_x, y, bar_w, bar_h),
                         border_radius=3)

        # Calculate proportions
        total = env.green_accumulator + env.red_accumulator + 1
        if env.last_green_duration + env.last_red_duration > 0:
            total_time = env.last_green_duration + env.last_red_duration
            green_frac = env.last_green_duration / total_time
        else:
            green_frac = 0.5

        green_w = int(bar_w * green_frac)
        red_w = bar_w - green_w

        # Green portion
        if green_w > 0:
            pygame.draw.rect(self.surface, COLOR_DASH_GREEN,
                             (bar_x, y, green_w, bar_h), border_radius=3)
        # Red portion
        if red_w > 0:
            pygame.draw.rect(self.surface, COLOR_DASH_RED,
                             (bar_x + green_w, y, min(red_w, 30), bar_h))

        # Border
        pygame.draw.rect(self.surface, (80, 80, 90), (bar_x, y, bar_w, bar_h), 1,
                         border_radius=3)

    def _draw_graph(self, env, y):
        """Draw the average delay time series graph using Pygame primitives."""
        gx = self.x + 35
        gy = y
        gw = self.w - 70
        gh = 240

        # Graph background
        pygame.draw.rect(self.surface, (240, 240, 242), (gx, gy, gw, gh), border_radius=4)
        pygame.draw.rect(self.surface, (180, 180, 185), (gx, gy, gw, gh), 1, border_radius=4)

        # Inner plot area
        margin_l, margin_r, margin_t, margin_b = 50, 15, 10, 35
        px = gx + margin_l
        py = gy + margin_t
        pw = gw - margin_l - margin_r
        ph = gh - margin_t - margin_b

        # Plot bg
        pygame.draw.rect(self.surface, (255, 255, 255), (px, py, pw, ph))
        pygame.draw.rect(self.surface, (180, 180, 180), (px, py, pw, ph), 1)

        data = env.avg_delay_history
        if len(data) < 2:
            no_data = self.font_small.render("Collecting data...", True, (140, 140, 140))
            self.surface.blit(no_data, (px + pw // 2 - no_data.get_width() // 2,
                                        py + ph // 2 - 8))
            return

        # Auto-scale Y
        max_val = max(max(data), 1)
        # Round up to nice number
        nice_max = math.ceil(max_val / 10) * 10
        if nice_max == 0:
            nice_max = 10

        # Grid lines and Y-axis labels
        num_y_ticks = 5
        for i in range(num_y_ticks + 1):
            val = nice_max * i / num_y_ticks
            yy = py + ph - int(ph * (val / nice_max))
            # Grid line
            pygame.draw.line(self.surface, (220, 220, 220), (px, yy), (px + pw, yy), 1)
            # Label
            lbl = self.font_small.render(f"{int(val)}", True, (100, 100, 100))
            self.surface.blit(lbl, (px - lbl.get_width() - 5, yy - 7))

        # X-axis ticks
        num_points = len(data)
        x_max_seconds = num_points  # Each data point = 1 second
        x_max_hours = x_max_seconds / 3600
        if x_max_hours < 0.01:
            x_label_text = "Seconds"
            x_divisor = 1
        elif x_max_hours < 1:
            x_label_text = "Minutes"
            x_divisor = 60
        else:
            x_label_text = "Hours since Simulation Start (h)"
            x_divisor = 3600

        # X-axis label
        x_axis_lbl = self.font_small.render(x_label_text, True, (100, 100, 100))
        self.surface.blit(x_axis_lbl,
                          (px + pw // 2 - x_axis_lbl.get_width() // 2, py + ph + 16))

        # Y-axis label (rendered sideways)
        y_label = self.font_small.render("Avg Delay (s)", True, (100, 100, 100))
        y_label_rot = pygame.transform.rotate(y_label, 90)
        self.surface.blit(y_label_rot, (gx + 2, py + ph // 2 - y_label_rot.get_height() // 2))

        # Plot data points and lines
        points = []
        for i, val in enumerate(data):
            xx = px + int(pw * i / max(num_points - 1, 1))
            yy = py + ph - int(ph * min(val / nice_max, 1.0))
            points.append((xx, yy))

        # Line
        if len(points) >= 2:
            pygame.draw.lines(self.surface, (40, 120, 220), False, points, 2)

        # Dots (draw fewer if too many points)
        step = max(1, len(points) // 80)
        for i in range(0, len(points), step):
            pygame.draw.circle(self.surface, (40, 120, 220), points[i], 3)
            pygame.draw.circle(self.surface, (255, 255, 255), points[i], 1)

        # X ticks
        num_x_ticks = min(6, num_points)
        for i in range(num_x_ticks + 1):
            idx = int(num_points * i / max(num_x_ticks, 1))
            idx = min(idx, num_points - 1)
            xx = px + int(pw * idx / max(num_points - 1, 1))
            val = idx / x_divisor
            pygame.draw.line(self.surface, (180, 180, 180), (xx, py + ph), (xx, py + ph + 4), 1)
            lbl = self.font_small.render(f"{val:.0f}", True, (100, 100, 100))
            self.surface.blit(lbl, (xx - lbl.get_width() // 2, py + ph + 4))