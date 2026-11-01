"""
Renders a realistic satellite-view intersection using Pygame.
Styled to look like aerial/satellite imagery — NOT a game.
"""

import pygame
import math
import os
from config import *


class IntersectionRenderer:
    """Draws the intersection with roads, markings, vehicles, and signals."""

    def __init__(self, surface):
        self.surface = surface
        self.bg_image = None

        # Try loading a satellite background image
        asset_path = os.path.join(os.path.dirname(__file__), "assets", "intersection.png")
        if os.path.exists(asset_path):
            self.bg_image = pygame.image.load(asset_path)
            self.bg_image = pygame.transform.scale(self.bg_image, (SIM_WIDTH, WINDOW_HEIGHT))

        # Pre-render noise texture for ground realism
        self._build_ground_texture()

    def _build_ground_texture(self):
        """Create a subtle noise texture overlay for realism."""
        self.ground_surface = pygame.Surface((SIM_WIDTH, WINDOW_HEIGHT))
        self.ground_surface.fill(COLOR_GRASS)

        import random as rnd
        for _ in range(8000):
            x = rnd.randint(0, SIM_WIDTH - 1)
            y = rnd.randint(0, WINDOW_HEIGHT - 1)
            shade = rnd.randint(-15, 15)
            c = tuple(max(0, min(255, COLOR_GRASS[i] + shade)) for i in range(3))
            self.ground_surface.set_at((x, y), c)

        for _ in range(40):
            x = rnd.randint(0, SIM_WIDTH)
            y = rnd.randint(0, WINDOW_HEIGHT)
            r = rnd.randint(15, 50)
            shade = rnd.randint(-25, -10)
            c = tuple(max(0, min(255, COLOR_GRASS[i] + shade)) for i in range(3))
            pygame.draw.circle(self.ground_surface, c, (x, y), r)

    def draw(self, env):
        """Render the full intersection scene."""
        if self.bg_image:
            self.surface.blit(self.bg_image, (0, 0))
        else:
            self.surface.blit(self.ground_surface, (0, 0))

        self._draw_roads()
        self._draw_lane_markings()
        self._draw_crosswalks()
        self._draw_stop_lines(env)
        self._draw_traffic_lights(env)
        self._draw_vehicles(env)

    def _draw_roads(self):
        sw = 6
        pygame.draw.rect(self.surface, COLOR_SIDEWALK,
                         (CENTER_X - ROAD_WIDTH // 2 - sw, 0,
                          ROAD_WIDTH + sw * 2, WINDOW_HEIGHT))
        pygame.draw.rect(self.surface, COLOR_SIDEWALK,
                         (0, CENTER_Y - ROAD_WIDTH // 2 - sw,
                          SIM_WIDTH, ROAD_WIDTH + sw * 2))
        pygame.draw.rect(self.surface, COLOR_ROAD,
                         (CENTER_X - ROAD_WIDTH // 2, 0,
                          ROAD_WIDTH, WINDOW_HEIGHT))
        pygame.draw.rect(self.surface, COLOR_ROAD,
                         (0, CENTER_Y - ROAD_WIDTH // 2,
                          SIM_WIDTH, ROAD_WIDTH))
        pygame.draw.rect(self.surface, COLOR_ROAD_DARK,
                         (CENTER_X - ROAD_WIDTH // 2,
                          CENTER_Y - ROAD_WIDTH // 2,
                          ROAD_WIDTH, ROAD_WIDTH))
        edge_color = (90, 90, 90)
        pygame.draw.line(self.surface, edge_color,
                         (CENTER_X - ROAD_WIDTH // 2, 0),
                         (CENTER_X - ROAD_WIDTH // 2, WINDOW_HEIGHT), 2)
        pygame.draw.line(self.surface, edge_color,
                         (CENTER_X + ROAD_WIDTH // 2, 0),
                         (CENTER_X + ROAD_WIDTH // 2, WINDOW_HEIGHT), 2)
        pygame.draw.line(self.surface, edge_color,
                         (0, CENTER_Y - ROAD_WIDTH // 2),
                         (SIM_WIDTH, CENTER_Y - ROAD_WIDTH // 2), 2)
        pygame.draw.line(self.surface, edge_color,
                         (0, CENTER_Y + ROAD_WIDTH // 2),
                         (SIM_WIDTH, CENTER_Y + ROAD_WIDTH // 2), 2)

    def _draw_lane_markings(self):
        dash_len = 20
        gap_len = 15
        x = CENTER_X
        for y in range(0, WINDOW_HEIGHT, dash_len + gap_len):
            if CENTER_Y - ROAD_WIDTH // 2 < y < CENTER_Y + ROAD_WIDTH // 2:
                continue
            pygame.draw.line(self.surface, COLOR_LANE_MARKING_YELLOW,
                             (x - 1, y), (x - 1, min(y + dash_len, WINDOW_HEIGHT)), 2)
            pygame.draw.line(self.surface, COLOR_LANE_MARKING_YELLOW,
                             (x + 1, y), (x + 1, min(y + dash_len, WINDOW_HEIGHT)), 2)
        y = CENTER_Y
        for x in range(0, SIM_WIDTH, dash_len + gap_len):
            if CENTER_X - ROAD_WIDTH // 2 < x < CENTER_X + ROAD_WIDTH // 2:
                continue
            pygame.draw.line(self.surface, COLOR_LANE_MARKING_YELLOW,
                             (x, y - 1), (min(x + dash_len, SIM_WIDTH), y - 1), 2)
            pygame.draw.line(self.surface, COLOR_LANE_MARKING_YELLOW,
                             (x, y + 1), (min(x + dash_len, SIM_WIDTH), y + 1), 2)
        dash_len_w = 12
        gap_len_w = 18
        white = COLOR_LANE_MARKING
        for offset in [-ROAD_WIDTH // 4, ROAD_WIDTH // 4]:
            x = CENTER_X + offset
            for y in range(0, WINDOW_HEIGHT, dash_len_w + gap_len_w):
                if CENTER_Y - ROAD_WIDTH // 2 - 5 < y < CENTER_Y + ROAD_WIDTH // 2 + 5:
                    continue
                pygame.draw.line(self.surface, white,
                                 (x, y), (x, min(y + dash_len_w, WINDOW_HEIGHT)), 1)
        for offset in [-ROAD_WIDTH // 4, ROAD_WIDTH // 4]:
            y = CENTER_Y + offset
            for x in range(0, SIM_WIDTH, dash_len_w + gap_len_w):
                if CENTER_X - ROAD_WIDTH // 2 - 5 < x < CENTER_X + ROAD_WIDTH // 2 + 5:
                    continue
                pygame.draw.line(self.surface, white,
                                 (x, y), (min(x + dash_len_w, SIM_WIDTH), y), 1)

    def _draw_crosswalks(self):
        stripe_w = 4
        stripe_gap = 6
        cw_offset = ROAD_WIDTH // 2 + 3
        y_start = CENTER_Y - cw_offset - 14
        for x in range(CENTER_X - ROAD_WIDTH // 2 + 4,
                        CENTER_X + ROAD_WIDTH // 2 - 4, stripe_w + stripe_gap):
            pygame.draw.rect(self.surface, COLOR_CROSSWALK,
                             (x, y_start, stripe_w, 12))
        y_start = CENTER_Y + cw_offset + 2
        for x in range(CENTER_X - ROAD_WIDTH // 2 + 4,
                        CENTER_X + ROAD_WIDTH // 2 - 4, stripe_w + stripe_gap):
            pygame.draw.rect(self.surface, COLOR_CROSSWALK,
                             (x, y_start, stripe_w, 12))
        x_start = CENTER_X - cw_offset - 14
        for y in range(CENTER_Y - ROAD_WIDTH // 2 + 4,
                        CENTER_Y + ROAD_WIDTH // 2 - 4, stripe_w + stripe_gap):
            pygame.draw.rect(self.surface, COLOR_CROSSWALK,
                             (x_start, y, 12, stripe_w))
        x_start = CENTER_X + cw_offset + 2
        for y in range(CENTER_Y - ROAD_WIDTH // 2 + 4,
                        CENTER_Y + ROAD_WIDTH // 2 - 4, stripe_w + stripe_gap):
            pygame.draw.rect(self.surface, COLOR_CROSSWALK,
                             (x_start, y, 12, stripe_w))

    def _draw_stop_lines(self, env):
        color = (200, 200, 200)
        sl_offset = ROAD_WIDTH // 2 + 2
        pygame.draw.line(self.surface, color,
                         (CENTER_X + MEDIAN_WIDTH // 2, CENTER_Y - sl_offset),
                         (CENTER_X + ROAD_WIDTH // 2 - 2, CENTER_Y - sl_offset), 3)
        pygame.draw.line(self.surface, color,
                         (CENTER_X - ROAD_WIDTH // 2 + 2, CENTER_Y + sl_offset),
                         (CENTER_X - MEDIAN_WIDTH // 2, CENTER_Y + sl_offset), 3)
        pygame.draw.line(self.surface, color,
                         (CENTER_X + sl_offset, CENTER_Y + MEDIAN_WIDTH // 2),
                         (CENTER_X + sl_offset, CENTER_Y + ROAD_WIDTH // 2 - 2), 3)
        pygame.draw.line(self.surface, color,
                         (CENTER_X - sl_offset, CENTER_Y - ROAD_WIDTH // 2 + 2),
                         (CENTER_X - sl_offset, CENTER_Y - MEDIAN_WIDTH // 2), 3)

    def _draw_traffic_lights(self, env):
        _, is_yellow, green_dirs = env.get_signal_state()
        light_size = 10
        offset = ROAD_WIDTH // 2 + 18
        ns_color = COLOR_YELLOW_LIGHT if is_yellow else (
            COLOR_GREEN_LIGHT if "North" in green_dirs else COLOR_RED_LIGHT)
        ew_color = COLOR_YELLOW_LIGHT if is_yellow else (
            COLOR_GREEN_LIGHT if "East" in green_dirs else COLOR_RED_LIGHT)
        positions_ns = [
            (CENTER_X + offset, CENTER_Y - offset),
            (CENTER_X - offset, CENTER_Y + offset),
        ]
        positions_ew = [
            (CENTER_X + offset, CENTER_Y + offset),
            (CENTER_X - offset, CENTER_Y - offset),
        ]
        for (px, py) in positions_ns:
            pygame.draw.rect(self.surface, (30, 30, 30),
                             (px - light_size - 2, py - light_size - 2,
                              light_size * 2 + 4, light_size * 2 + 4),
                             border_radius=3)
            pygame.draw.circle(self.surface, ns_color, (px, py), light_size)
            glow = pygame.Surface((light_size * 4, light_size * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*ns_color, 40),
                               (light_size * 2, light_size * 2), light_size * 2)
            self.surface.blit(glow, (px - light_size * 2, py - light_size * 2))
        for (px, py) in positions_ew:
            pygame.draw.rect(self.surface, (30, 30, 30),
                             (px - light_size - 2, py - light_size - 2,
                              light_size * 2 + 4, light_size * 2 + 4),
                             border_radius=3)
            pygame.draw.circle(self.surface, ew_color, (px, py), light_size)
            glow = pygame.Surface((light_size * 4, light_size * 4), pygame.SRCALPHA)
            pygame.draw.circle(glow, (*ew_color, 40),
                               (light_size * 2, light_size * 2), light_size * 2)
            self.surface.blit(glow, (px - light_size * 2, py - light_size * 2))

    def _draw_vehicles(self, env):
        for direction in DIRECTIONS:
            for v in env.vehicles[direction]:
                coords = v.get_rect_coords()
                if coords is None:
                    continue
                x, y, w, h, angle = coords
                if x < -50 or x > SIM_WIDTH + 50 or y < -50 or y > WINDOW_HEIGHT + 50:
                    continue
                shadow = pygame.Surface((w + 2, h + 2), pygame.SRCALPHA)
                shadow.fill((0, 0, 0, 50))
                self.surface.blit(shadow, (x + 2, y + 2))
                pygame.draw.rect(self.surface, v.color, (x, y, w, h), border_radius=2)
                if h > w:
                    detail_h = h // 3
                    detail_y = y + h // 3
                    darker = tuple(max(0, c - 40) for c in v.color)
                    pygame.draw.rect(self.surface, darker,
                                     (x + 1, detail_y, w - 2, detail_h), border_radius=1)
                else:
                    detail_w = w // 3
                    detail_x = x + w // 3
                    darker = tuple(max(0, c - 40) for c in v.color)
                    pygame.draw.rect(self.surface, darker,
                                     (detail_x, y + 1, detail_w, h - 2), border_radius=1)
                pygame.draw.rect(self.surface, (20, 20, 20), (x, y, w, h), 1,
                                 border_radius=2)