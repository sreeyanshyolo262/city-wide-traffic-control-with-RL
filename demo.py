"""
Traffic Signal RL Demo - Satellite Image View
Shows trained agent on a real aerial intersection image.

Controls:
  UP/DOWN/LEFT/RIGHT + number — Add cars to that direction
  Click lane buttons — Spawn cars
  SPACE — Pause/Resume
  R — Reset all cars
  ESC — Quit
"""

import os
import pygame
import sys
from collections import defaultdict
from config import *
from agent import QLearningAgent


class Vehicle:
    """A vehicle that moves through the intersection."""
    
    def __init__(self, direction, lane, turn="straight"):
        self.direction = direction  # "North", "South", "East", "West"
        self.lane = lane  # 0=left turn, 1=straight, 2=right turn
        self.turn = turn  # "left", "straight", "right"
        self.color = CAR_COLORS[hash((direction, lane)) % len(CAR_COLORS)]
        
        # Position based on direction and lane
        self.x, self.y = self._get_spawn_position()
        self.target_x, self.target_y = self.x, self.y
        self.passed_intersection = False
        self.waiting = True
        self.wait_time = 0
        self.speed = 3
        
    def _get_spawn_position(self):
        """Get spawn position based on direction and lane."""
        # Offsets for lanes (from center of road)
        lane_offset = (self.lane - 1) * 28
        
        if self.direction == "North":  # Coming from top, going down
            return CENTER_X + lane_offset + 20, 50
        elif self.direction == "South":  # Coming from bottom, going up
            return CENTER_X - lane_offset - 20, WINDOW_HEIGHT - 50
        elif self.direction == "East":  # Coming from right, going left
            return SIM_WIDTH - 50, CENTER_Y + lane_offset + 20
        else:  # West - Coming from left, going right
            return 50, CENTER_Y - lane_offset - 20
            
    def get_stop_position(self):
        """Get position where car should stop at red light."""
        if self.direction == "North":
            return self.x, CENTER_Y - ROAD_WIDTH - 30
        elif self.direction == "South":
            return self.x, CENTER_Y + ROAD_WIDTH + 30
        elif self.direction == "East":
            return CENTER_X + ROAD_WIDTH + 30, self.y
        else:  # West
            return CENTER_X - ROAD_WIDTH - 30, self.y
            
    def get_exit_position(self):
        """Get position where car exits based on turn."""
        if self.turn == "straight":
            if self.direction == "North":
                return self.x, WINDOW_HEIGHT + 50
            elif self.direction == "South":
                return self.x, -50
            elif self.direction == "East":
                return -50, self.y
            else:
                return SIM_WIDTH + 50, self.y
        elif self.turn == "left":
            if self.direction == "North":
                return -50, CENTER_Y - 40
            elif self.direction == "South":
                return SIM_WIDTH + 50, CENTER_Y + 40
            elif self.direction == "East":
                return CENTER_X + 40, -50
            else:
                return CENTER_X - 40, WINDOW_HEIGHT + 50
        else:  # right
            if self.direction == "North":
                return SIM_WIDTH + 50, CENTER_Y + 40
            elif self.direction == "South":
                return -50, CENTER_Y - 40
            elif self.direction == "East":
                return CENTER_X - 40, WINDOW_HEIGHT + 50
            else:
                return CENTER_X + 40, -50
                
    def update(self, can_go, cars_ahead):
        """Update car position."""
        stop_x, stop_y = self.get_stop_position()
        
        # Check if at intersection
        at_stop = False
        if self.direction in ["North", "South"]:
            at_stop = abs(self.y - stop_y) < 10
        else:
            at_stop = abs(self.x - stop_x) < 10
            
        if not self.passed_intersection:
            if at_stop and not can_go:
                # Wait at stop line
                self.waiting = True
                self.wait_time += 1
                return
            elif at_stop and can_go:
                self.passed_intersection = True
                self.waiting = False
                exit_x, exit_y = self.get_exit_position()
                self.target_x, self.target_y = exit_x, exit_y
            else:
                # Move towards stop line (check cars ahead)
                if cars_ahead > 0:
                    self.target_x, self.target_y = stop_x, stop_y
                    # Adjust for cars ahead
                    if self.direction == "North":
                        self.target_y = stop_y - cars_ahead * (CAR_LENGTH + CAR_GAP)
                    elif self.direction == "South":
                        self.target_y = stop_y + cars_ahead * (CAR_LENGTH + CAR_GAP)
                    elif self.direction == "East":
                        self.target_x = stop_x + cars_ahead * (CAR_LENGTH + CAR_GAP)
                    else:
                        self.target_x = stop_x - cars_ahead * (CAR_LENGTH + CAR_GAP)
                else:
                    self.target_x, self.target_y = stop_x, stop_y
                    
        # Move towards target
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = (dx**2 + dy**2)**0.5
        
        if dist > self.speed:
            self.x += (dx / dist) * self.speed
            self.y += (dy / dist) * self.speed
            self.waiting = False
        else:
            self.x, self.y = self.target_x, self.target_y
            
    def is_offscreen(self):
        """Check if car has left the screen."""
        return (self.x < -60 or self.x > SIM_WIDTH + 60 or 
                self.y < -60 or self.y > WINDOW_HEIGHT + 60)
                
    def draw(self, screen):
        """Draw the car."""
        # Determine rotation based on direction and turn
        if self.direction == "North":
            angle = 180
        elif self.direction == "South":
            angle = 0
        elif self.direction == "East":
            angle = 90
        else:
            angle = 270
            
        # Draw car body
        car_surf = pygame.Surface((CAR_WIDTH, CAR_LENGTH), pygame.SRCALPHA)
        pygame.draw.rect(car_surf, self.color, (0, 0, CAR_WIDTH, CAR_LENGTH), border_radius=3)
        # Windshield
        pygame.draw.rect(car_surf, (60, 80, 100), (2, 3, CAR_WIDTH-4, 6), border_radius=2)
        
        rotated = pygame.transform.rotate(car_surf, angle)
        rect = rotated.get_rect(center=(int(self.x), int(self.y)))
        screen.blit(rotated, rect)


class DemoSimulation:
    """Demo simulation with satellite background."""
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Traffic Signal RL Demo - Trained Agent | Click buttons to spawn cars")
        self.clock = pygame.time.Clock()
        
        # Load background image
        self.bg_image = None
        bg_path = os.path.join(os.path.dirname(__file__), 'assets', 'intersection_bg.jpg')
        if os.path.exists(bg_path):
            try:
                img = pygame.image.load(bg_path)
                self.bg_image = pygame.transform.scale(img, (SIM_WIDTH, WINDOW_HEIGHT))
            except:
                print("Could not load background image")
                
        # Fonts
        self.font_title = pygame.font.SysFont("Arial", 24, bold=True)
        self.font_btn = pygame.font.SysFont("Arial", 16, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 14)
        self.font_large = pygame.font.SysFont("Arial", 48, bold=True)
        
        # Traffic state
        self.vehicles = []
        self.signal_phase = 0  # 0 = NS green, 1 = EW green
        self.phase_timer = 0
        
        # Spawn buttons (direction -> (x, y, w, h))
        self.spawn_buttons = self._create_spawn_buttons()
        
        # Agent
        self.agent = QLearningAgent()
        self.agent.load("trained_agent.pkl")
        self.agent.set_evaluation_mode()
        
        # Stats
        self.total_passed = 0
        self.total_wait = 0
        
        # Colors
        self.colors = {
            "bg": (25, 25, 30),
            "panel": (35, 35, 40),
            "text": (220, 220, 220),
            "green": (0, 200, 80),
            "red": (220, 50, 50),
            "yellow": (220, 200, 60),
            "accent": (80, 180, 255),
        }
        
    def _create_spawn_buttons(self):
        """Create spawn buttons for each direction/lane."""
        buttons = {}
        btn_w, btn_h = 60, 30
        
        # North buttons (top of screen)
        buttons[("North", 0, "left")] = pygame.Rect(CENTER_X - 100, 10, btn_w, btn_h)
        buttons[("North", 1, "straight")] = pygame.Rect(CENTER_X - 30, 10, btn_w, btn_h)
        buttons[("North", 2, "right")] = pygame.Rect(CENTER_X + 40, 10, btn_w, btn_h)
        
        # South buttons (bottom of sim area)
        buttons[("South", 0, "left")] = pygame.Rect(CENTER_X + 40, WINDOW_HEIGHT - 45, btn_w, btn_h)
        buttons[("South", 1, "straight")] = pygame.Rect(CENTER_X - 30, WINDOW_HEIGHT - 45, btn_w, btn_h)
        buttons[("South", 2, "right")] = pygame.Rect(CENTER_X - 100, WINDOW_HEIGHT - 45, btn_w, btn_h)
        
        # East buttons (right side)
        buttons[("East", 0, "left")] = pygame.Rect(SIM_WIDTH - 75, CENTER_Y - 80, btn_w, btn_h)
        buttons[("East", 1, "straight")] = pygame.Rect(SIM_WIDTH - 75, CENTER_Y - 10, btn_w, btn_h)
        buttons[("East", 2, "right")] = pygame.Rect(SIM_WIDTH - 75, CENTER_Y + 60, btn_w, btn_h)
        
        # West buttons (left side)
        buttons[("West", 0, "left")] = pygame.Rect(10, CENTER_Y + 60, btn_w, btn_h)
        buttons[("West", 1, "straight")] = pygame.Rect(10, CENTER_Y - 10, btn_w, btn_h)
        buttons[("West", 2, "right")] = pygame.Rect(10, CENTER_Y - 80, btn_w, btn_h)
        
        return buttons
        
    def spawn_car(self, direction, lane, turn):
        """Spawn a new car."""
        car = Vehicle(direction, lane, turn)
        self.vehicles.append(car)
        
    def get_state(self):
        """Get state for RL agent."""
        def bucket(n):
            if n == 0: return 0
            if n <= 3: return 1
            if n <= 7: return 2
            return 3
            
        ns_queue = sum(1 for v in self.vehicles if v.direction in ["North", "South"] and v.waiting)
        ew_queue = sum(1 for v in self.vehicles if v.direction in ["East", "West"] and v.waiting)
        
        return (bucket(ns_queue), bucket(ew_queue), self.signal_phase)
        
    def get_queue_counts(self):
        """Get queue counts per direction."""
        counts = {"North": 0, "South": 0, "East": 0, "West": 0}
        for v in self.vehicles:
            if v.waiting and not v.passed_intersection:
                counts[v.direction] += 1
        return counts
        
    def update(self):
        """Update simulation state."""
        self.phase_timer += 1
        
        # Agent makes decision every ~2 seconds (60 frames)
        if self.phase_timer >= 60:
            self.phase_timer = 0
            state = self.get_state()
            action = self.agent.choose_action(state)
            
            # Only switch if needed and enough time has passed
            if action != self.signal_phase:
                self.signal_phase = action
                
        # Update vehicles
        for v in self.vehicles:
            # Check if this car can go
            can_go = False
            if self.signal_phase == 0 and v.direction in ["North", "South"]:
                can_go = True
            elif self.signal_phase == 1 and v.direction in ["East", "West"]:
                can_go = True
                
            # Count cars ahead in same lane
            cars_ahead = 0
            for other in self.vehicles:
                if other is v:
                    continue
                if other.direction == v.direction and other.lane == v.lane:
                    if not other.passed_intersection:
                        # Check if other is ahead
                        if v.direction == "North" and other.y > v.y:
                            cars_ahead += 1
                        elif v.direction == "South" and other.y < v.y:
                            cars_ahead += 1
                        elif v.direction == "East" and other.x < v.x:
                            cars_ahead += 1
                        elif v.direction == "West" and other.x > v.x:
                            cars_ahead += 1
                            
            v.update(can_go, cars_ahead)
            
            if v.waiting:
                self.total_wait += 1
                
        # Remove off-screen vehicles
        before = len(self.vehicles)
        self.vehicles = [v for v in self.vehicles if not v.is_offscreen()]
        self.total_passed += before - len(self.vehicles)
        
    def draw(self):
        """Draw everything."""
        # Clear screen
        self.screen.fill(self.colors["bg"])
        
        # Draw simulation area
        if self.bg_image:
            self.screen.blit(self.bg_image, (0, 0))
        else:
            # Draw simple intersection if no image
            self._draw_simple_intersection()
            
        # Draw traffic lights
        self._draw_traffic_lights()
        
        # Draw vehicles
        for v in self.vehicles:
            v.draw(self.screen)
            
        # Draw spawn buttons
        self._draw_spawn_buttons()
        
        # Draw dashboard panel
        self._draw_dashboard()
        
        pygame.display.flip()
        
    def _draw_simple_intersection(self):
        """Draw a simple intersection if no background image."""
        # Grass
        self.screen.fill(COLOR_GRASS, (0, 0, SIM_WIDTH, WINDOW_HEIGHT))
        
        # Roads
        pygame.draw.rect(self.screen, COLOR_ROAD, 
                        (CENTER_X - ROAD_WIDTH, 0, ROAD_WIDTH * 2, WINDOW_HEIGHT))
        pygame.draw.rect(self.screen, COLOR_ROAD,
                        (0, CENTER_Y - ROAD_WIDTH, SIM_WIDTH, ROAD_WIDTH * 2))
                        
        # Center line (yellow)
        pygame.draw.line(self.screen, COLOR_LANE_MARKING_YELLOW,
                        (CENTER_X, 0), (CENTER_X, CENTER_Y - ROAD_WIDTH), 3)
        pygame.draw.line(self.screen, COLOR_LANE_MARKING_YELLOW,
                        (CENTER_X, CENTER_Y + ROAD_WIDTH), (CENTER_X, WINDOW_HEIGHT), 3)
        pygame.draw.line(self.screen, COLOR_LANE_MARKING_YELLOW,
                        (0, CENTER_Y), (CENTER_X - ROAD_WIDTH, CENTER_Y), 3)
        pygame.draw.line(self.screen, COLOR_LANE_MARKING_YELLOW,
                        (CENTER_X + ROAD_WIDTH, CENTER_Y), (SIM_WIDTH, CENTER_Y), 3)
                        
        # Stop lines
        pygame.draw.line(self.screen, COLOR_STOP_LINE,
                        (CENTER_X, CENTER_Y - ROAD_WIDTH), 
                        (CENTER_X + ROAD_WIDTH, CENTER_Y - ROAD_WIDTH), 4)
        pygame.draw.line(self.screen, COLOR_STOP_LINE,
                        (CENTER_X - ROAD_WIDTH, CENTER_Y + ROAD_WIDTH),
                        (CENTER_X, CENTER_Y + ROAD_WIDTH), 4)
        pygame.draw.line(self.screen, COLOR_STOP_LINE,
                        (CENTER_X + ROAD_WIDTH, CENTER_Y),
                        (CENTER_X + ROAD_WIDTH, CENTER_Y + ROAD_WIDTH), 4)
        pygame.draw.line(self.screen, COLOR_STOP_LINE,
                        (CENTER_X - ROAD_WIDTH, CENTER_Y - ROAD_WIDTH),
                        (CENTER_X - ROAD_WIDTH, CENTER_Y), 4)
                        
    def _draw_traffic_lights(self):
        """Draw traffic light indicators."""
        ns_color = self.colors["green"] if self.signal_phase == 0 else self.colors["red"]
        ew_color = self.colors["green"] if self.signal_phase == 1 else self.colors["red"]
        
        # North light
        pygame.draw.circle(self.screen, ns_color, (CENTER_X + ROAD_WIDTH + 15, CENTER_Y - ROAD_WIDTH - 15), 12)
        # South light  
        pygame.draw.circle(self.screen, ns_color, (CENTER_X - ROAD_WIDTH - 15, CENTER_Y + ROAD_WIDTH + 15), 12)
        # East light
        pygame.draw.circle(self.screen, ew_color, (CENTER_X + ROAD_WIDTH + 15, CENTER_Y + ROAD_WIDTH + 15), 12)
        # West light
        pygame.draw.circle(self.screen, ew_color, (CENTER_X - ROAD_WIDTH - 15, CENTER_Y - ROAD_WIDTH - 15), 12)
        
    def _draw_spawn_buttons(self):
        """Draw clickable spawn buttons."""
        for (direction, lane, turn), rect in self.spawn_buttons.items():
            # Button background
            color = (60, 60, 70)
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            pygame.draw.rect(self.screen, self.colors["accent"], rect, 2, border_radius=5)
            
            # Arrow or text
            label = "←" if turn == "left" else ("↑" if turn == "straight" else "→")
            if direction in ["East", "West"]:
                label = "↑" if turn == "left" else ("→" if turn == "straight" else "↓")
                if direction == "West":
                    label = "↓" if turn == "left" else ("←" if turn == "straight" else "↑")
                    
            txt = self.font_btn.render(f"+{label}", True, self.colors["text"])
            self.screen.blit(txt, (rect.centerx - txt.get_width()//2, 
                                   rect.centery - txt.get_height()//2))
                                   
    def _draw_dashboard(self):
        """Draw the stats dashboard on the right."""
        panel_x = SIM_WIDTH
        panel_w = DASH_WIDTH
        
        # Background
        pygame.draw.rect(self.screen, self.colors["panel"], 
                        (panel_x, 0, panel_w, WINDOW_HEIGHT))
                        
        y = 30
        
        # Title
        title = self.font_title.render("Trained Model Demo", True, self.colors["accent"])
        self.screen.blit(title, (panel_x + panel_w//2 - title.get_width()//2, y))
        y += 50
        
        # Current signal
        phase_txt = "NS Green" if self.signal_phase == 0 else "EW Green"
        phase_color = self.colors["green"]
        signal = self.font_large.render(phase_txt, True, phase_color)
        self.screen.blit(signal, (panel_x + panel_w//2 - signal.get_width()//2, y))
        y += 70
        
        # Queue counts
        queues = self.get_queue_counts()
        
        subtitle = self.font_title.render("Queue Lengths", True, self.colors["text"])
        self.screen.blit(subtitle, (panel_x + 20, y))
        y += 35
        
        for direction in ["North", "South", "East", "West"]:
            count = queues[direction]
            label = f"  {direction}bound: {count}"
            color = self.colors["red"] if count > 5 else self.colors["text"]
            txt = self.font_btn.render(label, True, color)
            self.screen.blit(txt, (panel_x + 30, y))
            y += 28
            
        y += 30
        
        # Stats
        subtitle2 = self.font_title.render("Statistics", True, self.colors["text"])
        self.screen.blit(subtitle2, (panel_x + 20, y))
        y += 35
        
        stats = [
            f"  Vehicles Passed: {self.total_passed}",
            f"  Active Vehicles: {len(self.vehicles)}",
            f"  Agent States: {len(self.agent.q_table)}",
        ]
        
        for s in stats:
            txt = self.font_btn.render(s, True, self.colors["text"])
            self.screen.blit(txt, (panel_x + 30, y))
            y += 28
            
        y += 40
        
        # Instructions
        subtitle3 = self.font_title.render("Controls", True, self.colors["text"])
        self.screen.blit(subtitle3, (panel_x + 20, y))
        y += 35
        
        instructions = [
            "Click buttons to spawn cars",
            "SPACE = Pause/Resume",
            "R = Reset all vehicles", 
            "1/2/3 = Add 5/10/20 cars",
            "ESC = Quit",
        ]
        
        for inst in instructions:
            txt = self.font_small.render(inst, True, (150, 150, 150))
            self.screen.blit(txt, (panel_x + 30, y))
            y += 24
            
        # Phase indicator
        y = WINDOW_HEIGHT - 100
        phase_name = self.agent.get_stats()["phase_name"]
        phase_txt = self.font_large.render(phase_name, True, self.colors["green"])
        self.screen.blit(phase_txt, (panel_x + panel_w//2 - phase_txt.get_width()//2, y))
        
    def handle_click(self, pos):
        """Handle mouse click on spawn buttons."""
        for (direction, lane, turn), rect in self.spawn_buttons.items():
            if rect.collidepoint(pos):
                self.spawn_car(direction, lane, turn)
                return True
        return False
        
    def add_random_cars(self, count):
        """Add random cars to test."""
        import random
        directions = ["North", "South", "East", "West"]
        turns = ["left", "straight", "straight", "straight", "right"]  # More straight
        
        for _ in range(count):
            d = random.choice(directions)
            t = random.choice(turns)
            lane = 0 if t == "left" else (2 if t == "right" else 1)
            self.spawn_car(d, lane, t)
            
    def run(self):
        """Main loop."""
        running = True
        paused = False
        
        print("\n" + "="*50)
        print("  TRAFFIC SIGNAL RL - TRAINED MODEL DEMO")
        print("="*50)
        print("\nClick the +← +↑ +→ buttons to spawn cars")
        print("Watch the trained agent control the signals!\n")
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        self.vehicles = []
                        self.total_passed = 0
                    elif event.key == pygame.K_1:
                        self.add_random_cars(5)
                    elif event.key == pygame.K_2:
                        self.add_random_cars(10)
                    elif event.key == pygame.K_3:
                        self.add_random_cars(20)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)
                        
            if not paused:
                self.update()
                
            self.draw()
            
            if paused:
                # Draw pause overlay
                overlay = pygame.Surface((SIM_WIDTH, WINDOW_HEIGHT), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 100))
                self.screen.blit(overlay, (0, 0))
                pause_txt = self.font_large.render("PAUSED", True, (255, 255, 255))
                self.screen.blit(pause_txt, (CENTER_X - pause_txt.get_width()//2,
                                             CENTER_Y - pause_txt.get_height()//2))
                pygame.display.flip()
                
            self.clock.tick(FPS)
            
        pygame.quit()


def main():
    demo = DemoSimulation()
    demo.run()


if __name__ == "__main__":
    main()
