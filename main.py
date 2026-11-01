"""
Traffic Signal Control with Reinforcement Learning
Main application — launches the simulation with Pygame visualization.

Controls:
  1 / 2 / 3 — Set traffic density (Low / Medium / High)
  SPACE      — Pause / Resume
  S          — Save trained agent
  L          — Load trained agent
  R          — Reset simulation
  ESC        — Quit
"""

import os
import pygame
import sys
import argparse
from config import *
from environment import TrafficEnvironment
from agent import QLearningAgent
from renderer import IntersectionRenderer
from dashboard import Dashboard


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Traffic Signal RL Simulation")
    parser.add_argument("--load", action="store_true", help="Load trained agent")
    parser.add_argument("--model", type=str, default="trained_agent.pkl", help="Model path")
    args = parser.parse_args()
    
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(
        "Traffic Signal Control — RL  |  1/2/3: Density  |  Space: Pause  |  S: Save  |  L: Load  |  R: Reset")
    clock = pygame.time.Clock()

    # Create components
    env = TrafficEnvironment()
    agent = QLearningAgent()
    
    # Load trained agent if requested
    if args.load and os.path.exists(args.model):
        agent.load(args.model)
        agent.set_evaluation_mode()
        print("Running in evaluation mode with trained agent.")
    
    renderer = IntersectionRenderer(screen)
    dashboard = Dashboard(screen, SIM_WIDTH)

    font_phase = pygame.font.SysFont("Arial", 52, bold=True)
    font_phase_desc = pygame.font.SysFont("Arial", 18)
    font_msg = pygame.font.SysFont("Arial", 24, bold=True)
    
    # Message overlay state
    message = ""
    message_timer = 0

    paused = False
    decision_counter = 0
    DECISION_INTERVAL = PHASE_DURATION  # Make a decision every phase

    prev_state = env.get_state()
    prev_action = 0

    running = True
    while running:
        # ─── Event Handling ───
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    env = TrafficEnvironment()
                    agent = QLearningAgent()
                    prev_state = env.get_state()
                    prev_action = 0
                    message = "Simulation Reset"
                    message_timer = 60
                elif event.key == pygame.K_s:
                    agent.save(args.model)
                    message = f"Agent Saved ({len(agent.q_table)} states)"
                    message_timer = 90
                elif event.key == pygame.K_l:
                    if agent.load(args.model):
                        message = f"Agent Loaded ({len(agent.q_table)} states)"
                    else:
                        message = "No saved agent found"
                    message_timer = 90
                elif event.key == pygame.K_1:
                    env.set_arrival_rate(ARRIVAL_RATE_LOW)
                    message = "Traffic: LOW"
                    message_timer = 60
                elif event.key == pygame.K_2:
                    env.set_arrival_rate(ARRIVAL_RATE_MEDIUM)
                    message = "Traffic: MEDIUM"
                    message_timer = 60
                elif event.key == pygame.K_3:
                    env.set_arrival_rate(ARRIVAL_RATE_HIGH)
                    message = "Traffic: HIGH"
                    message_timer = 60

        if paused:
            # Draw "PAUSED" overlay
            pause_txt = font_phase.render("PAUSED", True, (255, 255, 255))
            screen.blit(pause_txt, (CENTER_X - pause_txt.get_width() // 2,
                                    CENTER_Y - pause_txt.get_height() // 2))
            pygame.display.flip()
            clock.tick(10)
            continue

        # ─── RL Decision ───
        decision_counter += 1
        action = None
        if decision_counter >= DECISION_INTERVAL:
            decision_counter = 0
            state = env.get_state()
            action = agent.choose_action(state)
            # Learn from previous transition
            reward = env.step(action)
            next_state = env.get_state()
            agent.learn(prev_state, prev_action, reward, next_state)
            prev_state = state
            prev_action = action
        else:
            env.step()

        # ─── Render ───
        screen.fill((0, 0, 0))

        # Intersection
        renderer.draw(env)

        # Phase label overlay on sim (bottom-left)
        stats = agent.get_stats()
        phase_text = font_phase.render(stats["phase_name"], True, (255, 255, 255))
        # Semi-transparent background
        overlay = pygame.Surface((phase_text.get_width() + 20,
                                  phase_text.get_height() + 60), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 120))
        screen.blit(overlay, (10, WINDOW_HEIGHT - phase_text.get_height() - 80))
        screen.blit(phase_text, (20, WINDOW_HEIGHT - phase_text.get_height() - 70))

        # Phase description lines
        for i, line in enumerate(stats["phase_desc"].split("\n")):
            desc_txt = font_phase_desc.render(line, True, (220, 220, 220))
            screen.blit(desc_txt, (25, WINDOW_HEIGHT - 65 + i * 20))

        # Progress bar (thin line at bottom of sim)
        if agent.epsilon > agent.epsilon_min:
            progress = 1.0 - (agent.epsilon - agent.epsilon_min) / (EPSILON_START - agent.epsilon_min)
            bar_y = WINDOW_HEIGHT - 8
            pygame.draw.line(screen, (60, 60, 60), (0, bar_y), (SIM_WIDTH, bar_y), 4)
            bar_end = int(SIM_WIDTH * progress)
            # Gradient from red to green
            r = int(220 * (1 - progress))
            g = int(220 * progress)
            pygame.draw.line(screen, (r, g, 60), (0, bar_y), (bar_end, bar_y), 4)
            # Dot at end
            pygame.draw.circle(screen, (r, g, 60), (bar_end, bar_y), 6)
        
        # Message overlay
        if message_timer > 0:
            message_timer -= 1
            msg_txt = font_msg.render(message, True, (255, 255, 100))
            msg_bg = pygame.Surface((msg_txt.get_width() + 20, msg_txt.get_height() + 10), pygame.SRCALPHA)
            msg_bg.fill((0, 0, 0, 180))
            screen.blit(msg_bg, (CENTER_X - msg_bg.get_width() // 2, 20))
            screen.blit(msg_txt, (CENTER_X - msg_txt.get_width() // 2, 25))

        # Dashboard
        dashboard.draw(env, agent)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()