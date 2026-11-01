"""
SUMO Traffic Signal Control with Reinforcement Learning
Main application — launches the simulation with SUMO-GUI visualization.

Usage:
  python main_sumo.py              # Train a new agent
  python main_sumo.py --load       # Load and run trained agent
  python main_sumo.py --eval       # Evaluate trained agent (no exploration)
  python main_sumo.py --no-gui     # Train without GUI (faster)

Controls (in SUMO-GUI):
  - Use SUMO's built-in controls to pause/resume
  - Press 'S' to save agent during training
"""

import os
import sys
import time
import argparse
import numpy as np

from config_sumo import *
from sumo_env import SumoEnvironment, build_network
from agent import QLearningAgent


def train(env, agent, episodes=50, save_path="trained_agent.pkl"):
    """Train the agent for multiple episodes."""
    print("\n" + "="*60)
    print("  TRAFFIC SIGNAL RL TRAINING")
    print("="*60)
    
    best_reward = float('-inf')
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        decision_step = 0
        
        print(f"\nEpisode {episode + 1}/{episodes}")
        print(f"  Epsilon: {agent.epsilon:.4f}")
        
        # Run for simulation time
        max_steps = int(STEPS_PER_EPISODE / SUMO_STEP_LENGTH)  # Convert seconds to steps
        decision_interval = int(DECISION_INTERVAL / SUMO_STEP_LENGTH)
        
        prev_state = state
        prev_action = 0
        
        while steps < max_steps:
            try:
                # Make decision at intervals
                if steps % decision_interval == 0:
                    action = agent.choose_action(state)
                    reward = env.step(action)
                    next_state = env.get_state()
                    
                    # Learn from transition
                    agent.learn(prev_state, prev_action, reward, next_state)
                    
                    prev_state = state
                    prev_action = action
                    state = next_state
                    total_reward += reward
                    decision_step += 1
                else:
                    # Just simulate without decision
                    env.step()
                
                steps += 1
                
                # Progress update every 10 simulated seconds
                if steps % int(10 / SUMO_STEP_LENGTH) == 0:
                    sim_time = env.get_simulation_time()
                    print(f"\r  Time: {sim_time:.0f}s | Reward: {total_reward:.1f} | "
                          f"States: {len(agent.q_table)} | ε: {agent.epsilon:.3f}", end="")
                    
            except Exception as e:
                print(f"\n  Simulation ended: {e}")
                break
        
        print(f"\n  Episode complete! Total reward: {total_reward:.1f}")
        
        # Save if best
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(save_path)
            print(f"  New best! Agent saved.")
    
    env.close()
    print("\n" + "="*60)
    print(f"  Training complete! Best reward: {best_reward:.1f}")
    print(f"  Agent saved to: {save_path}")
    print("="*60)


def evaluate(env, agent, duration=600):
    """Evaluate a trained agent."""
    print("\n" + "="*60)
    print("  EVALUATING TRAINED AGENT")
    print("="*60)
    
    agent.set_evaluation_mode()
    state = env.reset()
    
    total_reward = 0
    steps = 0
    max_steps = int(duration / SUMO_STEP_LENGTH)
    decision_interval = int(DECISION_INTERVAL / SUMO_STEP_LENGTH)
    
    while steps < max_steps:
        try:
            if steps % decision_interval == 0:
                action = agent.choose_action(state)
                reward = env.step(action)
                state = env.get_state()
                total_reward += reward
            else:
                env.step()
            
            steps += 1
            
            if steps % int(10 / SUMO_STEP_LENGTH) == 0:
                sim_time = env.get_simulation_time()
                queues = env.get_queue_lengths()
                total_queue = sum(queues.values())
                print(f"\r  Time: {sim_time:.0f}s | Reward: {total_reward:.1f} | "
                      f"Queue: {total_queue}", end="")
                
        except Exception as e:
            print(f"\n  Simulation ended: {e}")
            break
    
    env.close()
    print(f"\n\n  Evaluation complete! Total reward: {total_reward:.1f}")


def demo_trained_agent(agent_path="trained_agent.pkl"):
    """Demo the trained agent with SUMO-GUI."""
    print("\n" + "="*60)
    print("  DEMO: TRAINED TRAFFIC SIGNAL AGENT")
    print("="*60)
    
    # Load agent
    agent = QLearningAgent()
    if not agent.load(agent_path):
        print("No trained agent found. Please train first.")
        return
    
    agent.set_evaluation_mode()
    
    # Start SUMO with GUI
    env = SumoEnvironment(gui=True)
    env.start()
    
    state = env.get_state()
    total_reward = 0
    steps = 0
    decision_interval = int(DECISION_INTERVAL / SUMO_STEP_LENGTH)
    
    print("\nRunning trained agent... (Press Ctrl+C to stop)")
    print("Watch how the agent adapts signal timing to traffic conditions!\n")
    
    try:
        while True:
            if steps % decision_interval == 0:
                action = agent.choose_action(state)
                reward = env.step(action)
                state = env.get_state()
                total_reward += reward
                
                # Print status
                queues = env.get_queue_lengths()
                signals = env.get_signal_state()
                phase = "NS Green" if action == 0 else "EW Green"
                print(f"  [{env.get_simulation_time():6.1f}s] {phase:10} | "
                      f"N:{queues['North']:2} S:{queues['South']:2} "
                      f"E:{queues['East']:2} W:{queues['West']:2} | "
                      f"Reward: {reward:+.1f}")
            else:
                env.step()
            
            steps += 1
            
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user.")
    except Exception as e:
        print(f"\nSimulation ended: {e}")
    finally:
        env.close()
    
    print(f"\nTotal reward: {total_reward:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Traffic Signal RL with SUMO")
    parser.add_argument("--load", action="store_true", help="Load and run trained agent")
    parser.add_argument("--eval", action="store_true", help="Evaluate trained agent")
    parser.add_argument("--no-gui", action="store_true", help="Run without GUI")
    parser.add_argument("--episodes", type=int, default=50, help="Training episodes")
    parser.add_argument("--model", type=str, default="trained_agent.pkl", help="Model path")
    parser.add_argument("--build", action="store_true", help="Build SUMO network")
    args = parser.parse_args()
    
    # Check SUMO_HOME
    if 'SUMO_HOME' not in os.environ:
        print("ERROR: Please set SUMO_HOME environment variable")
        print("  Example: set SUMO_HOME=C:\\Program Files (x86)\\Eclipse\\Sumo")
        sys.exit(1)
    
    # Build network if requested or if it doesn't exist
    net_file = os.path.join(os.path.dirname(__file__), 'sumo_files', 'intersection.net.xml')
    if args.build or not os.path.exists(net_file):
        print("Building SUMO network...")
        if not build_network():
            print("Failed to build network. Check SUMO installation.")
            sys.exit(1)
    
    if args.load or args.eval:
        # Demo/evaluate trained agent
        demo_trained_agent(args.model)
    else:
        # Train new agent
        use_gui = not args.no_gui
        env = SumoEnvironment(gui=use_gui)
        agent = QLearningAgent()
        
        # Check if we should resume training
        if os.path.exists(args.model):
            response = input(f"Found existing model '{args.model}'. Resume training? (y/n): ")
            if response.lower() == 'y':
                agent.load(args.model)
        
        env.start()
        train(env, agent, episodes=args.episodes, save_path=args.model)


if __name__ == "__main__":
    main()
