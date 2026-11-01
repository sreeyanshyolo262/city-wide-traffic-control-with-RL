"""
Comparison Demo: RL Agent vs Fixed-Time Traffic Light
Visual side-by-side comparison with results at the end.

Usage:
  python compare_demo.py           # Run visual comparison
  python compare_demo.py --fast    # Quick comparison (no GUI)
"""

import os
import sys
import time
import argparse

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please set SUMO_HOME environment variable")

import traci
from config_sumo import *
from sumo_env import SumoEnvironment
from agent import QLearningAgent


def run_fixed_time_visual(duration=180):
    """Run VISUAL simulation with fixed-time signal (30s each phase)."""
    print("\n" + "="*60)
    print("  FIXED-TIME CONTROLLER (30s cycles)")
    print("  Watch how it ignores traffic conditions!")
    print("="*60)
    print("\nOpening SUMO-GUI... (Close window or wait to finish)\n")
    
    env = SumoEnvironment(gui=True)
    env.start()
    
    total_wait_time = 0
    total_passed = 0
    total_queue = 0
    samples = 0
    
    steps = 0
    max_steps = int(duration / SUMO_STEP_LENGTH)
    phase_duration = int(30 / SUMO_STEP_LENGTH)  # 30 second fixed phases
    
    current_action = 0  # Start with NS green
    phase_timer = 0
    last_print = 0
    
    try:
        while steps < max_steps:
            # Fixed-time logic: switch every 30 seconds
            phase_timer += 1
            if phase_timer >= phase_duration:
                phase_timer = 0
                current_action = 1 - current_action  # Toggle 0 <-> 1
            
            env.step(current_action)
            steps += 1
            
            # Count vehicles passed (accumulate each step)
            try:
                total_passed += traci.simulation.getArrivedNumber()
            except:
                pass
            
            # Collect metrics every second
            if steps % int(1 / SUMO_STEP_LENGTH) == 0:
                queues = env.get_queue_lengths()
                total_queue += sum(queues.values())
                samples += 1
                
                # Get waiting time
                for edge in env.incoming_edges:
                    try:
                        total_wait_time += traci.edge.getWaitingTime(edge)
                    except:
                        pass
            
            # Print status every 5 seconds
            sim_time = env.get_simulation_time()
            if sim_time - last_print >= 5:
                last_print = sim_time
                queues = env.get_queue_lengths()
                phase = "NS Green" if current_action == 0 else "EW Green"
                print(f"  [{sim_time:6.1f}s] {phase:10} (FIXED) | "
                      f"N:{queues['North']:2} S:{queues['South']:2} "
                      f"E:{queues['East']:2} W:{queues['West']:2} | "
                      f"Passed: {total_passed}")
                    
    except Exception as e:
        print(f"Simulation ended: {e}")
    
    avg_queue = total_queue / samples if samples > 0 else 0
    avg_wait = total_wait_time / samples if samples > 0 else 0
    env.close()
    
    print(f"\n  Fixed-Time Complete! Vehicles passed: {total_passed}")
    
    return {
        "avg_queue": avg_queue,
        "avg_wait_time": avg_wait,
        "vehicles_passed": total_passed,
        "type": "Fixed-Time (30s)"
    }


def run_rl_agent_visual(duration=180):
    """Run VISUAL simulation with trained RL agent."""
    print("\n" + "="*60)
    print("  RL AGENT (Trained Q-Learning)")
    print("  Watch how it adapts to traffic!")
    print("="*60)
    
    agent = QLearningAgent()
    if not agent.load("trained_agent.pkl"):
        print("No trained agent found!")
        return None
    
    agent.set_evaluation_mode()
    print("\nOpening SUMO-GUI... (Close window or wait to finish)\n")
    
    env = SumoEnvironment(gui=True)
    env.start()
    
    state = env.get_state()
    total_wait_time = 0
    total_queue = 0
    total_passed = 0
    samples = 0
    
    steps = 0
    max_steps = int(duration / SUMO_STEP_LENGTH)
    decision_interval = int(DECISION_INTERVAL / SUMO_STEP_LENGTH)
    last_print = 0
    last_action = 0
    
    try:
        while steps < max_steps:
            if steps % decision_interval == 0:
                action = agent.choose_action(state)
                last_action = action
                env.step(action)
                state = env.get_state()
            else:
                env.step()
            
            steps += 1
            
            # Count vehicles passed (accumulate each step)
            try:
                total_passed += traci.simulation.getArrivedNumber()
            except:
                pass
            
            # Collect metrics every second
            if steps % int(1 / SUMO_STEP_LENGTH) == 0:
                queues = env.get_queue_lengths()
                total_queue += sum(queues.values())
                samples += 1
                
                # Get waiting time
                for edge in env.incoming_edges:
                    try:
                        total_wait_time += traci.edge.getWaitingTime(edge)
                    except:
                        pass
            
            # Print status every 5 seconds
            sim_time = env.get_simulation_time()
            if sim_time - last_print >= 5:
                last_print = sim_time
                queues = env.get_queue_lengths()
                phase = "NS Green" if last_action == 0 else "EW Green"
                print(f"  [{sim_time:6.1f}s] {phase:10} (RL)    | "
                      f"N:{queues['North']:2} S:{queues['South']:2} "
                      f"E:{queues['East']:2} W:{queues['West']:2} | "
                      f"Passed: {total_passed}")
                    
    except Exception as e:
        print(f"Simulation ended: {e}")
    
    avg_queue = total_queue / samples if samples > 0 else 0
    avg_wait = total_wait_time / samples if samples > 0 else 0
    env.close()
    
    print(f"\n  RL Agent Complete! Vehicles passed: {total_passed}")
    
    return {
        "avg_queue": avg_queue,
        "avg_wait_time": avg_wait,
        "vehicles_passed": total_passed,
        "type": "RL Agent"
    }


def run_fast_comparison(duration=300):
    """Run fast comparison without GUI."""
    print(f"\nRunning {duration}s simulation for each controller (no GUI)...")
    
    # Fixed-time
    print("\nTesting Fixed-Time Controller...", end=" ", flush=True)
    env = SumoEnvironment(gui=False)
    env.start()
    
    total_wait_time = 0
    total_queue = 0
    fixed_passed = 0
    samples = 0
    steps = 0
    max_steps = int(duration / SUMO_STEP_LENGTH)
    phase_duration = int(30 / SUMO_STEP_LENGTH)
    current_action = 0
    phase_timer = 0
    
    while steps < max_steps:
        try:
            phase_timer += 1
            if phase_timer >= phase_duration:
                phase_timer = 0
                current_action = 1 - current_action
            
            env.step(current_action)
            steps += 1
            
            # Accumulate passed vehicles each step
            try:
                fixed_passed += traci.simulation.getArrivedNumber()
            except:
                pass
            
            if steps % int(1 / SUMO_STEP_LENGTH) == 0:
                queues = env.get_queue_lengths()
                total_queue += sum(queues.values())
                for edge in env.incoming_edges:
                    try:
                        total_wait_time += traci.edge.getWaitingTime(edge)
                    except:
                        pass
                samples += 1
        except:
            break
    
    fixed_queue = total_queue / samples if samples > 0 else 0
    fixed_wait = total_wait_time / samples if samples > 0 else 0
    env.close()
    print("Done")
    
    # RL Agent
    print("Testing RL Agent...", end=" ", flush=True)
    agent = QLearningAgent()
    agent.load("trained_agent.pkl")
    agent.set_evaluation_mode()
    
    env = SumoEnvironment(gui=False)
    env.start()
    
    state = env.get_state()
    total_wait_time = 0
    total_queue = 0
    rl_passed = 0
    samples = 0
    steps = 0
    decision_interval = int(DECISION_INTERVAL / SUMO_STEP_LENGTH)
    
    while steps < max_steps:
        try:
            if steps % decision_interval == 0:
                action = agent.choose_action(state)
                env.step(action)
                state = env.get_state()
            else:
                env.step()
            
            steps += 1
            
            # Accumulate passed vehicles each step
            try:
                rl_passed += traci.simulation.getArrivedNumber()
            except:
                pass
            
            if steps % int(1 / SUMO_STEP_LENGTH) == 0:
                queues = env.get_queue_lengths()
                total_queue += sum(queues.values())
                for edge in env.incoming_edges:
                    try:
                        total_wait_time += traci.edge.getWaitingTime(edge)
                    except:
                        pass
                samples += 1
        except:
            break
    
    rl_queue = total_queue / samples if samples > 0 else 0
    rl_wait = total_wait_time / samples if samples > 0 else 0
    env.close()
    print("Done")
    
    return (
        {"avg_queue": fixed_queue, "avg_wait_time": fixed_wait, "vehicles_passed": fixed_passed},
        {"avg_queue": rl_queue, "avg_wait_time": rl_wait, "vehicles_passed": rl_passed}
    )


def print_comparison(fixed_results, rl_results):
    """Print comparison table."""
    print("\n" + "="*70)
    print("  RESULTS COMPARISON: Fixed-Time vs RL Agent")
    print("="*70)
    print(f"\n{'Metric':<25} {'Fixed-Time':>15} {'RL Agent':>15} {'Improvement':>12}")
    print("-"*70)
    
    if fixed_results and rl_results:
        # Average Queue
        fixed_q = fixed_results['avg_queue']
        rl_q = rl_results['avg_queue']
        improvement_q = ((fixed_q - rl_q) / fixed_q * 100) if fixed_q > 0 else 0
        print(f"{'Avg Queue Length':<25} {fixed_q:>15.1f} {rl_q:>15.1f} {improvement_q:>+11.1f}%")
        
        # Average Wait Time
        fixed_w = fixed_results['avg_wait_time']
        rl_w = rl_results['avg_wait_time']
        improvement_w = ((fixed_w - rl_w) / fixed_w * 100) if fixed_w > 0 else 0
        print(f"{'Avg Wait Time (s)':<25} {fixed_w:>15.1f} {rl_w:>15.1f} {improvement_w:>+11.1f}%")
        
        # Vehicles Passed
        fixed_p = fixed_results['vehicles_passed']
        rl_p = rl_results['vehicles_passed']
        improvement_p = ((rl_p - fixed_p) / fixed_p * 100) if fixed_p > 0 else 0
        print(f"{'Vehicles Passed':<25} {fixed_p:>15} {rl_p:>15} {improvement_p:>+11.1f}%")
    
    print("\n" + "="*70)
    
    print("\n  SUMMARY:")
    if improvement_q > 0:
        print(f"  - RL Agent reduced queue length by {improvement_q:.1f}%")
    if improvement_w > 0:
        print(f"  - RL Agent reduced wait time by {improvement_w:.1f}%")
    if improvement_p > 0:
        print(f"  - RL Agent passed {improvement_p:.1f}% more vehicles")
    
    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Compare RL Agent vs Fixed-Time Controller")
    parser.add_argument("--fast", action="store_true", help="Fast comparison without GUI")
    parser.add_argument("--duration", type=int, default=180, help="Duration in seconds (default: 180)")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  TRAFFIC SIGNAL COMPARISON DEMO")
    print("  RL Agent vs Fixed-Time Controller")
    print("="*70)
    
    if args.fast:
        fixed_results, rl_results = run_fast_comparison(args.duration)
    else:
        duration = args.duration
        print(f"\nThis demo will run TWO {duration}s simulations back-to-back:")
        print("  1. Fixed-Time Controller (30s cycles) - watch the inefficiency")
        print("  2. RL Agent - watch how it adapts to traffic")
        print("\nResults will be compared at the end.\n")
        input("Press Enter to start the Fixed-Time simulation...")
        
        # Run fixed-time with GUI
        fixed_results = run_fixed_time_visual(duration)
        
        print("\n" + "-"*60)
        input("\nPress Enter to start the RL Agent simulation...")
        
        # Run RL agent with GUI
        rl_results = run_rl_agent_visual(duration)
    
    # Show comparison
    print_comparison(fixed_results, rl_results)
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main()
