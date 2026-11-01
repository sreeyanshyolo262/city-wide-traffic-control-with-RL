"""
Resume training for multi-agent DQN on 4x4 SUMO grid.
Loads pretrained B2/B3/C2/C3 weights if available, then continues training.
"""

import argparse
import numpy as np
from collections import defaultdict
from collections import deque

from multi_agent_env import MultiAgentSumoEnv
from dqn_agent import DQNAgent


def _weights_name(tl_id):
    return f"trained_agent_{tl_id}.weights.h5"


def _save_all_agents(agents):
    for tl_id, agent in agents.items():
        agent.save(_weights_name(tl_id))


def _load_all_agents(agents, strict=False):
    loaded = 0
    for tl_id, agent in agents.items():
        ok = agent.load(_weights_name(tl_id))
        if ok:
            loaded += 1
        elif strict:
            raise RuntimeError(f"Missing required checkpoint for {tl_id}: {_weights_name(tl_id)}")
    print(f"Loaded {loaded}/{len(agents)} agent checkpoints.")


def train_multi_agent_resume(
    episodes=20,
    use_gui=False,
    steps_per_episode=3000,
    replay_every=20,
    decision_interval_steps=20,
    traffic_scale=1.0,
    start_epsilon=0.15,
    strict_load=False,
):
    print("\n" + "=" * 60)
    print("  MULTI-AGENT DQN RESUME TRAINING")
    print("=" * 60)

    env = MultiAgentSumoEnv(gui=use_gui, traffic_scale=traffic_scale)
    env.start()

    agents = {}
    state_sizes = {}
    action_size = 2

    initial_states = env.get_all_states()
    for tl_id, state in initial_states.items():
        state_size = len(state)
        state_sizes[tl_id] = state_size
        agents[tl_id] = DQNAgent(state_size, action_size, agent_id=tl_id)
        print(f"Created agent for {tl_id} with state size {state_size}")

    _load_all_agents(agents, strict=strict_load)

    # Resume exploration from lower epsilon instead of 1.0
    for agent in agents.values():
        agent.epsilon = min(agent.epsilon, start_epsilon)

    batch_size = 32
    warmup_size = batch_size * 4
    done = False

    try:
        recent_global_rewards = deque(maxlen=10)
        for e in range(episodes):
            states = env.reset()
            total_rewards = defaultdict(float)
            interval_rewards = defaultdict(float)
            decision_states = {}
            decision_actions = {}

            for tl_id in agents:
                states[tl_id] = np.reshape(states[tl_id], [1, state_sizes[tl_id]])

            for t in range(steps_per_episode):
                if t % 200 == 0:
                    print(f"\rEpisode {e+1}/{episodes}, time {t}/{steps_per_episode}", end="")

                if t % decision_interval_steps == 0:
                    actions = {tl_id: agent.choose_action(states[tl_id]) for tl_id, agent in agents.items()}
                    decision_states = {tl_id: states[tl_id].copy() for tl_id in agents}
                    decision_actions = actions.copy()
                else:
                    actions = None
                rewards = env.step(actions)
                next_states = env.get_all_states()

                for tl_id, agent in agents.items():
                    next_state = np.reshape(next_states[tl_id], [1, state_sizes[tl_id]])
                    reward = rewards[tl_id]
                    interval_rewards[tl_id] += reward
                    total_rewards[tl_id] += reward

                    if (t + 1) % decision_interval_steps == 0 and tl_id in decision_actions:
                        agent.remember(
                            decision_states[tl_id],
                            decision_actions[tl_id],
                            interval_rewards[tl_id],
                            next_state,
                            done,
                        )
                        interval_rewards[tl_id] = 0.0

                states = {tl_id: np.reshape(s, [1, state_sizes[tl_id]]) for tl_id, s in next_states.items()}

                if ((t + 1) % decision_interval_steps == 0) and (t % replay_every == 0):
                    for agent in agents.values():
                        if len(agent.memory) >= warmup_size:
                            agent.replay(batch_size)

            for tl_id, agent in agents.items():
                if interval_rewards[tl_id] != 0.0 and tl_id in decision_actions:
                    agent.remember(
                        decision_states[tl_id],
                        decision_actions[tl_id],
                        interval_rewards[tl_id],
                        states[tl_id],
                        done,
                    )

            avg_rewards = {tl_id: r / steps_per_episode for tl_id, r in total_rewards.items()}
            global_mean_reward = float(np.mean(list(avg_rewards.values())))
            recent_global_rewards.append(global_mean_reward)
            rolling_mean = float(np.mean(recent_global_rewards))
            print(f"\nEpisode {e+1} complete. Average rewards: {avg_rewards}")
            print(f"Global mean reward: {global_mean_reward:.3f} | Rolling-10 mean: {rolling_mean:.3f}")
            _save_all_agents(agents)
            print("Checkpoint saved for all intersections.")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current models...")
        _save_all_agents(agents)
        print("Partial models saved.")
    finally:
        env.close()
        print("\nTraining complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume Multi-Agent DQN Training for SUMO")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--replay-every", type=int, default=20)
    parser.add_argument("--decision-interval-steps", type=int, default=20)
    parser.add_argument("--traffic-scale", type=float, default=1.0)
    parser.add_argument("--start-epsilon", type=float, default=0.15)
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()

    train_multi_agent_resume(
        episodes=args.episodes,
        use_gui=not args.no_gui,
        steps_per_episode=args.steps,
        replay_every=args.replay_every,
        decision_interval_steps=args.decision_interval_steps,
        traffic_scale=args.traffic_scale,
        start_epsilon=args.start_epsilon,
        strict_load=args.strict_load,
    )