# ============================================================
# Colonel Blotto — SARSA vs Q-Learning Experiment
# ============================================================
# This script runs the same Colonel Blotto setup as blotto_rl.py,
# but pits two learning agents against each other:
#   - Player 0: SARSA (on-policy TD control)
#   - Player 1: Q-Learning (off-policy TD control)
#
# Both agents observe the same game state and update their Q-tables
# each episode. The experiment isolates the effect of on-policy vs
# off-policy learning in a simultaneous, one-shot game setting.
# ============================================================

import pyspiel
import numpy as np

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import tabular_qlearner
from sarsa import SARSAAgent
from open_spiel.python import rl_tools


# ── Hyperparameters ──────────────────────────────────────────

MAX_EPISODES = 10000
NUM_PLAYERS = 2
NUM_FIELDS = 3
NUM_COINS = 10

np.random.seed(1)

# ── Environment ───────────────────────────────────────────────

settings = {'players': NUM_PLAYERS, 'fields': NUM_FIELDS, 'coins': NUM_COINS}

environment = rl_environment.Environment('blotto', **settings)

num_actions = environment.action_spec()['num_actions']
print("Possible Actions:", num_actions)

# ── Agents ────────────────────────────────────────────────────

# Player 0: SARSA — on-policy TD control.
# Updates Q(s,a) using the action actually taken next (a'), not the greedy max.
#   Q(s,a) ← Q(s,a) + α * [r + γ * Q(s', a') − Q(s,a)]
# Because Blotto is one-shot, γ·Q(s',a') = 0 and the update reduces to:
#   Q(s,a) ← Q(s,a) + α * [r − Q(s,a)]
sarsa_agent = SARSAAgent(
    player_id=0,
    num_actions=num_actions,
    epsilon_schedule=rl_tools.ConstantSchedule(0.2)
)

# Player 1: Q-Learning — off-policy TD control.
# Updates Q(s,a) using the greedy max over next actions, regardless of what
# was actually chosen. In a one-shot game this also collapses to:
#   Q(s,a) ← Q(s,a) + α * [r − Q(s,a)]
# The behavioural difference shows up in multi-step games; including both here
# makes this file the canonical SARSA vs Q-Learning comparison.
ql_agent = tabular_qlearner.QLearner(
    player_id=1,
    num_actions=num_actions,
    epsilon_schedule=rl_tools.ConstantSchedule(0.2)
)

# ── Tracking Variables ────────────────────────────────────────

won_games = [0, 0]

sarsa_won = []
ql_won = []

last_sarsa_probs = None
last_ql_probs = None

# ── Training Loop ─────────────────────────────────────────────

episode = 0
while episode < MAX_EPISODES:
    episode += 1
    print("EPISODE", episode)

    time_step = environment.reset()

    while not time_step.last():
        sarsa_step = sarsa_agent.step(time_step)
        ql_step = ql_agent.step(time_step)

        sarsa_action = sarsa_step.action
        ql_action = ql_step.action

        last_sarsa_probs = sarsa_step
        last_ql_probs = ql_step

        print('SARSA', sarsa_action, ':', environment.get_state.action_to_string(0, sarsa_action))
        print('QL   ', ql_action, ':', environment.get_state.action_to_string(1, ql_action))

        actions = [sarsa_action, ql_action]
        time_step = environment.step(actions)

    # Terminal step — both agents learn from the reward signal.
    sarsa_agent.step(time_step)
    ql_agent.step(time_step)

    rewards = environment.get_state.returns()
    print("Rewards:", rewards)

    won_games[0] += rewards[0] if rewards[0] > 0 else 0
    won_games[1] += rewards[1] if rewards[1] > 0 else 0

    sarsa_won.append(won_games[0])
    ql_won.append(won_games[1])

    print()

# ── Post-Training Analysis ────────────────────────────────────

print("\nWON Games")
print("SARSA Agent:", int(won_games[0]))
print("Q-Learning Agent:", int(won_games[1]))

print("\n--- SARSA Q-Values ---")
print(sarsa_agent._q_values['[0.0]'])

print("\n--- Q-Learning Q-Values ---")
print(ql_agent._q_values['[0.0]'])

def print_ranked_q_table(agent, label, environment):
    print(f"\n{label} — Ranked Q-Table (best actions first):")
    q_list = [agent._q_values['[0.0]'][act] for act in agent._q_values['[0.0]'].keys()]
    done = set()
    for val in sorted(q_list, reverse=True):
        if val in done:
            continue
        done.add(val)
        for act in agent._q_values['[0.0]'].keys():
            if val == agent._q_values['[0.0]'][act]:
                act_string = environment.get_state.action_to_string(0, act)
                print(val, act, act_string)

print_ranked_q_table(sarsa_agent, "SARSA", environment)
print_ranked_q_table(ql_agent, "Q-Learning", environment)
