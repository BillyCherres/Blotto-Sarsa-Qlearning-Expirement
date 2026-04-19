# ============================================================
# Colonel Blotto — SARSA vs Random Agent
# ============================================================
# Trains a SARSA agent (player 0) against a random opponent
# (player 1). Mirrors the structure of qlearning_vs_random.py —
# swap in SARSAAgent to isolate on-policy vs off-policy behavior
# against the same random baseline.
# ============================================================

import numpy as np
# =======================================================
# used this library to make graphs
import matplotlib.pyplot as plt
# =======================================================

from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent
from open_spiel.python import rl_tools
from sarsa import SARSAAgent


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

# SARSAAgent: on-policy TD control.
#   Q(s,a) ← Q(s,a) + α * [r + γ * Q(s', a') − Q(s,a)]
# In one-shot Blotto the bootstrap term is 0, so this reduces to:
#   Q(s,a) ← Q(s,a) + α * [r − Q(s,a)]
rl_agent = SARSAAgent(
    player_id=0,
    num_actions=num_actions,
    epsilon_schedule=rl_tools.ConstantSchedule(0.2)
)

# RandomAgent: uniform random baseline, never updates any model.
opponent = random_agent.RandomAgent(player_id=1, num_actions=num_actions)

# ── Tracking Variables ────────────────────────────────────────

won_games = [0, 0]

rl_won = []
opp_won = []

# ========================================================
# These are where the points are going to be stored
# (values for x and y of the graph)
x = []
y = []
episodeArr = []
# ========================================================

last_probs = None

# ── Training Loop ─────────────────────────────────────────────

episode = 0
while episode < MAX_EPISODES:
    episode += 1
    print("EPISODE", episode)

    time_step = environment.reset()

    while not time_step.last():
        rl_step = rl_agent.step(time_step)
        opp_step = opponent.step(time_step)

        rl_action = rl_step.action
        opp_action = opp_step.action

        last_probs = rl_step

        print('RL ', rl_action, ':', environment.get_state.action_to_string(0, rl_action))
        print('Opp', opp_action, ':', environment.get_state.action_to_string(1, opp_action))

        actions = [rl_action, opp_action]
        time_step = environment.step(actions)

    rl_agent.step(time_step)
    opponent.step(time_step)

    rewards = environment.get_state.returns()
    print("Rewards:", rewards)

    won_games[0] += rewards[0] if rewards[0] > 0 else 0
    won_games[1] += rewards[1] if rewards[1] > 0 else 0

    rl_won.append(won_games[0])
    opp_won.append(won_games[1])

# =========================================================
    # Add points to array
    x.append(won_games[0])
    y.append(won_games[1])
    episodeArr.append(episode)
    

    print()
# plot makes one graph (so two graphs here)
plt.plot(episodeArr, x)
plt.plot(episodeArr, y)
plt.plot(x, y)
plt.show()
# =========================================================

# ── Post-Training Analysis ────────────────────────────────────

print("\nWON Games")
print("SARSA Agent:", int(won_games[0]))
print("Opponent (Random):", int(won_games[1]))
print(last_probs)

print(rl_agent._q_values['[0.0]'])

q_list = [rl_agent._q_values['[0.0]'][act] for act in rl_agent._q_values['[0.0]'].keys()]

done = set()
for val in sorted(q_list, reverse=True):
    if val in done:
        continue
    done.add(val)
    for act in rl_agent._q_values['[0.0]'].keys():
        if val == rl_agent._q_values['[0.0]'][act]:
            act_string = environment.get_state.action_to_string(0, act)
            print(val, act, act_string)
