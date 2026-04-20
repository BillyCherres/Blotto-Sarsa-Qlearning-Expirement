# ============================================================
# Colonel Blotto — Tabular Q-Learning with OpenSpiel
# ============================================================
# Colonel Blotto is a classic game-theory / combinatorial problem:
#   - Two players each distribute N "coins" across K "fields" (battlefields).
#   - Both players reveal their allocations simultaneously (no hidden info).
#   - Each field is won by whoever put MORE coins on it.
#   - The player who wins the most fields wins the game.
#
# This script trains a tabular Q-learning agent (player 0) against
# a random opponent (player 1) over many episodes, then inspects
# the learned Q-values to see what Blotto strategy emerged.
# ============================================================

# pyspiel: Google DeepMind's OpenSpiel C++ game library exposed via Python
# bindings. It contains dozens of board/card/strategy games, including Blotto.
import pyspiel

# numpy: used here mainly to fix the random seed so runs are reproducible.
import numpy as np

# rl_environment wraps a pyspiel game in a Gym-style interface —
# exposes reset() / step(actions) and returns TimeStep namedtuples
# (observations, rewards, discounts, step_type) instead of raw C++ objects.
from open_spiel.python import rl_environment

# RandomAgent: a trivial baseline that picks uniformly at random from
# all legal actions every step — used as the training opponent.
from open_spiel.python.algorithms import random_agent

# QLearner: tabular Q-learning. Stores Q(state, action) estimates in a
# plain Python dict keyed by the string representation of the observation.
from open_spiel.python.algorithms import tabular_qlearner
from sarsa import SARSAAgent


# rl_tools: utility schedulers for RL hyperparameters such as epsilon
# (e.g. ConstantSchedule keeps ε fixed; LinearSchedule decays it over time).
from open_spiel.python import rl_tools



# ── Hyperparameters ──────────────────────────────────────────

# Total number of full games (episodes) to play during training.
# 1 million is large, but Blotto is one-shot (a single joint decision per
# episode), so we need many samples to get reliable Q-value estimates.
MAX_EPISODES = 100000

# Standard 2-player Blotto.
NUM_PLAYERS = 2

# Number of battlefields to contest.
# With 3 fields and 10 coins there are C(12,2) = 66 possible allocations.
NUM_FIELDS = 3

# Total coins each player must fully distribute across all fields.
# More coins → finer-grained allocations → larger action space.
NUM_COINS = 10

# Fix the random seed so that training results are reproducible across runs.
# This governs the random opponent's choices and any numpy-based tie-breaking
# inside the Q-learner.
np.random.seed(1)

# ── Environment ───────────────────────────────────────────────

# Build the keyword-argument dict that OpenSpiel's Blotto game expects.
settings = {'players' : NUM_PLAYERS, 'fields' : NUM_FIELDS, 'coins' : NUM_COINS}

# Instantiate the Gym-style wrapper around the Blotto game.
# Under the hood this calls pyspiel.load_game('blotto', settings) and
# wires up observation encoding, legal-action masks, and reward extraction.
environment = rl_environment.Environment('blotto', **settings)

# action_spec() returns metadata about the action space.
# 'num_actions' is the total number of distinct legal allocations —
# for 10 coins / 3 fields this equals 66.
num_actions = environment.action_spec()['num_actions']
print("Possible Actions:", num_actions)

# ── Agents ────────────────────────────────────────────────────

# QLearner implements the classic off-policy tabular Q-learning update:
#   Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') − Q(s,a)]
#
# Because Blotto is a *one-shot simultaneous* game (both players choose once
# and the episode immediately terminates), s' is always a terminal state,
# so the γ·max_a' Q(s',a') term is 0. The update simplifies to:
#   Q(s,a) ← Q(s,a) + α * [r − Q(s,a)]
# meaning each Q-value converges toward the expected reward for that action
# against the opponent's distribution of play.
#
# epsilon_schedule=ConstantSchedule(0.2) → ε-greedy with fixed ε=0.2:
#   80 % of the time: exploit — pick the highest-Q action.
#   20 % of the time: explore — pick a uniformly random action.
# Keeping ε constant (no decay) maintains exploration throughout all 1M
# episodes, which is useful when facing a fixed random opponent.
rl_agent = tabular_qlearner.QLearner(player_id=0, num_actions=num_actions, epsilon_schedule=rl_tools.ConstantSchedule(0.2))
#rl_agent = SARSAAgent(player_id=0, num_actions=num_actions, epsilon_schedule=rl_tools.ConstantSchedule(0.2))


# RandomAgent: picks uniformly at random from all legal actions every step.
# It never updates any internal model — it is purely a stochastic baseline.
# Training against it teaches the RL agent to exploit the *average* blotto
# distribution, which can reveal strong dominant strategies.
opponent = random_agent.RandomAgent(player_id=1, num_actions=num_actions)

# ── Tracking Variables ────────────────────────────────────────

# Cumulative sum of positive rewards for each player.
# In Blotto: reward = +1 (win), -1 (loss), 0 (draw).
# So won_games[i] equals the total number of wins for player i.
won_games = [0,0]

# Snapshot of won_games after every episode — useful for plotting a
# win-rate learning curve to visualise how quickly the agent improves.
rl_won = []
opp_won = []

# Stores the last StepOutput (action + probability vector) from the RL agent.
# Inspected after training to see what action / distribution it settled on.
last_probs = None

# ── Training Loop ─────────────────────────────────────────────

episode = 0
while episode < MAX_EPISODES:
    episode += 1
    print("EPISODE", episode)

    # Start a fresh game. Returns the first TimeStep with:
    #   .step_type    = StepType.FIRST
    #   .rewards      = None  (no reward yet)
    #   .observations = {'info_state': [...], 'legal_actions': [...]}
    time_step = environment.reset() #initial step per episode

    # Inner game loop. In Blotto this runs exactly ONCE per episode
    # because the game ends after a single simultaneous joint action.
    # time_step.last() is True when step_type == StepType.LAST.
    while not time_step.last():
        # agent.step(time_step) does two things:
        #   1. Selects an action according to the agent's policy.
        #      (ε-greedy for QLearner; uniform random for RandomAgent)
        #   2. Performs a Q-learning update IF this is not the very first
        #      call in the episode (no reward is available on the first call).
        # Returns a StepOutput namedtuple: .action (int index) and .probs (array).
        rl_step = rl_agent.step(time_step)
        opp_step = opponent.step(time_step)

        # Extract the integer action indices chosen by each agent.
        rl_action = rl_step.action   # Player 0's chosen allocation (index into action space)
        opp_action = opp_step.action # Player 1's chosen allocation (index into action space)

        # Cache the RL agent's last StepOutput so we can inspect it post-training.
        last_probs = rl_step

        # action_to_string(player_id, action_index) decodes the integer action
        # back to a human-readable string, e.g. "0=3;1=3;2=4" meaning:
        #   3 coins on field 0, 3 coins on field 1, 4 coins on field 2.
        print('RL', rl_action, ':', environment.get_state.action_to_string(0, rl_action))
        print('Opp', opp_action, ':', environment.get_state.action_to_string(1, opp_action))

        # Pack both players' actions into a list indexed by player_id and
        # advance the game. The returned TimeStep will have step_type=LAST
        # and non-None rewards since Blotto terminates after one move.
        actions = [rl_action, opp_action]
        time_step = environment.step(actions)

    #Episode over, step both agents with final state
    # Calling agent.step() on the TERMINAL TimeStep triggers the crucial update:
    #   Q(s, a_taken) ← Q(s, a_taken) + α * [r_terminal − Q(s, a_taken)]
    # This is where the RL agent actually learns — it now has the reward signal.
    rl_agent.step(time_step)
    # RandomAgent.step() on a terminal state is effectively a no-op (it doesn't
    # learn), but called for API symmetry so both agents observe the outcome.
    opponent.step(time_step)

    #print(environment.get_state)
    # get_state.returns() gives cumulative rewards from episode start.
    # For a one-shot game this equals the single terminal reward.
    rewards = environment.get_state.returns()
    print ("Rewards:", rewards)

    # Only count positive rewards as wins. Negative rewards (losses) are
    # ignored here — we're tracking wins, not losses, separately.
    won_games[0] += rewards[0] if rewards[0] > 0 else 0
    won_games[1] += rewards[1] if rewards[1] > 0 else 0

    # Record a snapshot for later win-rate analysis / plotting.
    rl_won.append(won_games[0])
    opp_won.append(won_games[1])

    print()

# ── Post-Training Analysis ────────────────────────────────────

print("\nWON Games")
print("RL Agent:", int(won_games[0]))
print('Opponent:', int(won_games[1]))
# Print the last StepOutput — shows the action taken on episode 1M and
# the probability distribution the ε-greedy policy put over all actions.


# rl_agent._q_values is a dict-of-dicts:
#   _q_values[obs_string][action_index] → float Q-value
#
# Because Blotto is one-shot, every episode starts with the SAME initial
# observation for player 0. OpenSpiel encodes this as the string '[0.0]'
# (the player's remaining-coins signal before any allocation — a constant).
# So there is only ONE key in _q_values that matters, and printing it shows
# the entire learned Q-table.
#print(len(rl_agent._q_values))


# Print the ranked Q-value table: highest Q-value first.
# High Q-value → this Blotto allocation won frequently against the random opponent.
# Low  Q-value → this allocation tended to lose.
#
# 'done' is a deduplication set: multiple actions can share the same Q-value
# (common early in training). We print each unique value exactly once, then
# list every action tied at that value.



#print()
#print('Won Trend')
#print('RL Agent:', rl_won)   # Full per-episode cumulative win history for the RL agent
#print('Opponent:', opp_won)  # Full per-episode cumulative win history for the random opponent
