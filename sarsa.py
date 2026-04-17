import collections
import numpy as np

from open_spiel.python import rl_tools
from open_spiel.python.rl_agent import StepOutput


class SARSAAgent:

    def __init__(
        self,
        player_id,
        num_actions,
        step_size=0.5,
        epsilon_schedule=rl_tools.ConstantSchedule(0.2),
        discount_factor=1.0,
    ):
        self._player_id = player_id
        self._num_actions = num_actions
        self._alpha = step_size
        self._epsilon = epsilon_schedule
        self._gamma = discount_factor

        # Same structure as QLearner._q_values so post-training analysis
        # code can be reused without changes.
        self._q_values = collections.defaultdict(
            lambda: collections.defaultdict(float)
        )

        # Carry previous (time_step, action) across calls so we can do
        # the delayed SARSA update once we know the next action.
        self._prev_time_step = None
        self._prev_action = None

    # ------------------------------------------------------------------ #

    def step(self, time_step):
        # Terminal: do the final update with r and no bootstrap, then reset.
        if time_step.last():
            if self._prev_time_step is not None:
                self._update(time_step, next_action=None)
            self._prev_time_step = None
            self._prev_action = None
            return

        obs = str(time_step.observations["info_state"][self._player_id])
        legal = time_step.observations["legal_actions"][self._player_id]

        # Pick a' via epsilon-greedy.
        if np.random.random() < self._epsilon.value:
            action = np.random.choice(legal)
        else:
            action = self._greedy(obs, legal)

        # Now that we know a', we can update Q(s, a) using Q(s', a').
        # On the very first call there's no previous step yet, so skip.
        if self._prev_time_step is not None:
            self._update(time_step, next_action=action)

        self._prev_time_step = time_step
        self._prev_action = action

        return StepOutput(action=action, probs=self._probs(obs, legal))

    # ------------------------------------------------------------------ #

    def _update(self, next_time_step, next_action):
        """
        Q(s, a) += alpha * [r + gamma * Q(s', a') - Q(s, a)]

        next_action=None signals a terminal transition, so the
        bootstrap term drops out and the target is just r.
        """
        obs = str(self._prev_time_step.observations["info_state"][self._player_id])
        a = self._prev_action
        r = next_time_step.rewards[self._player_id]

        if next_action is None:
            target = r
        else:
            next_obs = str(next_time_step.observations["info_state"][self._player_id])
            target = r + self._gamma * self._q_values[next_obs][next_action]

        self._q_values[obs][a] += self._alpha * (target - self._q_values[obs][a])

    def _greedy(self, obs, legal_actions):
        # Ties broken randomly so early training doesn't get stuck on
        # whichever action happens to be visited first.
        q = self._q_values[obs]
        best_val = max(q[a] for a in legal_actions)
        best = [a for a in legal_actions if q[a] == best_val]
        return np.random.choice(best)

    def _probs(self, obs, legal_actions):
        # Epsilon-greedy probability for each action — mostly for logging.
        probs = np.zeros(self._num_actions)
        eps = self._epsilon.value
        greedy = self._greedy(obs, legal_actions)
        n = len(legal_actions)
        for a in legal_actions:
            probs[a] = eps / n + (1 - eps if a == greedy else 0)
        return probs
