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
from open_spiel.python.algorithms import tabular_qlearner
from open_spiel.python.algorithms import random_agent
from open_spiel.python import rl_tools
from sarsa import SARSAAgent

class universalBlotto :

    def __init__ (
            self,
            player1, # 0 = random, 1 = sarsa , 2 = ql
            player2, 
            training = 0, # 0 = no, 1 = file 1, etc.
            episodes = 100000,
            simulations = 1,
            players = 2,
            fields = 3,
            coins = 10,
    ) :
        np.random.seed(1) 
        self.MAX_EPISODES = episodes
        self.NUM_PLAYERS = players
        self.NUM_FIELDS = fields
        self.NUM_COINS = coins
        self.simCount = simulations
        self.input = training
        self.p1Type = player1
        self.p2Type = player2
        
        
        # ── Environment ───────────────────────────────────────────────

        settings = {'players': self.NUM_PLAYERS, 'fields': self.NUM_FIELDS, 'coins': self.NUM_COINS}

        self.environment = rl_environment.Environment('blotto', **settings)

        self.num_actions = self.environment.action_spec()['num_actions']
        print("Possible Actions:", self.num_actions)

        # ── Agents ────────────────────────────────────────────────────
        self.p1 = self.playerSetUp(self.p1Type)
        self.p2 = self.playerSetUp(self.p2Type)
        self.p1Label = self.playerLables(self.p1Type)
        self.p2Label = self.playerLables(self.p2Type)
        
        
    
    
    



        # ── Tracking Variables ────────────────────────────────────────

        self.won_games = [0, 0]

        self.rl_won = []
        self.opp_won = []

        # ========================================================
        # These are where the points are going to be stored
        # (values for x and y of the graph)
        self.x = []
        self.y = []
        self.episodeArr = []
        self.avgX = []
        self.avgY = []
        # ========================================================

        self.last_probs = None

        # ── Training Loop ─────────────────────────────────────────────

    def playSim(self):

        simulation = 0
        while simulation < self.simCount:
            print("Current Simulation: ", simulation)
            # reset agents and personal scores
            self.p1 = self.playerSetUp(self.p1Type)
            self.p2 = self.playerSetUp(self.p2Type)
            self.won_games[0] = 0
            self.won_games[1] = 0
            simulation += 1
            self.playGame()

        for i in range(len(self.avgX)):
            self.avgX[i] /= self.simCount
            self.avgY[i] /= self.simCount
            
            
        



    def playGame(self):
        episode = 0
        while episode < self.MAX_EPISODES:
            episode += 1
            #print("EPISODE", episode)

            time_step = self.environment.reset()

            while not time_step.last():
                rl_step = self.p1.step(time_step)
                opp_step = self.p2.step(time_step)

                rl_action = rl_step.action
                opp_action = opp_step.action

                self.last_probs = rl_step

                #print('RL ', rl_action, ':', self.environment.get_state.action_to_string(0, rl_action))
                #print('Opp', opp_action, ':', self.environment.get_state.action_to_string(1, opp_action))

                actions = [rl_action, opp_action]
                time_step = self.environment.step(actions)

            self.p1.step(time_step)
            self.p2.step(time_step)

            rewards = self.environment.get_state.returns()
            #print("Rewards:", rewards)

            self.won_games[0] += rewards[0] if rewards[0] > 0 else 0
            self.won_games[1] += rewards[1] if rewards[1] > 0 else 0

            
            self.rl_won.append(self.won_games[0])
            self.opp_won.append(self.won_games[1])

            self.x.append(self.won_games[0])
            self.y.append(self.won_games[1])
            self.episodeArr.append(episode)

            if (len(self.avgX) < self.MAX_EPISODES):
                self.avgX.append(self.won_games[0])
                self.avgY.append(self.won_games[1])

            else: # add to current ones and divide to get average later
                self.avgX[episode - 1] += (self.won_games[0])
                self.avgY[episode - 1] += (self.won_games[1])
            

            #print()

    def plotMultipleGraphs(self):
        for i in range(self.simCount):
            plt.figure()
            # in one graphs
            # plot the points
            plt.plot(self.episodeArr[self.MAX_EPISODES * i: self.MAX_EPISODES * (i + 1)], 
                     self.x[self.MAX_EPISODES * i: self.MAX_EPISODES * (i + 1)], label = self.p1Label)
            plt.plot(self.episodeArr[self.MAX_EPISODES * i: self.MAX_EPISODES * (i + 1)], 
                     self.y[self.MAX_EPISODES * i: self.MAX_EPISODES * (i + 1)], label = self.p2Label)
            #plt.plot(self.x[self.MAX_EPISODES * i: self.MAX_EPISODES * (i + 1)], 
            #         self.y[self.MAX_EPISODES * i: self.MAX_EPISODES * (i + 1)], 
            #         label = str(self.p1Label, "vs", self.p2Label))

            plt.xlabel("Episodes")
            plt.ylabel("Wins")
 
            plt.legend()
            plt.show()
    
    def plotAverageGraph(self):
            plt.figure()
            # in one graphs
            # plot the points
            plt.plot(self.episodeArr[0: self.MAX_EPISODES], 
                     self.avgX, label = self.p1Label)
            plt.plot(self.episodeArr[0: self.MAX_EPISODES], 
                     self.avgY, label = self.p2Label)
            #plt.plot(self.avgX, 
            #         self.avgY, 
            #         label = str(self.p1Label, "vs", self.p2Label))
            plt.xlabel("Episodes")
            plt.ylabel("Wins")
            plt.title("Average Wins per Simulation")

            plt.legend()
            plt.show()
       

    # ── Post-Training Analysis ────────────────────────────────────
    def postTrainingAnalysis(self):
        print("\nWON Games")
        print("SARSA Agent:", int(self.won_games[0]))
        print("Opponent (Random):", int(self.won_games[1]))
        print(self.last_probs)

        print(self.p1._q_values['[0.0]'])

        q_list = [self.p1._q_values['[0.0]'][act] for act in self.p1._q_values['[0.0]'].keys()]

        done = set()
        for val in sorted(q_list, reverse=True):
            if val in done:
                continue
            done.add(val)
            for act in self.p1._q_values['[0.0]'].keys():
                if val == self.p1._q_values['[0.0]'][act]:
                    act_string = self.environment.get_state.action_to_string(0, act)
                    print(val, act, act_string)

# ================================== SETUP ==========================================
    def playerSetUp(self, playerNum):
            if playerNum == 0:
                return random_agent.RandomAgent(player_id=1, num_actions=self.num_actions)
            elif playerNum == 1:
                return SARSAAgent(
                            player_id=0,
                            num_actions=self.num_actions,
                            epsilon_schedule=rl_tools.ConstantSchedule(0.2)
                            )
            elif playerNum == 2:
                return tabular_qlearner.QLearner(
                                player_id=1,
                                num_actions=self.num_actions,
                                epsilon_schedule=rl_tools.ConstantSchedule(0.2)
                            )
            
    def playerLables(self, playerNum):
                if playerNum == 0:
                    return "Random Agent"
                elif playerNum == 1:
                    return "SARSA Agent"
                elif playerNum == 2:
                    return "QL Agent"