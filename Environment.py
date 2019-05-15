import numpy as np
import copy
class Environment:
    def __init__(self):
        self.deck = np.arange(-5,11)
        self.black_score = 0
        self.white_score = 0

    def set_env(self,state):
        self.deck = copy.copy(state[30:46])
        self.black_score = copy.copy(state[-2])
        self.white_score = copy.copy(state[-1])
    
    def full_reset(self):
        self.deck = np.arange(-5, 11)
        self.black_score = 0
        self.white_score = 0

    def step(self,black_action,white_action):

        local_deck = np.array([i for i in self.deck if i != 0])
        env_action = np.random.choice(local_deck)
        self.deck[env_action+5] = 0

        if black_action == white_action:
            return env_action
        if black_action > white_action and env_action > 0:
            self.black_score += env_action
        if black_action > white_action and env_action < 0:
            self.white_score += env_action
        if black_action < white_action and env_action > 0:
            self.white_score += env_action
        if black_action < white_action and env_action < 0:
            self.black_score += env_action
        return env_action

    def check_status(self):
        if all (i==0 for i in self.deck):
            return True
        else: return False

    def eval_game(self):
        if self.black_score > self.white_score:
            return 1
        if self.black_score < self.white_score:
            return -1
        if self.black_score == self.white_score:
            return 0

    def get_player_scores(self):
        return [self.black_score,self.white_score]

    def update_values(self,bot_action,random_action,env_action):
        if bot_action > random_action and env_action > 0:
            self.black_score += env_action
        if bot_action > random_action and env_action < 0:
            self.white_score += env_action
        if bot_action < random_action and env_action > 0:
            self.white_score += env_action
        if bot_action < random_action and env_action < 0:
            self.black_score += env_action
        return [self.black_score,self.white_score]