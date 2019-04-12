import numpy as np
import copy
class Environment:
    def __init__(self):
        self.deck = np.arange(-5,11)
        self.curr_deck = np.arange(-5,11)
        self.player0score = 0
        self.player1score = 0

    def full_reset(self):
        self.deck = np.arange(-5, 11)
        self.curr_deck = np.arange(-5, 11)
        self.player0score = 0
        self.player1score = 0


    def reset_env(self):
        self.deck = copy.copy(self.curr_deck)
        self.player0score = 0
        self.player1score = 0

    def reset_deck(self):
        self.deck = copy.copy(self.curr_deck)

    def reset49(self,player0buffer,player1buffer):
        self.deck = copy.copy(self.curr_deck)
        self.player0score = player0buffer
        self.player1score = player1buffer

    def update_env(self,deck):
        self.curr_deck = deck

    def step(self,action0,action1):
        local_deck = np.array([i for i in self.deck if i != 0])
        target_card = np.random.choice(local_deck)
        self.deck[target_card+5] = 0
        if action0 > action1 and target_card > 0:
            self.player0score += target_card
        if action0 > action1 and target_card < 0:
            self.player1score += target_card
        if action0 < action1 and target_card > 0:
            self.player1score += target_card
        if action0 < action1 and target_card < 0:
            self.player0score += target_card
        return target_card

    def check_status(self):
        if all (i==0 for i in self.deck):
            return True
        else: return False

    def eval_game(self):
        if self.player0score > self.player1score:
            return 1
        if self.player0score < self.player1score:
            return -1
        if self.player0score == self.player1score:
            return 0

    def get_player_scores(self):
        return [self.player0score,self.player1score]

    def update_values(self,bot_action,random_action,target_card):
        #self.deck[target_card + 5] = 0
        if bot_action > random_action and target_card > 0:
            self.player0score += target_card
        if bot_action > random_action and target_card < 0:
            self.player1score += target_card
        if bot_action < random_action and target_card > 0:
            self.player1score += target_card
        if bot_action < random_action and target_card < 0:
            self.player0score += target_card
        self.curr_deck[target_card+5]=0
        return [self.player0score,self.player1score]