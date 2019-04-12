from Api import Api
import numpy as np

botscore = 0
randomscore =0
draw= 0
do = True
while do:
    deck = np.arange(-5, 11)
    done = False
    api = Api()
    actionspace = np.arange(1, 16)
    while not done:

        random_action = 0
        while random_action == 0:
            random_action = np.random.choice(actionspace)
        actionspace[random_action-1]=0

        env_action = 0
        while env_action == 0:
            env_action = np.random.choice(deck)
        deck[env_action + 5] = 0
        done,bot_action,bot_score,random_score = api.do_action(random_action,env_action)

        print("Bot Score: "+ str(bot_score))
        print("Random policy Score: "+str(random_score))
        print("")

        if all(a == 0 for a in actionspace):
            done = True
            if bot_score > random_score:
                botscore +=1
            if bot_score < random_score:
                randomscore +=1
            if bot_score == random_score:
                draw +=1
            print("")
            print("Bot: " + str(botscore))
            print("Random: " + str(randomscore))
            if botscore > 0 or randomscore > 0 :
                print("Winrate: " + str(botscore/(randomscore+botscore)))
            if botscore + randomscore + draw == 50:
                do = False
                break