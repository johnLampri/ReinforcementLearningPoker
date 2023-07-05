import random
import numpy as np
import collections

import os
import pickle

from rlcard.utils.utils import *

class PolicyIterationAgent():


    def __init__(self, env,model_path='./qlearning_model',a=0.5,g=0.9, exploration_decay=0.001):
        self.exploration_prob= 1
        self.gamma=g
        self.a=a
        self.min_exploration_prob=0.01
        self.exploration_prob_decay=0.001
        self.use_raw=False
        self.env=env
        self.model_path = model_path
        self.iteration=0
        self.policy = collections.defaultdict(np.array)

        self.possible_public_cards=[]
        self.possible_agent_cards=[]
        self.possible_adversary_cards=[]

        self.set_cards()

        self.current_public_cards=None

    def check_current_player_is_Qagent(self,current_player):
        agents=self.env.get_agents()
        for id,agent in enumerate(agents):
            if isinstance(agent,PolicyIterationAgent):
                if(self.env.get_player_id()==id):
                    return True
        return False    
    
    def get_Qagent(self):
        agents=self.env.get_agents()
        for id,agent in enumerate(agents):
            if isinstance(agent,PolicyIterationAgent):
                return id

    def set_cards(self):
        '''
        Initialises all the possible card in hands and on the table(public cards)'''
        ranks=self.get_rank_list()
        self.possible_agent_cards=ranks
        self.possible_adversary_cards=ranks
        for i in ranks:
             for j in ranks:
               self.possible_public_cards.append(i+j)

    def train(self):
        
        while True:
            old_pi = self.policy.copy()
            V=self.policy_evaluation()
            self.iteration+=1
            if self.compare_policies(old_pi,self.policy): # you have converged to the optimal policy if the "improved" policy is exactly the same as in the previous step
                break
            if self.iteration ==20:
                break
    
    def compare_policies(self,old_pi,pi):
        '''
        A function that compares two dictionairies
        '''
        keys = old_pi.keys()
        # Convert keys to a list if needed
        keys_list = list(keys)


        keys2 = pi.keys()

        # Convert keys to a list if needed
        keys_list2 = list(keys2)

        if keys_list2 == keys_list:

            same_items = all(np.array_equal(old_pi[key], pi[key]) for key in old_pi)
            if same_items:
               return True
            else:
                return False

    def policy_evaluation(self):
        for i in self.possible_agent_cards:
            for j in self.possible_adversary_cards:
                for u in self.possible_public_cards:
                    self.env.reset()
                    self.current_public_cards=u
                    self.change_game_state(i,j)
                    self.traverse_tree()



    def change_game_state(self,agent_card,adversary_card):
        '''Changes the environment to train the Agent
        '''
        current_player=self.env.get_player_id()
        cards=self.get_rank_list()
        agent_id = self.get_Qagent()
        adversary_id= (agent_id-1)%2
        self.env.game.players[agent_id].hand[0]=Card('H',agent_card)
        self.env.game.players[adversary_id].hand[0]=Card('H',adversary_card)
        pcards=self.env.game.public_cards
        if pcards:
            self.env.game.public_cards[0]=Card('H',self.current_public_cards[0])
            self.env.game.public_cards[1] = Card('H',self.current_public_cards[1])

    def check_public_cards(self):
        '''
        Checks if public cards have been dealt
        '''
        pcards=self.env.game.public_cards
        if pcards:
            self.env.game.public_cards[0]=Card('H',self.current_public_cards[0])
            self.env.game.public_cards[1] = Card('H',self.current_public_cards[1])


    def model_change(self,obs):
            a= np.where(obs[5:10] ==1)
            b= np.where(obs[10:15] ==1)
            if a[0]!=[] and b[0]!=[]:
                if(a[0][0] <b[0][0]):
                    temp=obs[10:15].copy()
                    obs[10:15]=obs[5:10]
                    obs[5:10]=temp
            return obs

    # def custom_reward(self,reward,Qaction):
    #         if Qaction == 0 or Qaction == 1 or Qaction == 3 :
    #             if  reward<0:
    #                 return -reward
    #             else:
    #                 return reward
    #         if Qaction==2:
    #             if reward>0:
    #                 return -reward
    #             else:
    #                 return reward

    def get_represented_state(self,obs):
        list=['A','T','J', 'Q', 'K']
        hand= list[np.argmax(obs[0:5])]
        contains1=False
        for i in obs[5:15]:
            if i!=0:
                contains1=True
                break    
        if  contains1:
            public_card1= list[np.argmax(obs[5:10])]
            public_card2= list[np.argmax(obs[10:15])]
        else:
            public_card1=''
            public_card2=''
        mycoins= str(np.argmax(obs[15:22])%7)
        adversarycoins= str(np.argmax(obs[22:29])%7)
        return hand+public_card1+public_card2+mycoins+adversarycoins


    def traverse_tree(self):
        self.check_public_cards()
        current_player = self.env.get_player_id()
        if self.env.is_over():
            id=self.get_Qagent()

            return self.env.get_payoffs()[id] 
        if self.check_current_player_is_Qagent(current_player):
            #V=0
            state=self.env.get_state(current_player)
            obs1=state['obs']
            obs1=self.model_change(obs1)
            obs2=tuple(obs1)
            obs=self.get_represented_state(obs2)
            _, legal_actions = self.get_state(current_player)
            reward= {}

            #initialise the state in the policy dictionary(we use -inf for the non legal actions)

            if obs not in self.policy:
                self.policy[obs]=[float('-inf'),float('-inf'),float('-inf'),float('-inf')]
                for i in  legal_actions:
                    self.policy[obs][i]= 0
            propability=self.action_probs(obs,legal_actions,self.policy)
            for action in legal_actions:
                prob=propability[action]
                self.env.step(action)
                v=self.traverse_tree()
                self.env.step_back()
                reward[action]= v
            self.policy_improvement(obs,reward,v,prob,legal_actions)
            return v*self.gamma
        else:
            state = self.env.get_state(current_player)
            action= self.env.agents[current_player].step(state)
            self.env.step(action)
            temp= self.traverse_tree()
            self.env.step_back()     

            return temp


    def policy_improvement(self,obs,reward,payoff,prob,legal_actions):
        r=[float('-inf'),float('-inf'),float('-inf'),float('-inf')]
        for i in reward:
            r=reward[i]
        probs=self.softmax(r)
        for i in legal_actions:
            self.policy[obs][i]+=prob*(payoff+self.gamma*reward[i])



    def softmax(self,x):
        e_x = np.exp(x - np.max(x))  # Subtracting the maximum value for numerical stability
        return e_x / np.sum(e_x)

    def action_probs(self, obs, legal_actions, policy):
        ''' Obtain the action probabilities of the current state

        Args:
            obs (str): state_str
            legal_actions (list): List of leagel actions
            player_id (int): The current player
            policy (dict): The used policy

        Returns:
            (tuple) that contains:
                action_probs(numpy.array): The action probabilities
                legal_actions (list): Indices of legal actions
        '''
        if obs not in policy.keys():
            action_probs = np.zeros(self.env.num_actions)
            random_action= random.choice(legal_actions)
            action_probs[random_action]=1
            self.policy[obs] = action_probs
        else:
            action_probs = policy[obs].copy()
            action_probs1= self.softmax(action_probs)
            action_probs = remove_illegal(action_probs1, legal_actions)
        return action_probs

    def get_rank_list(self):
        return ['A','T','J','Q','K']

    def step(self,state):   
        ''' Predict the action for genrating training data but
            have the predictions disconnected from the computation graph

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
        '''
        probs = self.action_probs(state['obs'].tostring(), list(state['legal_actions'].keys()), self.policy)
        action = np.argmax(probs)

        info = {}
        info['probs'] = {state['raw_legal_actions'][i]: float(probs[list(state['legal_actions'].keys())[i]]) for i in range(len(state['legal_actions']))}

        return action, info


    def eval_step(self, state):
        ''' Given a state, predict action based on average policy

        Args:
        state (numpy.array): State representation

        Returns:
        action (int): Predicted action
        info (dict): A dictionary containing information
        '''

        return self.step(state)

    def get_state(self, player_id):
        ''' Get state_str of the player

        Args:
        player_id (int): The player id

        Returns:
        (tuple) that contains:
        state (str): The state str
        legal_actions (list): Indices of legal actions
        '''
        state = self.env.get_state(player_id)
        return state['obs'].tostring(), list(state['legal_actions'].keys())



    def save(self):
        ''' Save model
        '''
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        policy_file = open(os.path.join(self.model_path, 'quality.pkl'),'wb')
        pickle.dump(self.policy, policy_file)
        policy_file.close()



        iteration_file = open(os.path.join(self.model_path, 'iteration.pkl'),'wb')
        pickle.dump(self.iteration, iteration_file)
        iteration_file.close()

    def load(self):
        ''' Load model
        '''
        if not os.path.exists(self.model_path):
            return

        policy_file = open(os.path.join(self.model_path, 'policy.pkl'),'wb')
        pickle.dump(self.policy, policy_file)
        policy_file.close()

        iteration_file = open(os.path.join(self.model_path, 'iteration.pkl'),'rb')
        self.iteration = pickle.load(iteration_file)
        iteration_file.close()

