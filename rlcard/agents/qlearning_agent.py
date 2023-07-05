import random
import numpy as np
import collections

import os
import pickle

from rlcard.utils.utils import *

class QLearningAgent():


    def __init__(self, env,model_path='./qlearning_model',a=0.8,g=0.3, exploration_decay=0.001):
        self.exploration_prob= 1
        self.gamma=g
        self.a=a
        self.min_exploration_prob=0.01
        self.exploration_prob_decay=0.001
        self.use_raw=False
        self.env=env
        self.model_path = model_path
        self.iteration=0
        self.reward = collections.defaultdict(np.array)

    def check_current_player_is_Qagent(self,current_player):
        agents=self.env.get_agents()
        for id,agent in enumerate(agents):
            if isinstance(agent,QLearningAgent):
                if(self.env.get_player_id()==id):
                    return True
        return False    
    
    def get_Qagent(self):
        agents=self.env.get_agents()
        for id,agent in enumerate(agents):
            if isinstance(agent,QLearningAgent):
                return id

    def train(self):
        ''' Do one iteration of Q Learning
        '''
        # Firstly, traverse tree  for each player
        # The values are recorded in traversal

        current_player = self.env.get_player_id()
        if self.check_current_player_is_Qagent(current_player):

            self.env.reset()
            self.traverse_tree()
            if self.check_current_player_is_Qagent(current_player):
                self.iteration=+1
                self.exploration_prob=max(self.min_exploration_prob, np.exp(-self.exploration_prob_decay*self.iteration))
        return 

    def decide_Qaction(self,reward,legal_actions):
        #decides the action on the current step according to the Q Learning algorithm 
        if np.random.rand()<self.exploration_prob:
            Qaction= np.random.choice(legal_actions)               
        else:
            reward_sublist = [float('-inf'),float('-inf'),float('-inf'),float('-inf')] 
            for i in  legal_actions:
                reward_sublist[i]= reward[i]
            Qaction= np.argmax(reward_sublist) 
        return Qaction

    def traverse_tree(self):
        current_player = self.env.get_player_id()
        if self.env.is_over():
            id=self.get_Qagent()
            #returns the reward
            return self.env.get_payoffs()[id] ,0#self.env.get_payoffs()[id] 
        if self.check_current_player_is_Qagent(current_player):
            #gets the current state
            state=self.env.get_state(current_player)
            obs1=state['obs']
            legal_actions=state['legal_actions']
            #we modify the state array
            obs1=self.model_change(obs1)
            obs2=tuple(obs1)
            obs=self.get_represented_state(obs2)
            #obs, legal_actions = self.get_state(current_player)
            reward= {}
            #initialise the state in the reward dictionary(we use -inf for the non legal actions)
            if obs not in self.reward:
                self.reward[obs]=[float('-inf'),float('-inf'),float('-inf'),float('-inf')]
                for i in  legal_actions:
                    self.reward[obs][i]= 0
            Qaction=self.decide_Qaction(self.reward[obs],legal_actions)
            #we act on the chosen action
            self.env.step(Qaction)
            temp_return= self.traverse_tree()
            temp, qnext=temp_return
            self.env.step_back()
            reward[Qaction]=temp#self.custom_reward(temp,Qaction)
            self.update_policy(obs,reward,legal_actions,Qaction, qnext)
            return temp, self.gamma*np.max(self.reward[obs])
        else :
            #if the opponent plays, we get his action and we move to the next round
            state = self.env.get_state(current_player)
            action= self.env.agents[current_player].step(state)
            self.env.step(action)
            temp,ff= self.traverse_tree()
            self.env.step_back()     

            return temp , ff

    def update_policy(self, obs, next_state_values, legal_actions,current_action,qnext):
        ''' Update policy based on the current values
        '''

        Qvalue=self.reward[obs][current_action]
        #we update the value
        Qvalue=(1-self.a)*Qvalue+self.a*(next_state_values[current_action] + self.gamma*qnext )
        self.reward[obs][current_action]=Qvalue

    
    def step(self,state):
        ''' Predict the action for genrating training data but
            have the predictions disconnected from the computation graph

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
        '''

        obs1=state['obs']
        obs1=self.model_change(obs1)
        obs2=tuple(obs1)
        obs=self.get_represented_state(obs2)
        legal_actions=list(state['legal_actions'].keys())
        if obs not in self.reward:
            self.reward[obs]=[float('-inf'),float('-inf'),float('-inf'),float('-inf')]
            for i in  legal_actions:
                 self.reward[obs][i]= 0
            action= np.argmax(self.reward[obs])
            action_integer=int(action)    

        else:
            actions=self.reward[obs].copy()
            action=np.argmax(actions)
            action_integer=int(action)    

        info = {}
        #info['probs'] = {state['raw_legal_actions'][i]: float(decisions[list(state['legal_actions'].keys())[i]]) for i in range(len(state['legal_actions']))}

        return action_integer, info


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

        quality_file = open(os.path.join(self.model_path, 'quality.pkl'),'wb')
        pickle.dump(self.reward, quality_file)
        quality_file.close()

        iteration_file = open(os.path.join(self.model_path, 'iteration.pkl'),'wb')
        pickle.dump(self.iteration, iteration_file)
        iteration_file.close()

    def load(self):
        ''' Load model
        '''
        if not os.path.exists(self.model_path):
            return

        quality_file = open(os.path.join(self.model_path, 'quality.pkl'),'wb')
        pickle.dump(self.reward, quality_file)
        quality_file.close()

        iteration_file = open(os.path.join(self.model_path, 'iteration.pkl'),'rb')
        self.iteration = pickle.load(iteration_file)
        iteration_file.close()



    def model_change(self,obs):
            #The order of the public cards are not important so we order the cards by rank to end up with less states
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




