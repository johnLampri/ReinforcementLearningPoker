import numpy as np
from rlcard.games.rlpoker.utils import Hand


class ThresholdAgent(object):
    ''' A threshold agent  that will bet the maximum ammount with a high enough compination.
        In the first Round(when the public cards are unknown) it will always bet when it has a K or an A.
        In the second Round it will always bet with any pair.
    '''

    def __init__(self, num_actions):
       
        self.use_raw = False
        self.num_actions = num_actions

    @staticmethod
    def step(state):
        ''' Chooses  an action given the curent state in gerenerating training data.

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted by the agent
        '''
        legal_actions=state['raw_legal_actions']
        if(not state['raw_obs']['public_cards']):
            hand_card= state['raw_obs']['hand'][0]
            if 'A' in hand_card or 'K' in hand_card:
                if 'raise' in legal_actions:
                    return 1
                elif 'call' in legal_actions:
                    return 0
                elif 'check' in legal_actions:
                    return 3
                elif 'fold' in legal_actions:
                    return 2

            else:
                if 'check' in legal_actions:
                    return 3
                elif 'call' in legal_actions:
                    return 0
                elif 'raise' in legal_actions:
                    return 1
                elif 'fold' in legal_actions:
                    return 2
        else:
            if (i for i in state['raw_obs']['public_cards'] if hand_card[1] in i):
                if 'raise' in legal_actions:
                    return 1
                elif 'call' in legal_actions:
                    return 0
                elif 'check' in legal_actions:
                    return 3
                elif 'fold' in legal_actions:
                    return 2
            else:
                if 'check' in legal_actions:
                    return 3
                elif 'call' in legal_actions:
                    return 0
                elif 'raise' in legal_actions:
                    return 1
                elif 'fold' in legal_actions:
                    return 2
        return np.random.choice(list(state['legal_actions'].keys()))

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities
        '''
        #probs = [0 for _ in range(self.num_actions)]
       # for i in state['legal_actions']:
        #    probs[i] = 1/len(state['legal_actions'])

        info = {}
       # info['probs'] = {state['raw_legal_actions'][i]: probs[list(state['legal_actions'].keys())[i]] for i in range(len(state['legal_actions']))}

        return self.step(state), info 


























