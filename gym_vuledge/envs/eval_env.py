import gym
from gym import spaces
import numpy as np
from gym_vuledge.envs.evalnet import ROADNET
import logging

class EvalEnv(gym.Env):
    """A vulnerable edge detection environment for OpenAI gym"""

    def __init__(self):
        super(EvalEnv, self).__init__()

        # General variables defining the environment
        self.NUM_DISRUPT = 5

        # Load backbone road network 
        self.net = ROADNET()
        self.NUM_EDGE = self.net.G.number_of_edges()
        
        # Action. Define what the agent can do
        # Select the most vulnerable edge from the k-candidates
        self.action_space = spaces.Discrete(len(self.net.edge_atk.candidates))

        # Observation
        low = np.zeros(self.NUM_EDGE * 7)

        # cap veh, current veh, vis cnt, forward_flow_cnt, alive_or_not, speed_limit, length
        high_per_edge = np.array([3000, 3000, 1000, 3000, 1, 110, 3000])

        high = np.array([])
        for i in range(0, self.NUM_EDGE):
            high = np.append(high, high_per_edge)

        self.observation_space = spaces.Box(low, high, shape=(self.NUM_EDGE * 7, ), dtype=np.int16)
        
        # episode over
        self.episode_over = False
        self.info = {}

    def step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : int

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (That is, any links
                in the topology has been saturated)
            info (dict) :
                diagnostic information useful for debugging. It can sometimes
                be useful for learning (for example, it might contain the raw
                probabilities behind the environment's last state change).
                However, official evaluations of your agent are not allowed to
                use this for learning.
        """

        is_dup_action = self.take_action(action)
        reward = self.get_reward(is_dup_action)
        ob = self.net.get_state()

        return ob, reward, self.episode_over, self.info 

    def take_action(self, action):             
        
        is_dup_action = self.net.disrupt(action)

        # End episode if finished
        if self.net.edge_atk.atk_cnt == self.NUM_DISRUPT:
            logging.info ('Disrupted all edges, ending episode')
            self.episode_over = True

        return is_dup_action

    def get_reward(self, is_dup_action):

        # reward as the difference in the # of vehicles between current system and baseline
        # baseline is the expected number of vehicles in network without any disruption
        baselines = [873, 919, 943, 765, 0] # averaged over 100 simulations
        vnum_wo_disruption = baselines[self.net.edge_atk.atk_cnt - 1]
        current_vnum = self.net.mv.v_num[-1]

        diff = current_vnum - vnum_wo_disruption

        if self.net.edge_atk.atk_cnt == 5:
            reward = diff # higher reward at the end of episode
        else:
            reward = diff / 3

        if is_dup_action: reward = 0 # constant negative reward if action is duplicated

        return reward

    def reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """

        self.episode_over = False

        # initialize graph and simpy env
        self.net.init_graph()

        return self.net.get_state()

    def render(self, mode='human', close=False):
        # no render. may be in future works?
        return 1   
