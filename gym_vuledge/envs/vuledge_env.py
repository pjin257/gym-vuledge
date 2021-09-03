import gym
from gym import spaces
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import random
from gym_vuledge.envs.roadnet import ROADNET, GV
import logging
import networkx as nx

class VulEdgeEnv(gym.Env):
    """A vulnerable edge detection environment for OpenAI gym"""

    def __init__(self):
        super(VulEdgeEnv, self).__init__()

        # General variables defining the environment
        self.NUM_CAND = 10
        self.ONEHOT_CAND = LabelBinarizer().fit_transform(np.arange(0, self.NUM_CAND, 1))  # one-hot encode the k-candidates
        self.NUM_DISRUPT = 3

        # Load backbone road network 
        self.net = ROADNET(self.NUM_CAND)
        self.NUM_EDGE = self.net.G.number_of_edges()
        
        # Action. Define what the agent can do
        # Select the most vulnerable edge from the k-candidates
        self.action_space = spaces.Discrete(self.NUM_CAND)

        # Observation
        low = np.zeros(self.NUM_EDGE * 17)

        high_tmp1 = np.array([5000*5, 3, 2000, 2000, 2000])  # cap, sat, vis cnt, 1hop inst, 2hop inst
        high_tmp2 = np.full(1 + 1 + self.NUM_CAND, 1) # alive or not, bc, onehot cand
        high_edge = np.append(high_tmp1, high_tmp2) 

        high = np.array([])
        for i in range(0, self.NUM_EDGE):
            high = np.append(high, high_edge)

        self.observation_space = spaces.Box(low, high, shape=(self.NUM_EDGE * 17, ), dtype=np.float64)
        
        # episode over
        self.episode_over = True
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

        self.take_action(action)
        reward = self.get_reward()
        ob = self.net.get_state()

        return ob, reward, self.episode_over, self.info 

    def take_action(self, action):             
        
        self.net.disrupt(action)

        # End episode if finished
        if self.net.edge_atk.atk_cnt == self.NUM_DISRUPT:
            logging.info ('Disrupted all edges, ending episode')
            self.episode_over = True

    def get_reward(self):

        # reward as the average term delay
        curr_time = self.net.env.now
        acc_delay = 0
        v_cnt = 0
        atk_time = self.net.edge_atk.last_atk_time

        # Get delay of vehicles on the way
        for edge in self.net.G.edges:
            for v in self.net.tg.Q_dic[edge]:
                if v.gen_time < atk_time:
                    delay = curr_time - atk_time
                else:
                    delay = curr_time - v.gen_time
                acc_delay += delay
                v_cnt += 1

        # Get delay of vehicles finished their travels
        for v in self.net.mv.finished:
            if v.arrival_time >= atk_time:
                delay = v.arrival_time - atk_time
                acc_delay += delay
                v_cnt += 1
        
        reward = acc_delay / v_cnt

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
        self.net.init_env()
                       
        # run simulation until the end of warming-up time to get the initial observation
        self.net.env.run(until=GV.WARMING_UP)

        return self.net.get_state()

    def render(self, mode='human', close=False):
        # no render. may be in future works?
        return 1   
