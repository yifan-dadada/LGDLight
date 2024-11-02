# from agent_DQN_single import *
from agent_D3QN_GAT import *
import numpy as np


class IntersectionAgent:
    def __init__(self, crossID, buffer_size, net_info, net_obs_lanes_state_dim,
                 learning_rate, gamma, epsilon, epsilon_decay, target_update, device):
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.crossID = crossID
        self.net_info = net_info
        self.net_obs_lanes_state_dim = net_obs_lanes_state_dim

        self.state = None
        self.action = None
        self.next_action_green = None
        self.SARS = None
        self.next_SA = None
        self.reward_array = []
        self.reward_array_total = []
        self.action_array = []

        self.action_dim = self.net_info.phase_num_dic[self.crossID]


    def draw_intersection_line(self, vis):
        self.reward_array_total.append(sum(self.reward_array)-sum(self.reward_array_total))
        vis.line(X=list(range(len(self.reward_array_total))), Y=self.reward_array_total, win=f'{self.crossID}-Round_reward',
                 opts=dict(title=f'{self.crossID}-Round_reward'))
        vis.line(X=list(range(len(self.reward_array))), Y=self.reward_array, win=f'{self.crossID}-reward',
                 opts=dict(title=f'{self.crossID}-reward'))
