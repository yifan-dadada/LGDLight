import collections
import random
from collections import namedtuple, deque

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.optim import Adam, AdamW

from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.data import Batch

class ReplayBuffer:
    """ 经验回放池 """

    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done, tls_id):
        self.buffer.append((state, action, reward, next_state, done, tls_id))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, tls_id = zip(*transitions)
        return state, action, reward, next_state, done, tls_id

    def size(self):
        return len(self.buffer)


class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_heads=4):
        super(GATNet, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        return x


class DuelingGATNet(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_heads, tls_ids_action_dim, tls_edges):
        super(DuelingGATNet, self).__init__()
        self.tls_edges = tls_edges

        self.gat_net = GATNet(in_channels, hidden_channels, num_heads)

        self.value_streams = nn.ModuleDict({
            f"value_{tls_id}": nn.Sequential(
                nn.Linear(hidden_channels * len(tls_edges[tls_id]), 1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            ) for tls_id in tls_ids_action_dim
        })

        self.advantage_streams = nn.ModuleDict({
            f"advantage_{tls_id}": nn.Sequential(
                nn.Linear(hidden_channels * len(tls_edges[tls_id]), 1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, tls_ids_action_dim[tls_id])
            ) for tls_id in tls_ids_action_dim
        })

    def forward(self, data, tls_id):
        x = self.gat_net(data)

        batch_size = data.num_graphs
        node_features_batch = []
        for i in range(batch_size):
            node_features = torch.cat([x[data.batch == i][idx].unsqueeze(0) for idx in self.tls_edges[tls_id]], dim=1)
            node_features_batch.append(node_features)
        node_features_batch = torch.cat(node_features_batch, dim=0)

        value_stream = self.value_streams[f"value_{tls_id}"](node_features_batch)
        advantage_stream = self.advantage_streams[f"advantage_{tls_id}"](node_features_batch)

        q_values = value_stream + advantage_stream - advantage_stream.mean(dim=1, keepdim=True)

        return q_values

class SharedGATAgent:
    def __init__(self, tls_ids_action_dim, tls_edges, in_channels, hidden_channels, learning_rate, gamma, epsilon,
                 epsilon_decay, target_update, buffer_size, device):

        self.tls_ids_action_dim = tls_ids_action_dim
        self.tls_edges = tls_edges
        self.device = device
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.gamma = gamma

        self.q_net = DuelingGATNet(in_channels, hidden_channels, num_heads=4, tls_ids_action_dim=tls_ids_action_dim, tls_edges=tls_edges).to(device)
        self.target_q_net = DuelingGATNet(in_channels, hidden_channels, num_heads=4, tls_ids_action_dim=tls_ids_action_dim, tls_edges=tls_edges).to(device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = AdamW(self.q_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.loss_array_net = {key: [] for key in tls_ids_action_dim}
        self.count = 0

        print(f"Agent_model---\n", self.q_net)

    def take_action(self, state, tls_id, test_agent=False):
        state = state.to(self.device)
        if test_agent:
            state = Batch.from_data_list([state]).to(self.device)
            q_values = self.q_net(state, tls_id)
            action = q_values[0].argmax().item()
        else:
            if np.random.random() < self.epsilon:
                action = np.random.randint(self.tls_ids_action_dim[tls_id])
            else:
                state = Batch.from_data_list([state]).to(self.device)
                q_values = self.q_net(state, tls_id)
                action = q_values[0].argmax().item()
        return action

    def update(self, transition_dict, tls_id):
        states_data = [state.to(self.device) for state in transition_dict['states']]
        next_states_data = [next_state.to(self.device) for next_state in transition_dict['next_states']]

        states = Batch.from_data_list(states_data).to(self.device)
        next_states = Batch.from_data_list(next_states_data).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        tls_ids = transition_dict['tls_id']
        q_values = self.q_net(states, tls_id).gather(1, actions)

        with torch.no_grad():
            max_action = self.q_net(next_states, tls_id).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states, tls_id).gather(1, max_action)
            q_targets = rewards + self.gamma * max_next_q_values

        dqn_loss = self.loss_fn(q_values, q_targets)
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1

        self.loss_array_net[tls_id].append(dqn_loss.item())

    def save(self, path):
        torch.save(self.q_net.state_dict(), f"{path}_gat.pth")
        for tls_id in self.tls_ids_action_dim:
            torch.save(self.q_net.value_streams[f"value_{tls_id}"].state_dict(), f"{path}_{tls_id}_value.pth")
            torch.save(self.q_net.advantage_streams[f"advantage_{tls_id}"].state_dict(), f"{path}_{tls_id}_advantage.pth")

    def load(self, path):
        self.q_net.load_state_dict(torch.load(f"{path}_gat.pth"))
        for tls_id in self.tls_ids_action_dim:
            self.q_net.value_streams[f"value_{tls_id}"].load_state_dict(torch.load(f"{path}_{tls_id}_value.pth"))
            self.q_net.advantage_streams[f"advantage_{tls_id}"].load_state_dict(torch.load(f"{path}_{tls_id}_advantage.pth"))

    def draw_loss_line(self, vis):
        for cross in self.loss_array_net:
            vis.line(X=list(range(len(self.loss_array_net[cross]))), Y=self.loss_array_net[cross],
                     win=f'{cross}-loss_GAT_net',
                     opts=dict(title=f'{cross}-loss_GAT_net'))
