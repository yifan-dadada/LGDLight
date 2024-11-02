# -*- coding: utf-8 -*-

import time
import json
import os.path

from agent_D3QN_GAT import *
from sumo_net_data import *
from intersection_agent import *
from utils import *

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(torch.cuda.is_available())
print(device)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# TODO
learning_rate = 1e-4
num_episodes = 202
gamma = 0.90
epsilon = 1
epsilon_decay = 0.96
target_update = 10
buffer_size = 50000
batch_size = 64

all_yellow_time = 5
green_time = 10
continue_green_time = 5
total_reward_list = []
total_travel_time_list = []
best_travel_time = 10000000

flow_stable_time = 600
episode_replay_update = 1
replay_times_base = 10

# TODO
sim_file_path = Your_Data_Directory
RL_net_path = Your_Road_Network_File
rou_path = Your_Vehicle_File

rl_data_txt_path = os.path.join(sim_file_path, f"rl_data.txt")
this_train_time = str(time.strftime('%Y%m%d%H%M%S'))
log_path = f"./logs_train/{sim_file_path.split('/')[-1]}-{this_train_time}"


net_info = NetInfo(net_path=RL_net_path, rl_data_txt_path=rl_data_txt_path)
target_update *= len(net_info.list_intersection_id)

net_obs_lanes_num_dim = len(net_info.lane_list)
net_obs_lanes_state_dim = 10

GAT_net_agent = SharedGATAgent(tls_ids_action_dim=net_info.phase_num_dic, tls_edges=net_info.tls_edges,
                               in_channels=net_info.max_lanes*net_obs_lanes_state_dim, hidden_channels=64,
                               learning_rate=learning_rate, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay,
                               target_update=target_update, buffer_size=buffer_size, device=device)

intersection_dic = {}
for tls_id in net_info.list_intersection_id:
    intersection_dic[tls_id] = IntersectionAgent(tls_id, buffer_size, net_info, net_obs_lanes_state_dim,
                                                 learning_rate, gamma, epsilon, epsilon_decay, target_update, device)

for episode in range(num_episodes):
    if episode == 0:
        GAT_net_agent.optimizer = AdamW(GAT_net_agent.q_net.parameters(), lr=1e-4)  # 改变学习率为0.00001
    if episode == 110:
        GAT_net_agent.optimizer = AdamW(GAT_net_agent.q_net.parameters(), lr=1e-5)  # 改变学习率为0.00001  # 改变学习率为0.00001
    print("-----sim_round-----:", {episode})
    sim_round_total = 0

    control_simulation(log_path, episode, RL_net_path, rou_path, net_info, green_time, intersection_dic, all_yellow_time,
                       flow_stable_time, continue_green_time, GAT_net_agent, net_obs_lanes_state_dim)

    if episode >= episode_replay_update:
        print(f"---update仿真第{episode - episode_replay_update}轮的update---开始！")

        if episode <= 100:
            replay_times = episode * replay_times_base
        else:
            replay_times = 100 * replay_times_base
        print(f"--update仿真第{episode - episode_replay_update}轮，replay_times--:", replay_times)
        train_agent(net_info, intersection_dic, replay_times, batch_size, GAT_net_agent)
        epsilon = epsilon * epsilon_decay
        if epsilon < 0.1:
            epsilon = 0.1
        GAT_net_agent.epsilon = epsilon
        print("--update仿真第{episode-episode_replay_update}轮，探索率 epsilon---:", epsilon)
        print(f"---update仿真第{episode - episode_replay_update}轮的update---结束！")

        print(f"-----test仿真第{episode - episode_replay_update}轮-----开始！")
        returns_all_intersections, average_travel_time_round = (
            test_agent_control_simulation(log_path, episode, RL_net_path, rou_path, net_info, green_time,
                                          continue_green_time, intersection_dic, all_yellow_time,
                                          GAT_net_agent, net_obs_lanes_state_dim))
        total_reward_list.append(returns_all_intersections)
        total_travel_time_list.append(float(average_travel_time_round[0]))
        print(f"-----test仿真第{episode - episode_replay_update}轮-----: 总的奖励为：{returns_all_intersections}，\n"
              f"平均行驶时间：{average_travel_time_round[0]}，车辆数：{average_travel_time_round[1]}，"
              f"速度：{average_travel_time_round[2]}， 损失时间：{average_travel_time_round[3]}")
        print("平均旅行时间结果average_travel_time:", total_travel_time_list)
        if float(average_travel_time_round[0]) < best_travel_time:
            best_travel_time = float(average_travel_time_round[0])
            model_path = f"./models/{log_path.split('/')[-1]}"
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(GAT_net_agent.q_net, f"{model_path}/{'GAT'}-q_net.pth")
            torch.save(GAT_net_agent.target_q_net, f"{model_path}/{'GAT'}-target_q_net.pth")

save_log(net_info, intersection_dic, log_path, total_reward_list, total_travel_time_list, GAT_net_agent)
