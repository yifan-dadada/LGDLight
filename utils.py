from sumo_env import *
from tqdm import tqdm


def control_simulation(log_path, episode, RL_net_path, rou_path, net_info, green_time, intersection_dic,
                       all_yellow_time, flow_stable_time, continue_green_time, GAT_net_agent, net_obs_lanes_state_dim):
    """
    控制整个仿真正常运行一轮
    """
    env = SumoEnv(log_path=log_path, simLibrary='libsumo', sim_round=episode, net_path=RL_net_path, rou_path=rou_path,
                  net_info=net_info, net_obs_lanes_state_dim=net_obs_lanes_state_dim)

    for tlsID in net_info.list_intersection_id:
        env.set_phase(tlsID=tlsID, phase_index=0)
        env.set_time(tlsID=tlsID, phase_time=green_time)
        intersection_dic[tlsID].state = env.get_observation_gat_net_state(graph=net_info.line_graph)
        intersection_dic[tlsID].action = 0

    done = False

    for sim_step in range(env.end_time):
        env.tls_phase_state = env.get_all_tls_current_phase(net_info.list_intersection_id)
        env.tls_time_state = env.get_all_next_switch_time(net_info.list_intersection_id)
        env.observation_gat_net_state = env.get_observation_gat_net_state(graph=net_info.line_graph)
        for tlsID in net_info.list_intersection_id:
            next_switch = env.tls_time_state[tlsID]
            if next_switch == 0:
                record_replay = env.switch_phase_action(intersection_class=intersection_dic[tlsID], net_info=net_info,
                                                        tlsID=tlsID, action=intersection_dic[tlsID].action,
                                                        state=intersection_dic[tlsID].state,
                                                        all_yellow_time=all_yellow_time, green_time=green_time,
                                                        continue_green_time=continue_green_time,
                                                        GAT_net_agent=GAT_net_agent)
                if record_replay and sim_step >= flow_stable_time:
                    intersection_dic[tlsID].replay_buffer.add(intersection_dic[tlsID].SARS[0],
                                                              intersection_dic[tlsID].SARS[1],
                                                              intersection_dic[tlsID].SARS[2],
                                                              intersection_dic[tlsID].SARS[3],
                                                              intersection_dic[tlsID].SARS[4],
                                                              intersection_dic[tlsID].SARS[5]
                                                              )
        if env.sim_step_now >= env.end_time - 1:
            done = True
            env.close()
            break
        else:
            env.step(step_num=1)


def test_agent_control_simulation(log_path, episode, RL_net_path, rou_path, net_info, green_time, continue_green_time,
                                  intersection_dic, all_yellow_time, GAT_net_agent, net_obs_lanes_state_dim):
    """
    测试agent
    """
    env = SumoEnv(log_path=log_path, simLibrary='libsumo', sim_round=episode, net_path=RL_net_path, rou_path=rou_path,
                  test_agent=True, net_info=net_info, net_obs_lanes_state_dim=net_obs_lanes_state_dim)

    returns_all_intersections = 0
    average_travel_time_round = 0

    for tlsID in net_info.list_intersection_id:
        env.set_phase(tlsID=tlsID, phase_index=0)
        env.set_time(tlsID=tlsID, phase_time=green_time)
        intersection_dic[tlsID].state = env.get_observation_gat_net_state(graph=net_info.line_graph)
        intersection_dic[tlsID].action = 0

    done = False

    for sim_step in range(env.end_time):
        env.tls_phase_state = env.get_all_tls_current_phase(net_info.list_intersection_id)
        env.tls_time_state = env.get_all_next_switch_time(net_info.list_intersection_id)
        env.observation_gat_net_state = env.get_observation_gat_net_state(graph=net_info.line_graph)
        for tlsID in net_info.list_intersection_id:
            next_switch = env.tls_time_state[tlsID]
            if next_switch == 0:
                record_reward, record_action = env.test_agent_switch_phase_action(
                    intersection_class=intersection_dic[tlsID], net_info=net_info,
                    tlsID=tlsID, action=intersection_dic[tlsID].action,
                    state=intersection_dic[tlsID].state,
                    all_yellow_time=all_yellow_time, green_time=green_time,
                    continue_green_time=continue_green_time,
                    test_agent=True,
                    GAT_net_agent=GAT_net_agent)
                if record_reward is not None:
                    returns_all_intersections += record_reward
                    intersection_dic[tlsID].reward_array.append(record_reward)
                    intersection_dic[tlsID].action_array.append(record_action)
        if env.sim_step_now >= env.end_time - 1:
            done = True
            env.close()
            average_travel_time_round = env.obtain_att_test_statistics()
            return returns_all_intersections, average_travel_time_round
        else:
            env.step(step_num=1)

def train_agent(net_info, intersection_dic, replay_times, batch_size, GAT_net_agent):
    for update_round in tqdm(range(replay_times), desc="Training Progress"):
        for tlsID in intersection_dic:
            b_s, b_a, b_r, b_ns, b_d, b_tls = intersection_dic[tlsID].replay_buffer.sample(batch_size)
            transition_dict = {
                'states': b_s,
                'actions': b_a,
                'next_states': b_ns,
                'rewards': b_r,
                'dones': b_d,
                'tls_id': b_tls
            }
            GAT_net_agent.update(transition_dict, tlsID)


def save_log(net_info, intersection_dic, log_path, total_reward_list, total_travel_time_list, GAT_net_agent):
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(f"{log_path}/loss"):
        os.makedirs(f"{log_path}/loss")
    if not os.path.exists(f"{log_path}/reward"):
        os.makedirs(f"{log_path}/reward")

    with open(f"{log_path}/loss/total_reward_list.txt", 'a') as file:
        for item in total_reward_list:
            file.write(f"{str(item)}\n")
    with open(f"{log_path}/reward/total_travel_time_list.txt", 'a') as file:
        for item in total_travel_time_list:
            file.write(f"{str(item)}\n")

    for tlsID in net_info.list_intersection_id:
        with open(f"{log_path}/reward/{tlsID}-reward.txt", 'a') as file:
            for item in intersection_dic[tlsID].reward_array:
                file.write(f"{str(item)}\n")

    with open(f"{log_path}/loss/intersection_loss.json", 'w', encoding='utf-8') as f:
        json.dump(GAT_net_agent.loss_array_net, f, ensure_ascii=False, indent=4)
