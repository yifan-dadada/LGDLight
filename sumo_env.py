import json
import os
import sys
import xml.etree.ElementTree as ET
import collections
import random

import numpy as np
from sumolib import checkBinary  # noqa

# if 'SUMO_HOME' in os.environ:
#     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#     sys.path.append(tools)
# else:
#     sys.exit("please declare environment variable 'SUMO_HOME'")

import collections
import numpy as np

import torch
from torch_geometric.data import Data


class SumoEnv:
    """
    开启仿真的类
    """

    def __init__(self, log_path, simLibrary, sim_round, net_info, net_obs_lanes_state_dim,
                 net_path=None, rou_path=None, sumocfg_path=None, test_agent=False):
        self.sim_step_now = 0
        if sumocfg_path is not None:
            self.sumocfg_xml_path = sumocfg_path
        self.end_time = 3600
        self.test_agent = test_agent
        self.sim_round = sim_round
        self.net_info = net_info
        self.net_obs_lanes_state_dim = net_obs_lanes_state_dim

        cmd_parameters = []
        if test_agent:
            self.log_path_sumo = os.path.join(f"{log_path}", "log_sumo")
            print("log_path_sumo:", self.log_path_sumo)
            if not os.path.exists(self.log_path_sumo):
                os.makedirs(self.log_path_sumo)
                os.makedirs(os.path.join(self.log_path_sumo, "statistic"))
                os.makedirs(os.path.join(self.log_path_sumo, "tripinfo"))
            cmd_parameters = [None, '-n', net_path, '-r', rou_path, '--no-warnings', 'true',
                              # "-c", self.sumocfg_xml_path,
                              "--start", "--quit-on-end",
                              "--duration-log.statistics",
                              # "--tripinfo-output.write-unfinished",
                              "--statistic-output",
                              os.path.join(self.log_path_sumo, "statistic", f"{self.sim_round}.xml"),
                              # "--queue-output", self.queue_output_xml_path,
                              "--tripinfo-output",
                              os.path.join(self.log_path_sumo, "tripinfo", f"{self.sim_round}.xml")]
        else:
            cmd_parameters = [None, '-n', net_path, '-r', rou_path, '--no-warnings', 'true',
                              # "-c", self.sumocfg_xml_path,
                              "--start", "--quit-on-end", ]

        if simLibrary == "sumo-gui":
            import traci
            # traci.simulation.getCurrentTime()
            self.sumoBinary = checkBinary('sumo-gui')  # 默认开启sumo
            cmd_parameters[0] = self.sumoBinary
            self.con = traci.start(cmd_parameters, label=str(self.sim_round))
            self.traci_con = traci.getConnection(label=str(self.sim_round))
            print("开启仿真gui！")

        elif simLibrary == "libsumo":
            import libsumo as traci_con
            self.traci_con = traci_con
            self.sumoBinary = checkBinary('sumo')
            cmd_parameters[0] = self.sumoBinary
            self.con = self.traci_con.start(cmd_parameters, label=str(self.sim_round))
            print("不开启仿真gui，使用libsumo接口和libsumo库仿真")
        else:
            import traci
            self.sumoBinary = checkBinary('sumo')
            cmd_parameters[0] = self.sumoBinary
            self.con = traci.start(cmd_parameters, label=str(self.sim_round))
            self.traci_con = traci.getConnection(label=str(self.sim_round))
            print("不开启仿真gui，使用traci接口和sumo库仿真")

        self.lane_controlled_by_tls = {}
        for tls_id in net_info.list_intersection_id:
            tls_control_lanes = self.traci_con.trafficlight.getControlledLanes(tlsID=tls_id)
            for lane_id in net_info.lane_list:
                if lane_id in tls_control_lanes:
                    self.lane_controlled_by_tls[lane_id] = tls_id
        self.tls_phase_state = self.get_all_tls_current_phase(net_info.list_intersection_id)
        self.tls_time_state = self.get_all_next_switch_time(net_info.list_intersection_id)
        self.observation_gat_net_state = self.get_observation_gat_net_state(graph=net_info.line_graph)


    def step(self, step_num=1):
        for step in range(step_num):
            self.traci_con.simulationStep()
            self.sim_step_now += 1
            if self.sim_step_now >= self.end_time:
                return True
        return False

    def next_switch_time(self, tlsID):
        return self.traci_con.trafficlight.getNextSwitch(tlsID=tlsID) - self.traci_con.simulation.getTime()

    def get_all_next_switch_time(self, list_intersection_id):
        """输入红绿灯的id，返回字典，包含每个信号灯当前的相位剩余时长"""
        all_tls_time_state = {}
        for tls_id in list_intersection_id:
            all_tls_time_state[tls_id] = self.next_switch_time(tlsID=tls_id)
        return all_tls_time_state

    def get_current_phase(self, tlsID):
        return self.traci_con.trafficlight.getPhase(tlsID=tlsID)

    def get_all_tls_current_phase(self, list_intersection_id):
        """输入红绿灯的id，返回字典，包含每个信号灯当前的相位"""
        all_tls_phase_state = {}
        for tls_id in list_intersection_id:
            all_tls_phase_state[tls_id] = self.get_current_phase(tlsID=tls_id)
        return all_tls_phase_state

    def set_phase(self, tlsID, phase_index):
        self.traci_con.trafficlight.setPhase(tlsID=tlsID, index=phase_index)

    def set_time(self, tlsID, phase_time=3):
        self.traci_con.trafficlight.setPhaseDuration(tlsID=tlsID, phaseDuration=phase_time)

    def get_lane_tls_time(self, lane_list):
        lane_tls_time = []
        for lane in lane_list:
            if lane not in self.lane_controlled_by_tls:
                lane_tls_time.append(-1)
            else:
                lane_tls_time.append(self.tls_time_state[self.lane_controlled_by_tls[lane]])
        return np.array(lane_tls_time)

    def get_lane_tls_state(self, lane_list):
        lane_tls_state = []
        for lane in lane_list:
            if lane not in self.lane_controlled_by_tls:
                lane_tls_state.append(-1)
            else:
                lane_tls_state.append(self.tls_phase_state[self.lane_controlled_by_tls[lane]] // 2)
        return np.array(lane_tls_state)

    def get_lane_vehicle_total(self, lane_list):
        lane_vehicle_num_total = []
        for lane in lane_list:
            lane_vehicle_num_total.append(round(self.traci_con.lane.getLastStepVehicleNumber(lane), 2))
        return np.array(lane_vehicle_num_total)

    def get_lane_vehicle_halting(self, lane_list):
        lane_vehicle_num_halting = []
        for lane in lane_list:
            lane_vehicle_num_halting.append(round(self.traci_con.lane.getLastStepHaltingNumber(lane), 2))
        return np.array(lane_vehicle_num_halting)

    def get_lane_vehicle_halting_length(self, lane_list):
        lane_vehicle_halting_length = []
        for lane in lane_list:
            lane_vehicle_halting_length.append(round(self.traci_con.lane.getLastStepLength(lane), 2))
        return np.array(lane_vehicle_halting_length)

    def get_lane_vehicle_waiting_time(self, lane_list):
        lane_vehicle_waiting_time = []
        for lane in lane_list:
            lane_vehicle_waiting_time.append(round(self.traci_con.lane.getWaitingTime(lane), 2))
        return np.array(lane_vehicle_waiting_time)

    def get_lane_vehicle_mean_speed(self, lane_list):
        lane_vehicle_mean_speed = []
        for lane in lane_list:
            lane_vehicle_mean_speed.append(round(self.traci_con.lane.getLastStepMeanSpeed(lane), 2))
        return np.array(lane_vehicle_mean_speed)

    def get_lane_vehicle_occupancy(self, lane_list):
        lane_vehicle_occupancy = []
        for lane in lane_list:
            lane_vehicle_occupancy.append(round(self.traci_con.lane.getLastStepOccupancy(lane), 2))
        return np.array(lane_vehicle_occupancy)


    def get_observation_gat_net_state(self, graph):

        """
        输入图
        更新图中节点的特征
        从SUMO仿真中获取实时的车道信息并更新节点特征
        """
        for node, data in graph.nodes(data=True):
            features_list = []
            for lane_index, lane_id in enumerate(data['lanes_id']):
                lane_features = [round(self.traci_con.lane.getLastStepVehicleNumber(lane_id), 2),
                                 round(self.traci_con.lane.getLastStepHaltingNumber(lane_id), 2),
                                 round(self.traci_con.lane.getLastStepLength(lane_id), 2),
                                 round(self.traci_con.lane.getWaitingTime(lane_id), 2),
                                 round(self.traci_con.lane.getLastStepMeanSpeed(lane_id), 2),
                                 round(self.traci_con.lane.getLastStepOccupancy(lane_id), 2)]
                if lane_id not in self.lane_controlled_by_tls:
                    lane_features.append(-1)
                else:
                    lane_features.append(self.tls_phase_state[self.lane_controlled_by_tls[lane_id]] // 2)
                if lane_id not in self.lane_controlled_by_tls:
                    lane_features.append(-1)
                else:
                    lane_features.append(self.tls_time_state[self.lane_controlled_by_tls[lane_id]])
                lane_features.append(data['length'])
                lane_features.append(data['lanes_index'][lane_index])

                features_list.append(lane_features)

            graph.nodes[node]['features'] = features_list
        """
        获取GAT模型所需的动态输入数据
        :return: 包含节点特征矩阵和边索引的PyTorch Geometric数据对象
        """
        node_features_dict = {}
        node_features = []
        edge_index = []

        for node, data in graph.nodes(data=True):
            features = []
            for lane_features in data['features']:
                features.extend(lane_features)
            while len(features) < self.net_info.max_lanes * self.net_obs_lanes_state_dim:
                features.append(0)
            node_features_dict[node] = features

        for number in range(len(self.net_info.node_index)):
            for node_index in self.net_info.node_index:
                if self.net_info.node_index[node_index] == number:
                    node_features.append(node_features_dict[node_index])

        for src, dst in graph.edges():
            edge_index.append([self.net_info.node_index[src], self.net_info.node_index[dst]])
        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.int).t().contiguous()
        data = Data(x=x, edge_index=edge_index)
        return data

    def get_reward_lane_waiting_time(self, lane_list):
        lane_vehicle_waiting_time = []
        for lane in lane_list:
            lane_vehicle_waiting_time.append(round(self.traci_con.lane.getWaitingTime(lane), 2))
        wait_time_total = 0
        for lane_wait in lane_vehicle_waiting_time:
            wait_time_total += lane_wait
        return -wait_time_total

    def get_reward_lane_waiting_time_tls(self, tlsID, net_info):
        lane_vehicle_waiting_time = []
        for direction in net_info.intersection_entering_approach_lanes_dic[tlsID]:
            for lane in net_info.intersection_entering_approach_lanes_dic[tlsID][direction]:
                lane_vehicle_waiting_time.append(round(self.traci_con.lane.getWaitingTime(lane), 2))
        waiting_time_vehicle = 0
        for waiting_time in lane_vehicle_waiting_time:
            waiting_time_vehicle += waiting_time
        return -waiting_time_vehicle

    def get_reward_lane_travel_time_tls(self, tlsID, net_info):
        lane_vehicle_travel_time = []
        for direction in net_info.intersection_entering_approach_lanes_dic[tlsID]:
            for lane in net_info.intersection_entering_approach_lanes_dic[tlsID][direction]:
                lane_vehicle_travel_time.append(round(self.traci_con.lane.getTraveltime(lane), 2))
        travel_time_vehicle = 0
        for travel_time in lane_vehicle_travel_time:
            travel_time_vehicle += travel_time
        return -travel_time_vehicle

    def get_reward_lane_queue_vehicle(self, lane_list):
        lane_vehicle_queue_vehicle = []
        for lane in lane_list:
            lane_vehicle_queue_vehicle.append(round(self.traci_con.lane.getLastStepHaltingNumber(lane), 2))
        wait_queue_vehicle = 0
        for queue_vehicle in lane_vehicle_queue_vehicle:
            wait_queue_vehicle += queue_vehicle
        return -wait_queue_vehicle

    def get_reward_lane_queue_vehicle_tls(self, tlsID, net_info):
        lane_vehicle_queue_vehicle_tls = []
        for direction in net_info.intersection_entering_approach_lanes_dic[tlsID]:
            for lane in net_info.intersection_entering_approach_lanes_dic[tlsID][direction]:
                lane_vehicle_queue_vehicle_tls.append(round(self.traci_con.lane.getLastStepHaltingNumber(lane), 2))
        wait_queue_vehicle = 0
        for queue_vehicle in lane_vehicle_queue_vehicle_tls:
            wait_queue_vehicle += queue_vehicle
        return -wait_queue_vehicle

    def switch_phase_action(self, intersection_class, net_info, tlsID, action, state, all_yellow_time, green_time,
                            continue_green_time, GAT_net_agent):
        """
        输入路口的智能体、路网信息、路口ID（红绿灯ID）、上一个动作、上一个状态、黄灯时间、绿灯时间
        输出该智能体选择的下一个相位
        返回True表示是绿灯末尾，有经验变化
        返回False表示是黄灯末尾，没有经验，或者相位出错
        """
        action_phase = action * 2
        if self.tls_phase_state[tlsID] % 2 == 0:
            next_state = self.observation_gat_net_state
            reward = self.get_reward_lane_queue_vehicle_tls(tlsID=tlsID, net_info=net_info)
            pre_action_phase = GAT_net_agent.take_action(next_state, tls_id=tlsID) * 2

            if pre_action_phase != action_phase:
                intersection_class.next_action_green = pre_action_phase
                self.set_phase(tlsID=tlsID, phase_index=(action_phase + 1))
                self.set_time(tlsID=tlsID, phase_time=all_yellow_time)
                intersection_class.SARS = (state, action, reward, next_state, False, tlsID)
                intersection_class.state = next_state
                intersection_class.action = pre_action_phase // 2
                return True

            else:
                self.set_phase(tlsID=tlsID, phase_index=pre_action_phase)
                self.set_time(tlsID=tlsID, phase_time=continue_green_time)

                intersection_class.SARS = (state, action, reward, next_state, False, tlsID)
                intersection_class.state = next_state
                intersection_class.action = pre_action_phase // 2
                return True

        elif self.tls_phase_state[tlsID] % 2 == 1:
            self.set_phase(tlsID=tlsID, phase_index=intersection_class.next_action_green)
            self.set_time(tlsID=tlsID, phase_time=green_time)

            return False
        else:
            print(f"！错误！：路口{tlsID}，相位设置可能存在问题！")
            return False

    def test_agent_switch_phase_action(self, intersection_class, net_info, tlsID, action, state, all_yellow_time, green_time,
                                       continue_green_time, GAT_net_agent, test_agent=True):
        """
        用于测试agent时选择动作的函数
        输入路口的智能体、路网信息、路口ID（红绿灯ID）、上一个动作、上一个状态、黄灯时间、绿灯时间
        输出该智能体选择的下一个相位
        返回True表示是绿灯末尾，有经验变化
        返回False表示是黄灯末尾，没有经验，或者相位出错
        """
        action_phase = action * 2
        if self.tls_phase_state[tlsID] % 2 == 0:
            next_state = self.observation_gat_net_state
            reward = self.get_reward_lane_queue_vehicle_tls(tlsID=tlsID, net_info=net_info)
            pre_action_phase = GAT_net_agent.take_action(next_state, tlsID, test_agent) * 2

            if pre_action_phase != action_phase:
                intersection_class.next_action_green = pre_action_phase
                self.set_phase(tlsID=tlsID, phase_index=(action_phase + 1))
                self.set_time(tlsID=tlsID, phase_time=all_yellow_time)

                intersection_class.state = next_state
                intersection_class.action = pre_action_phase // 2
                return reward, pre_action_phase // 2

            else:
                self.set_phase(tlsID=tlsID, phase_index=pre_action_phase)
                self.set_time(tlsID=tlsID, phase_time=continue_green_time)
                intersection_class.state = next_state
                intersection_class.action = pre_action_phase // 2
                return reward, pre_action_phase // 2

        elif self.tls_phase_state[tlsID] % 2 == 1:
            self.set_phase(tlsID=tlsID, phase_index=intersection_class.next_action_green)
            self.set_time(tlsID=tlsID, phase_time=green_time)

            return None, None
        else:
            print(f"！错误！：路口{tlsID}，相位设置可能存在问题！")
            return False

    def close(self):
        self.traci_con.close()

    def obtain_att_test_statistics(self):
        if self.test_agent:
            tree = ET.parse(os.path.join(self.log_path_sumo, "statistic", f"{self.sim_round}.xml"))
            root = tree.getroot()

            att_result = []
            for child in root.findall('.//vehicleTripStatistics'):
                duration = child.get('duration')
                count = child.get('count')
                speed = child.get('speed')
                timeLoss = child.get('timeLoss')
                att_result.append(duration)
                att_result.append(count)
                att_result.append(speed)
                att_result.append(timeLoss)
            return att_result
        else:
            print(f"--这是train轮次，不获取指标信息！--")
            return "--这是train轮次，不获取指标信息！--"


if __name__ == '__main__':
    pass
