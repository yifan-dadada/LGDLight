import math
import networkx as nx
import os
import sys
import json

import sumolib
import torch
from torch_geometric.data import Data


class NetInfo(object):
    def __init__(self, net_path, rl_data_txt_path):
        self.net = sumolib.net.readNet(net_path, withPrograms=True)
        self.list_approaches = ["W", "S", "E", "N"]

        self.list_intersection_id = self._get_list_intersection_id()
        print(f"路口数：{len(self.list_intersection_id)} ---", self.list_intersection_id)

        self.dic_entering_approach_to_edge = self._get_intersection_approach_to_edge(in_or_out="entering")
        self.dic_exiting_approach_to_edge = self._get_intersection_approach_to_edge(in_or_out="exiting")

        self.intersection_entering_approach_lanes_dic = self._get_intersection_entering_approach_lanes_dic()
        self.intersection_entering_lanes_length_dic = self._get_intersection_entering_lanes_length_dic()
        self.intersection_exiting_approach_lanes_dic = self._get_intersection_exiting_approach_lanes_dic()
        self.intersection_exiting_lanes_length_dic = self._get_intersection_exiting_lanes_length_dic()

        self.phase_num_dic, self.green_phase_map, self.yellow_phase_map = self._get_green_yellow_phase_map()

        self.lane_list = self._get_lane_list()
        self.line_graph, self.node_index = self._create_line_graph()
        print("路网转为线图后：", self.line_graph)
        print("节点的index：", self.node_index)
        self.max_lanes = max(len(edge.getLanes()) for edge in self.net.getEdges())

        self.tls_edges = self._get_tls_edges()
        print("路口的周围的道路（tls_edges）：", self.tls_edges)


        with open(rl_data_txt_path, "w") as f:
            f.write("list_intersection_id:" + str(self.list_intersection_id))
            f.write("\ndic_entering_approach_to_edge =" + str(self.dic_entering_approach_to_edge))
            f.write("\ndic_exiting_approach_to_edge =" + str(self.dic_exiting_approach_to_edge))
            f.write("\nintersection_entering_approach_lanes_dic =" + str(self.intersection_entering_approach_lanes_dic))
            f.write("\nintersection_entering_lanes_length_dic =" + str(self.intersection_entering_lanes_length_dic))
            f.write("\nintersection_exiting_approach_lanes_dic =" + str(self.intersection_exiting_approach_lanes_dic))
            f.write("\nintersection_exiting_lanes_length_dic =" + str(self.intersection_exiting_lanes_length_dic))
            f.write("\nphase_num_dic =" + str(self.phase_num_dic))
            f.write("\ngreen_phase_map =" + str(self.green_phase_map))
            f.write("\nyellow_phase_map =" + str(self.yellow_phase_map))
            f.write("\nlane_list =" + str(self.lane_list))
            f.write("\nself.tls_edges =" + str(self.tls_edges))


    def _get_list_intersection_id(self):
        """
        红绿灯路口的id
        :return: 返回列表，内容为str
        """
        list_intersection_id = [i.getID() for i in self.net.getTrafficLights()]
        return list_intersection_id

    def _get_approach(self, from_node, to_node):
        """
        判断两个点之间的位置关系，从而得到道路的方向。getCoord是sumolib的函数
        """
        dx = from_node.getCoord()[0] - to_node.getCoord()[0]
        dy = from_node.getCoord()[1] - to_node.getCoord()[1]
        angle = math.atan2(dy, dx) * 180.0 / math.pi
        if -45 <= angle < 45:
            return "E"
        elif 45 <= angle < 135:
            return "N"
        elif -135 <= angle < -45:
            return "S"
        else:
            return "W"

    def _get_intersection_approach_to_edge(self, in_or_out):
        intersection_approach_to_edge = {}

        for node_id in self.list_intersection_id:
            intersection_approach_to_edge[node_id] = {approach: [] for approach in self.list_approaches}

        for edge in self.net.getEdges():
            to_node = edge.getToNode()
            from_node = edge.getFromNode()
            to_node_id = to_node.getID()
            from_node_id = from_node.getID()

            if to_node_id in self.list_intersection_id or from_node_id in self.list_intersection_id:
                if in_or_out == "entering":
                    if to_node_id in self.list_intersection_id:
                        approach = self._get_approach(from_node, to_node)
                        if approach is not None:
                            intersection_approach_to_edge[to_node_id][approach].append(edge.getID())
                elif in_or_out == "exiting":
                    if from_node_id in self.list_intersection_id:
                        approach = self._get_approach(to_node, from_node)
                        if approach is not None:
                            intersection_approach_to_edge[from_node_id][approach].append(edge.getID())
                else:
                    print("Error: 请输入要判断进入还是离开的方向！！！")

        for node_id in intersection_approach_to_edge:
            for approach in intersection_approach_to_edge[node_id]:
                if len(intersection_approach_to_edge[node_id][approach]) == 1:
                    intersection_approach_to_edge[node_id][approach] = intersection_approach_to_edge[node_id][approach][0]
                elif len(intersection_approach_to_edge[node_id][approach]) == 0:
                    intersection_approach_to_edge[node_id][approach] = None

        return intersection_approach_to_edge

    def _get_intersection_entering_approach_lanes_dic(self):
        """
        返回路口进车道id
        :return: 返回字典，没有车道的方向为空列表[]
        """
        enter_dict = {}
        for intersection in self.list_intersection_id:
            enter_dict[intersection] = {'W': [], 'S': [], 'E': [], 'N': []}

        for tls_id in self.list_intersection_id:
            for direction in self.dic_entering_approach_to_edge[tls_id]:
                edge_id = self.dic_entering_approach_to_edge[tls_id][direction]
                if edge_id:
                    edge = self.net.getEdge(edge_id)
                    for lane in edge.getLanes():
                        enter_dict[tls_id][direction].append(lane.getID())

        # print("intersection_entering_approach_lanes_dic", enter_dict)
        return enter_dict

    def _get_intersection_entering_lanes_length_dic(self):
        """
        返回路口进入车道的长度
        :return: 返回字典，没有车道的方向为空字典
        """
        enter_length_dict = {}
        for intersection in self.list_intersection_id:
            enter_length_dict[intersection] = {'W': {}, 'S': {}, 'E': {}, 'N': {}}

        for tls_id in self.list_intersection_id:
            for direction in self.dic_entering_approach_to_edge[tls_id]:
                edge_id = self.dic_entering_approach_to_edge[tls_id][direction]
                if edge_id:
                    edge = self.net.getEdge(edge_id)
                    for lane in edge.getLanes():
                        enter_length_dict[tls_id][direction][lane.getID()] = lane.getLength()
        return enter_length_dict

    def _get_intersection_exiting_approach_lanes_dic(self):
        """
        返回路口出车道id
        :return: 返回字典，没有车道的方向为空列表[]
        """
        exit_dict = {}
        for intersection in self.list_intersection_id:
            exit_dict[intersection] = {'W': [], 'S': [], 'E': [], 'N': []}

        for tls_id in self.list_intersection_id:
            for direction in self.dic_exiting_approach_to_edge[tls_id]:
                edge_id = self.dic_exiting_approach_to_edge[tls_id][direction]
                if edge_id:
                    edge = self.net.getEdge(edge_id)
                    for lane in edge.getLanes():
                        exit_dict[tls_id][direction].append(lane.getID())

        # print("intersection_exiting_approach_lanes_dic", exit_dict)
        return exit_dict

    def _get_intersection_exiting_lanes_length_dic(self):
        """
        返回路口驶出车道的长度
        :return: 返回字典，没有车道的方向为空字典
        """
        exit_length_dict = {}
        for intersection in self.list_intersection_id:
            exit_length_dict[intersection] = {'W': {}, 'S': {}, 'E': {}, 'N': {}}

        for tls_id in self.list_intersection_id:
            for direction in self.dic_exiting_approach_to_edge[tls_id]:
                edge_id = self.dic_exiting_approach_to_edge[tls_id][direction]
                if edge_id:
                    edge = self.net.getEdge(edge_id)
                    for lane in edge.getLanes():
                        exit_length_dict[tls_id][direction][lane.getID()] = lane.getLength()
        return exit_length_dict

    def _get_green_yellow_phase_map(self):
        """
        获取每个路口的绿灯相位数量、绿灯相位的值、黄灯相位的值
        :return: phase_num_dic, green_phase_map, yellow_phase_map
        """
        green_dict = {}
        yellow_dict = {}
        phase_num_dic = {}

        for intersection in self.list_intersection_id:
            green_dict[intersection] = []
            yellow_dict[intersection] = []
            phase_num_dic[intersection] = 0

        for tls_id in green_dict:
            green_list = []
            yellow_list = []
            tls = self.net.getTLS(tls_id)
            program = tls.getPrograms()

            for i, phase in enumerate(list(program.values())[0].getPhases()):
                if i % 2 == 0:
                    green_list.append(phase.state)
                if i % 2 == 1:
                    yellow_list.append(phase.state)

            green_dict[tls_id] = green_list
            yellow_dict[tls_id] = yellow_list

        for key in green_dict:
            phase_num_dic[key] = len(green_dict[key])

        return phase_num_dic, green_dict, yellow_dict

    def _get_lane_list(self):
        lane_list = []
        edges = self.net.getEdges(withInternal=False)
        for edge in edges:
            lanes = edge.getLanes()
            for lane in lanes:
                lane_list.append(lane.getID())
        return lane_list

    def _create_line_graph(self):
        """
        创建SUMO路网的有向线图
        :return: NetworkX有向图和节点索引字典
        """
        G = nx.DiGraph()
        node_index = {}
        current_index = 0

        for edge in self.net.getEdges():
            edge_id = edge.getID()
            from_node = edge.getFromNode().getID()
            to_node = edge.getToNode().getID()

            lanes = edge.getLanes()
            lanes_id = [lane.getID() for lane in lanes]
            lanes_index = [lane.getIndex() for lane in lanes]

            G.add_node(edge_id, lanes_id=lanes_id, lanes_index=lanes_index, length=edge.getLength(), num_lanes=edge.getLaneNumber())
            node_index[edge_id] = current_index
            current_index += 1

            for successor in edge.getOutgoing():
                successor_id = successor.getID()
                G.add_edge(edge_id, successor_id, from_node=to_node, to_node=successor.getToNode().getID())

        return G, node_index

    def _get_tls_edges(self):
        """
        获取每个路口周围的道路索引，包括进入和离开的道路
        :return: 路口周围的道路字典，值是道路的索引列表
        """
        tls_edges = {}
        for tls_id in self.list_intersection_id:
            edges = []
            for approach, edge_id in self.dic_entering_approach_to_edge[tls_id].items():
                if edge_id:
                    edges.append(self.node_index[edge_id])
            for approach, edge_id in self.dic_exiting_approach_to_edge[tls_id].items():
                if edge_id:
                    edges.append(self.node_index[edge_id])

            tls_edges[tls_id] = edges
        return tls_edges

    def visualize_graph(self):
        """
        可视化有向线图
        """
        print(self.line_graph)
        print("Nodes and their attributes:")
        for node, attrs in self.line_graph.nodes(data=True):
            print(f"Node {node}: {attrs}")

        print("\nEdges and their attributes:")
        for u, v, attrs in self.line_graph.edges(data=True):
            print(f"Edge from {u} to {v}: {attrs}")

        print("\nBasic graph information:")
        print(f"Number of nodes: {self.line_graph.number_of_nodes()}")
        print(f"Number of edges: {self.line_graph.number_of_edges()}")

        print("\nNode degrees (in-degree and out-degree):")
        for node in self.line_graph.nodes():
            print(f"Node {node}: in-degree = {self.line_graph.in_degree(node)}, out-degree = {self.line_graph.out_degree(node)}")

        print("\nAdjacency list:")
        for line in nx.generate_adjlist(self.line_graph):
            print(line)

        print("\nDetailed adjacency information:")
        for node in self.line_graph.nodes():
            for neighbor in self.line_graph.neighbors(node):
                print(f"Node {node} has an edge to {neighbor} with attributes {self.line_graph.get_edge_data(node, neighbor)}")

    def get_gat_input(self):
        """
        获取GAT模型所需的输入数据
        :return: 包含节点特征矩阵和边索引的PyTorch Geometric数据对象
        """
        node_features = []
        edge_index = []

        for node, data in self.line_graph.nodes(data=True):
            node_features.append([0, 0, 0, 0, 0, 0, 0, 0, data['length'], data['num_lanes']])
        print(node_features)

        for src, dst in self.line_graph.edges():
            edge_index.append([self.node_index[src], self.node_index[dst]])

        x = torch.tensor(node_features, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        data = Data(x=x, edge_index=edge_index)
        print("初始化GAT输入图：", data)
        return data

    def update_node_features(self, env):
        """
        更新图中节点的特征
        从SUMO仿真中获取实时的车道信息并更新节点特征
        """
        for node in self.line_graph.nodes:
            edge = self.net.getEdge(node)
            lanes = edge.getLanes()
            features = []
            for lane in lanes:
                lane_id = lane.getID()
                features.append([
                    env.lane.getLastStepVehicleNumber(lane_id),
                    env.lane.getLastStepHaltingNumber(lane_id),
                    env.lane.getLength(lane_id),
                    env.lane.getWaitingTime(lane_id),
                    env.lane.getLastStepMeanSpeed(lane_id),
                    env.lane.getLastStepOccupancy(lane_id),
                ])
            self.line_graph.nodes[node]['features'] = features

    def get_gat_input_dynamic(self, env, num_attributes):        # num_attributes = 8  # 每条车道的属性数量
        """
        获取GAT模型所需的动态输入数据
        :return: 包含节点特征矩阵和边索引的PyTorch Geometric数据对象
        """
        self.update_node_features(env)

        node_features = []
        edge_index = []

        max_lanes = max(len(edge.getLanes()) for edge in self.net.getEdges())

        for node, data in self.line_graph.nodes(data=True):
            features = []
            for lane_features in data['features']:
                features.extend(lane_features)
            while len(features) < max_lanes * num_attributes:
                features.append(0)
            node_features.append(features)

        for src, dst in self.line_graph.edges():
            edge_index.append([self.node_index[src], self.node_index[dst]])

        x = torch.tensor(node_features, dtype=torch.float)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        data = Data(x=x, edge_index=edge_index)

        print("node_features:", len(node_features), node_features)
        print("edge_index2:", edge_index)
        print("x:", x)
        print("data:", data)

        return data


if __name__ == '__main__':
    net_info_class = NetInfo(net_path="data/jinkai_hanghai-15-40/net_1h-0.4-jinkai_hagnhai-15-40.net-RL.xml",
                             rl_data_txt_path="data/jinkai_hanghai-15-40/rl_data.txt")
