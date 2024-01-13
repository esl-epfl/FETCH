import numpy as np
import pandas as pd
import json
import networkx as nx
from channel_possibility import double_banana, EEG_electrodes


def load_json(num_channels):
    with open(f"../feasible_channels/feasible_{num_channels}edges.json", 'r') as json_file:
        selected_channels = json.load(json_file)

    # the json file is a list of lists, each list is a set of channels
    # example: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    return selected_channels


def create_dataframe(num_channels):
    """
    # return a dataframe with five columns:
    # 1. channel_id
    # 2. channel_list like [0, 3, 6]
    # 3. channel_set like {0, 3, 6}
    # 4. channel_mask like [1, 0, 0, 1, 0, 0, 1, 0, 0]
    # 5. num_channels_wearable :number of channels in wearable
    :param num_channels: number of channels in EEG set
    """
    selected_channels = load_json(num_channels)
    # selected_channels is a list of lists, each list is a set of channels
    # example: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # create a dataframe with the four columns
    df = pd.DataFrame(columns=['channel_list', 'channel_set', 'channel_id', 'channel_mask', 'num_channels_wearable'])
    df['channel_list'] = selected_channels
    df['channel_set'] = df['channel_list'].apply(lambda x: set(x))
    df['channel_id'] = df.index
    df['channel_mask'] = df['channel_list'].apply(lambda x: np.array([1 if i in x else 0 for i in range(num_channels)]))
    df['num_channels_wearable'] = df['channel_list'].apply(lambda x: len(x))
    print(df.sample(n=5))
    return df


def node_set_to_channel_set(node_set):
    """
    # return the channel_set of a node_set
    :param node_set: the set of nodes
    :return: the channel_set according to the node_set, the number of nodes in the final graph,
    and the updated node_set
    """

    # create nx_graph from double_banana. double_banana is a list of tuples like [('FP1', 'F7'), ('F7', 'T3'), ...]
    nx_graph = nx.Graph()
    nx_graph.add_edges_from(double_banana)

    # Remove the nodes that are not in the node_set
    # print("Extra nodes", [node for node in nx_graph.nodes if node not in node_set])
    nx_graph.remove_nodes_from([node for node in nx_graph.nodes if node not in node_set])

    # Remove isolated nodes
    # print("Isolated nodes", list(nx.isolates(nx_graph)))
    nx_graph.remove_nodes_from(list(nx.isolates(nx_graph)))

    # print("Updated graph edges", nx_graph.edges)
    # print("Updated graph nodes", nx_graph.nodes)

    # Create a set containing the index of nx_graph edges in double_banana
    # For example, if we have [('FP1', 'F7'), ('F7', 'T3'), ...] in nx_graph.edges
    # and [('FP1', 'F7'), ('F7', 'T3'), ...] in double_banana
    # then the set will be {0, 1, ...}
    # some times the order of nodes in a tuple is reversed, so we need to check both
    channel_set = set()
    for edge in nx_graph.edges:
        if edge in double_banana:
            channel_set.add(double_banana.index(edge))
        elif (edge[1], edge[0]) in double_banana:
            channel_set.add(double_banana.index((edge[1], edge[0])))

    updated_node_set = set(nx_graph.nodes)
    return channel_set, len(nx_graph.nodes), updated_node_set


def channel_id_to_channel_set(df, channel_id):
    """
    # return the channel_set of a channel_id
    :param channel_id: the id of the channel
    :param df: the dataframe
    """
    return df.iloc[channel_id, 1]


def channel_id_to_channel_mask(df, channel_id):
    """
    # return the channel_mask of a channel_id
    :param channel_id: the id of the channel
    :param df: the dataframe
    """
    return df.iloc[channel_id, 3]


def channel_set_to_channel_id(df, channel_set):
    """
    # return the channel_id of a channel_list
    :param channel_set: the list of channels
    :param df: the dataframe
    """
    if channel_set not in df['channel_set'].values:
        return -1
    return df[df['channel_set'] == channel_set].index[0]


def channel_mask_to_channel_id(df, channel_mask):
    """
    # return the channel_id of a channel_mask
    :param channel_mask: the mask of channels

    """
    print(channel_mask)
    print(df['channel_mask'].values[0])
    print(channel_mask == df['channel_mask'].values[0])
    return df[df['channel_mask'] == channel_mask].index[0]


def channel_mask_to_channel_list(df, channel_mask):
    """
    # return the channel_list of a channel_mask
    :param channel_mask: the mask of channels
    :param df: the dataframe
    """
    channel_id = channel_mask_to_channel_id(df, channel_mask)
    return df.iloc[channel_id, 0]


def channel_set_to_channel_mask(df, channel_set):
    """
    # return the channel_mask of a channel_list
    :param channel_set: the set of channels
    :param df: the dataframe
    """
    if channel_set not in df['channel_set'].values:
        return -1
    return df[df['channel_set'] == channel_set].iloc[0, 3]


# Channel Selection Methods
class SequentialForwardSelection:
    # This class is used to select channels in a sequential forward way
    # It starts with an empty node set and add one node at a time
    def __init__(self, num_channels):
        self.num_channels = num_channels
        self.df = create_dataframe(num_channels)
        self.node_set = set()

    def select(self):
        # select the next node
        # find the node with the highest score
        max_score = -1
        max_node_name = ''
        max_node_name2 = ''  # in case we have empty node_set
        pre_non_isolated_nodes = len(self.node_set)
        # if we have empty node_set, we cannot choose a single node
        # Therefore, we choose an edge with two nodes
        if pre_non_isolated_nodes == 0:
            for edge in double_banana:
                node_set = self.node_set.copy()
                node_set.add(edge[0])
                node_set.add(edge[1])
                channel_set, _, _ = node_set_to_channel_set(node_set)
                channel_id = channel_set_to_channel_id(self.df, channel_set)
                if channel_id == -1:
                    continue
                score = self.score(channel_set)
                if score > max_score:
                    max_score = score
                    max_node_name = edge[0]
                    max_node_name2 = edge[1]
            if max_node_name == '':
                return
            self.node_set.add(max_node_name)
            self.node_set.add(max_node_name2)

        else:
            for potential_node in EEG_electrodes:
                if potential_node not in self.node_set:
                    print("potential_node", potential_node)
                    node_set = self.node_set.copy()
                    node_set.add(potential_node)
                    channel_set, num_nodes, _ = node_set_to_channel_set(node_set)
                    if num_nodes != pre_non_isolated_nodes + 1:
                        continue

                    channel_id = channel_set_to_channel_id(self.df, channel_set)
                    if channel_id == -1:
                        print("channel_id == -1", channel_set)
                        continue
                    score = self.score(channel_set)  # TODO: train a model and get the final AUC score
                    if score > max_score:
                        max_score = score
                        max_node_name = potential_node
            if max_node_name == '':
                return
            self.node_set.add(max_node_name)

    def score(self, channel_set):
        """
        # return the score of the selected channels
        """
        return sum(channel_set)

    def get_node_set(self):
        """
        # return the node_set of the selected channels
        """
        return self.node_set

    def get_channel_set(self):
        """
        # return the channel_set of the selected channels
        """
        return node_set_to_channel_set(self.node_set)[0]


class sequentialBackwardSelection:
    # This class is similar to SequentialForwardSelection but in the backward steps
    # It starts with a full node set and remove one node at a time
    def __init__(self, num_channels):
        self.num_channels = num_channels
        self.df = create_dataframe(num_channels)
        self.node_set = set(EEG_electrodes)

    def select(self, num_nodes_to_remove):
        # select the next channel
        # find the node with the lowest score
        min_score = 100000
        min_node_name = ''
        min_node_name2 = ''  # in case we have to remove two nodes
        pre_non_isolated_nodes = len(self.node_set)
        for potential_node in EEG_electrodes:
            if potential_node in self.node_set:
                node_set = self.node_set.copy()
                node_set.remove(potential_node)
                channel_set, num_nodes, updated_node_set = node_set_to_channel_set(node_set)
                if num_nodes != pre_non_isolated_nodes - num_nodes_to_remove:
                    continue
                channel_id = channel_set_to_channel_id(self.df, channel_set)
                if channel_id == -1:
                    continue
                score = self.score(channel_set)
                if score < min_score:
                    min_score = score
                    min_node_name = potential_node
                    if num_nodes_to_remove == 2:  # in case that we have to remove two nodes (due to the edges)
                        # find the other node to remove by differentiating the self.node_set and node_set
                        nodes_to_remove = self.node_set - updated_node_set
                        # min_node_name2 is the node to remove after removing potential_node from nodes_to_remove
                        min_node_name2 = list(nodes_to_remove - {potential_node})[0]

        if min_node_name == '':
            return -1
        self.node_set.remove(min_node_name)
        if num_nodes_to_remove == 2:
            self.node_set.remove(min_node_name2)
        return min_node_name

    def score(self, channel_set):
        """
        # return the score of the selected channels
        """
        return sum(channel_set)

    def get_node_set(self):
        """
        # return the node_set of the selected channels
        """
        return self.node_set

    def get_channel_set(self):
        """
        # return the channel_list of the selected channels
        """
        return node_set_to_channel_set(self.node_set)[0]


# test function for SequentialForwardSelection
def test_SFS():
    num_channels = 20
    sfs = SequentialForwardSelection(num_channels)
    for i in range(len(EEG_electrodes) - 1):
        sfs.select()
        print(i, sfs.get_node_set())
        print(i, sfs.get_channel_set())


# test function for sequentialBackwardSelection
def test_SBS():
    num_channels = 20
    sbs = sequentialBackwardSelection(num_channels)
    num_nodes_to_remove = 1
    for i in range(len(EEG_electrodes) - 1):
        node_removed = sbs.select(num_nodes_to_remove)
        if node_removed == -1:
            num_nodes_to_remove += 1
        else:
            num_nodes_to_remove = 1
        print(i, sbs.get_node_set())


# function to test channel_mask_to_channel_list
def test_channel_mask_to_channel_list():
    num_channels = 20
    df = create_dataframe(num_channels)
    channel_mask = np.zeros(num_channels, dtype=int)
    channel_mask[19] = 1
    channel_mask[1] = 0
    print(channel_mask_to_channel_id(df, channel_mask))


def test_channel_id_to_channel_set():
    num_channels = 20
    df = create_dataframe(num_channels)
    channel_id = 1000
    print(channel_id_to_channel_set(df, channel_id))
    print(channel_id_to_channel_mask(df, channel_id))


# function to test channel_set_to_channel_id
def test_channel_set_to_channel_id():
    num_channels = 20
    df = create_dataframe(num_channels)
    channel_set = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}
    print(channel_set_to_channel_id(df, channel_set))


# test function for node_set_to_channel_set
def test_node_set_to_channel_set():
    num_channels = 20
    df = create_dataframe(num_channels)
    node_set = {'FP1', 'F7', 'T3', 'T5', 'O1', 'F3', 'C3', 'P3', 'Fz', 'Cz',
                'Pz', 'FP2', 'F8', 'T4', 'T6', 'O2', 'F4',
                'C4', 'P4', 'A2'}
    print(node_set_to_channel_set(node_set))


if __name__ == '__main__':
    test_SBS()
