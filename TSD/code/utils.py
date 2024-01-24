import numpy as np
from sklearn.metrics import precision_recall_curve
import pandas as pd
import json
from channel_possibility import double_banana


def thresh_max_f1(y_true, y_prob):
    """
    Find the best threshold based on precision-recall curve to maximize F1-score.
    Binary classification only
    """
    if len(set(y_true)) > 2:
        raise NotImplementedError

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresh_filt = []
    fscore = []
    n_thresh = len(thresholds)
    for idx in range(n_thresh):
        curr_f1 = (2 * precision[idx] * recall[idx]) / \
            (precision[idx] + recall[idx])
        if not (np.isnan(curr_f1)):
            fscore.append(curr_f1)
            thresh_filt.append(thresholds[idx])
    # locate the index of the largest f score
    ix = np.argmax(np.array(fscore))
    best_thresh = thresh_filt[ix]
    return np.float(best_thresh)


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
    # print(df.sample(n=5))
    return df


def channel_list_to_node_set(x):
    node_set = set()
    edge_lists = [double_banana[a] for a in x]
    for node1, node2 in edge_lists:
        node_set.add(node1)
        node_set.add(node2)
    return len(node_set)


def get_feasible_ids_with_num_nodes(num_nodes):
    df = create_dataframe(20)
    df['number_nodes'] = df['channel_list'].apply(channel_list_to_node_set)
    df_num_nodes = df[df['number_nodes'] == num_nodes]
    return df_num_nodes['channel_id'].tolist()
