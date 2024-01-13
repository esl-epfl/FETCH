import numpy as np
import pandas as pd
import json


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
    def __init__(self, num_channels):
        self.num_channels = num_channels
        self.df = create_dataframe(num_channels)
        self.channel_list = []
        self.channel_mask = np.zeros(num_channels, dtype=int)

    def select(self):
        # select the next channel
        # find the channel with the highest score
        max_score = -1
        max_channel_id = -1
        for i in range(self.num_channels):
            if i not in self.channel_list:
                channel_list = self.channel_list.copy()
                channel_list.append(i)
                channel_set = set(channel_list)
                channel_id = channel_set_to_channel_id(self.df, channel_set)
                # channel_mask = channel_set_to_channel_mask(self.df, channel_set)
                if channel_id == -1:
                    continue
                score = self.score(channel_set)  # TODO: train a model and get the final AUC score
                if score > max_score:
                    max_score = score
                    max_channel_id = i
        if max_channel_id == -1:
            return
        self.channel_list.append(max_channel_id)
        self.channel_mask[max_channel_id] = 1

    def score(self, channel_set):
        """
        # return the score of the selected channels
        """
        return sum(channel_set)

    def get_channel_list(self):
        """
        # return the channel_list of the selected channels
        """
        return self.channel_list

    def get_channel_mask(self):
        """
        # return the channel_mask of the selected channels
        """
        return self.channel_mask


class sequentialBackwardSelection:
    # This class is similar to SequentialForwardSelection but in the backward steps
    # So it starts with a full channels set and remove one channel at a time
    def __init__(self, num_channels):
        self.num_channels = num_channels
        self.df = create_dataframe(num_channels)
        self.channel_list = list(range(num_channels))
        self.channel_mask = np.ones(num_channels, dtype=int)

    def select(self):
        # select the next channel
        # find the channel with the lowest score
        min_score = 100000
        min_channel_id = -1
        for i in range(self.num_channels):
            if i in self.channel_list:
                channel_list = self.channel_list.copy()
                channel_list.remove(i)
                channel_set = set(channel_list)
                channel_id = channel_set_to_channel_id(self.df, channel_set)
                # channel_mask = channel_set_to_channel_mask(self.df, channel_set)
                if channel_id == -1:
                    continue
                score = self.score(channel_set)
                if score < min_score:
                    min_score = score
                    min_channel_id = i
        if min_channel_id == -1:
            return
        self.channel_list.remove(min_channel_id)
        self.channel_mask[min_channel_id] = 0

    def score(self, channel_set):
        """
        # return the score of the selected channels
        """
        return sum(channel_set)

    def get_channel_list(self):
        """
        # return the channel_list of the selected channels
        """
        return self.channel_list

    def get_channel_mask(self):
        """
        # return the channel_mask of the selected channels
        """
        return self.channel_mask


# test function for SequentialForwardSelection
def test_SFS():
    num_channels = 20
    sfs = SequentialForwardSelection(num_channels)
    for i in range(num_channels):
        sfs.select()
        print(i, sfs.get_channel_list())
        print(i, sfs.get_channel_mask())


# test function for sequentialBackwardSelection
def test_SBS():
    num_channels = 20
    sbs = sequentialBackwardSelection(num_channels)
    for i in range(num_channels):
        sbs.select()
        print(i, sbs.get_channel_list())
        print(i, sbs.get_channel_mask())


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
    channel_set = {0, 2}
    print(channel_set_to_channel_id(df, channel_set))


if __name__ == '__main__':
    test_SBS()
