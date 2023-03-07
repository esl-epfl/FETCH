import numpy as np
import pickle
import os
from utils.params import dataset_parameter, EEG_channels
import pandas as pd
import json
from tqdm import  tqdm

SEG_N = 1024


def get_data(pretrain_mode=False, dataset='TUSZ', vision_based=False, mode_specific=None):
    def set_labels(x):

        for seizure_num in range(len(x['onsets'])):
            if x['onsets'][seizure_num][0] == x['offsets'][seizure_num][0]:
                continue
            if vision_based:
                start = max(x['onsets'][seizure_num][0] // x['fs_FP1'] // 12, 0)
                end = int(np.ceil(x['offsets'][seizure_num][0] // x['fs_FP1'] / 12))
            else:
                start = max(x['onsets'][seizure_num][0] // x['fs_FP1'] - 3, 0)
                end = x['offsets'][seizure_num][0] // x['fs_FP1'] + 4

            x['labels'][start:end] = 1

    if dataset == "epilepsiae":
        rootdir = '../input/Epilepsiae_total' if pretrain_mode else '../input/Epilepsiae_info'
    elif dataset == "TUSZ":
        rootdir = "../input/TUSZ_zc"
    elif dataset == "TUSZ_STFT":
        rootdir = "../input/TUSZ_STFT"
    else:
        raise Exception("Dataset unknown!")
    df = pd.read_csv(rootdir + '/{}_labels.csv'.format(dataset))
    if not pretrain_mode:
        if vision_based:
            df['labels_len'] = (df['length'] // df['fs_FP1']) // 12
        else:
            df['labels_len'] = (df['length'] // df['fs_FP1']) - 4
        df['labels'] = df['labels_len'].apply(lambda x: np.zeros(x, dtype=np.int))
        df['onsets'] = df['onsets'].apply(lambda x: json.loads(x.replace('\n', ',')))
        df['offsets'] = df['offsets'].apply(lambda x: json.loads(x.replace('\n', ',')))
        df.apply(set_labels, axis=1)

    train_ns = df[(df['mode'] == 'trainSet') & (df['label'] == 0.0)]
    drop_indices = np.random.choice(train_ns.index, 3650, replace=False)
    df_subset = df.drop(drop_indices)
    df = df_subset.sort_values(by='patient')

    # df = df.groupby('patient').head(1)

    if dataset == "epilepsiae":
        test_set = [x for x in df['file_name'].tolist() if x.startswith('Patient_1_')]
        excluded = []
        if pretrain_mode:
            validation_set = [x for x in (df.groupby('patient').head(1))['file_name'].tolist() if
                              (x.startswith('Patient_2_') or x.startswith('Patient_3_'))]
        else:
            validation_set = [x for x in df['file_name'].tolist() if x.startswith('Patient_2_')]
    else:  # dataset = TUSZ, TUSZ_STFT
        if pretrain_mode:
            validation_set = [x for x in df['file_name'].tolist() if
                              ('06175' in x or '01543' in x)]
        else:
            validation_set = [x for x in df[df['mode'] == 'devSet']['file_name'].tolist()]
        test_set = [x for x in df[df['mode'] == 'testSet']['file_name'].tolist()]
        excluded = [x for x in df['file_name'].tolist() if x.startswith('dev') and (not x in test_set)]

    print('test Set', len(test_set))
    training_set = [x for x in df['file_name'].tolist() if
                    (not x in test_set) and (not x in validation_set) and (not x in excluded)]
    df_file_name = df.set_index('file_name')

    pretrain_scratch_mode = "pretrain" if pretrain_mode else "scratch"
    train_len = dataset_parameter[dataset][pretrain_scratch_mode]["train_len"]
    val_len = dataset_parameter[dataset][pretrain_scratch_mode]["val_len"]
    test_len = dataset_parameter[dataset][pretrain_scratch_mode]["test_len"]

    mode_list = ['train', 'val', 'test'] if mode_specific is None else [mode_specific]
    mode_len = {'train': train_len, 'val': val_len, 'test': test_len}

    if vision_based:
        if mode_specific is None:
            X = {'train': np.zeros((train_len, 100*len(EEG_channels), 14)),
                 'test': np.zeros((test_len, 100*len(EEG_channels), 14)),
                 'val': np.zeros((val_len, 100*len(EEG_channels), 14))}
        else:
            X = {mode_specific: np.zeros((mode_len[mode_specific], 100*len(EEG_channels), 14))}
    else:
        band_feature_size = dataset_parameter[dataset][pretrain_scratch_mode]["band_feature_size"]
        zc_feature_size = dataset_parameter[dataset][pretrain_scratch_mode]["zc_feature_size"]
        X = {'train': np.zeros((train_len, band_feature_size + zc_feature_size)),
             'test': np.zeros((test_len, band_feature_size + zc_feature_size)),
             'val': np.zeros((val_len, band_feature_size + zc_feature_size))}
    labels = {'train': np.zeros((0, 1)), 'test': np.zeros((0, 1)), 'val': np.zeros((0, 1))}
    valid_labels = {'train': np.zeros(0, dtype=np.int), 'test': np.zeros(0, dtype=np.int),
                    'val': np.zeros(0, dtype=np.int)}
    minute_labels = {'train': np.zeros(0, dtype=np.int), 'test': np.zeros(0, dtype=np.int),
                     'val': np.zeros(0, dtype=np.int)}
    sample_time = {'train': np.zeros(0, dtype=np.int), 'test': np.zeros(0, dtype=np.int),
                   'val': np.zeros(0, dtype=np.int)}
    total_dataset = {'train': training_set, 'test': test_set, 'val': validation_set}
    pat_start_end = {'train': {new_list: [] for new_list in range(30)},
                     'test': {new_list: [] for new_list in range(30)},
                     'val': {new_list: [] for new_list in range(30)}}
    feature_name= ['meanAmpl', 'LL', 'p_delta', 'p_theta', 'p_alfa', 'p_beta', 'ZC standard', '16', '32', '64', '128', '256']
    for mode in mode_list:
        total_len = 0
        start_index = 0
        for t_file in tqdm(total_dataset[mode], desc="Loading {} ".format(mode)):
            with open(rootdir + '/{}_STFT.pickle'.format(t_file.split('/')[-1]), 'rb') as pickle_file:
                stft_data = pickle.load(pickle_file)
                stft_data_reshaped = np.reshape(stft_data[:, :, :, :], newshape=(-1, 2100, 14))
                X[mode][start_index:start_index + stft_data.shape[0], :] = stft_data_reshaped

                total_len += stft_data.shape[0]
                start_index += stft_data.shape[0]
                y = np.zeros(stft_data.shape[0]) if pretrain_mode else df_file_name.loc[t_file, 'labels']
                if stft_data.shape[0] != y.shape[0]:
                    print("Error in shape of {}: {} and {}\n".format(t_file, stft_data.shape, y.shape))
            # pat_num = t_file.split('/')[-1].split('_')[0] if dataset == "TUSZ" else int(t_file.split('_')[1]) - 1

            # valid_start = labels[mode].shape[0] + 56
            # # if mode == 'test': valid_start = valid_start - 4
            # valid_end = labels[mode].shape[0] + y.shape[0]
            # minute_labels[mode] = np.concatenate(
            #     (minute_labels[mode], np.arange(start=valid_start, stop=valid_end, step=60)))
            # valid_labels[mode] = np.concatenate((valid_labels[mode], np.arange(start=valid_start, stop=valid_end)))
            #
            # if pat_num not in pat_start_end[mode]:
            #     pat_start_end[mode][pat_num] = []
            # pat_start_end[mode][pat_num].append((labels[mode].shape[0], labels[mode].shape[0] + y.shape[0]))
            #
            # sample_time[mode] = np.concatenate((sample_time[mode], np.arange(start=0, stop=y.shape[0])))
            labels[mode] = np.concatenate((labels[mode], np.expand_dims(y, axis=1)))

        print("total_len ", mode, total_len)
        X[mode] = X[mode][:total_len]
    # print(X["train"].shape)
    # print(X["val"].shape)
    # print(X["test"].shape)

    # mean_train = np.mean(X["train"], axis=0)[None, ...]
    # std_train = np.std(X["train"], axis=0)[None, ...]
    # X["train"] = (X["train"] - mean_train) / std_train
    # X["val"] = (X["val"] - mean_train) / std_train
    # X["test"] = (X["test"] - mean_train) / std_train

    # print(valid_labels["test"].shape)
    return X, labels, minute_labels, pat_start_end, sample_time, valid_labels


def dataset_training(mode, test_patient, max_len=899):
    X_total = []
    y_total = []
    seizure_len = []
    all_filenames = get_all_filenames(False)
    for filename in all_filenames[mode][test_patient]:
        file_pat = int(filename.split('/')[-1][3:5])

        with open(filename, "rb") as pickle_file:
            data = pickle.load(pickle_file)
            x = np.array(data["X"])
            y = np.array(data["y"])
            if np.sum(y) == 0:
                continue
            seizure_len.append(np.sum(y))
            y = np.expand_dims(y, -1)
            if x.shape[0] == max_len:
                x_selected = x
                y_selected = y
            elif x.shape[0] < max_len:
                diff = max_len - x.shape[0]
                x = np.pad(x, pad_width=[(0, diff), (0, 0), (0, 0)], constant_values=0)
                x_selected = x
                y = np.pad(y, pad_width=[(0, diff), (0, 0)], constant_values=0)
                y_selected = y
            elif x.shape[0] > max_len:
                for start in range(0, x.shape[0] - max_len, (max_len//4)):
                    end = start + max_len
                    if np.sum(y[start:end]) == 0:
                        continue
                    x_selected = x[start:end, :, :]
                    y_selected = y[start:end]

            X_total.append(x_selected)
            y_total.append(y_selected)

    return np.asarray(X_total), np.asarray(y_total)


def get_all_filenames(entire_dataset=False):
    all_filenames = {'train': {}, 'valid': {}}
    for mode in 'train', 'valid':
        dirname = "../input/chbmit/{}".format(SEG_N, mode)
        if entire_dataset:
            filenames = ["{}/{}".format(dirname, x) for x in os.listdir(dirname)]
            all_filenames[mode]['-1'] = filenames
            continue
        else:
            for test_patient in range(1, 25):
                filenames = ["{}/{}".format(dirname, x) for x in os.listdir(dirname) if not
                x.startswith("chb{:02d}".format(test_patient))]
                all_filenames[mode][str(test_patient)] = filenames
    return all_filenames


def get_eglass_features(test_patient, seq_len = 899):
    features_dict = pickle.load(open("../input/chbmit/Features_Eglass_chb.pickle", "rb"))
    train_files = [x for x in features_dict.keys() if not x.startswith("chb{:02d}".format(test_patient))]
    # test_files = [x for x in features_dict.keys() if x.startswith("chb{:02d}".format(test_patient))]
    train_data = np.zeros((0, seq_len, 108))
    train_label = np.zeros((0, seq_len))
    for pat_file in train_files:
        if np.sum( features_dict[pat_file]['y']) == 0:
            continue
        x = features_dict[pat_file]['X']
        y = features_dict[pat_file]['y']
        if x.shape[0] == seq_len:
            x_selected = x
            y_selected = y
        elif x.shape[0] < seq_len:
            diff = seq_len - x.shape[0]
            x = np.pad(x, pad_width=[(0, diff), (0, 0)], constant_values=0)
            x_selected = x
            y = np.pad(y, pad_width=[(0, diff)], constant_values=0)
            y_selected = y
        elif x.shape[0] > seq_len:
            for start in range(0, x.shape[0] - seq_len, (seq_len // 4)):
                end = start + seq_len
                if np.sum(y[start:end]) == 0:
                    continue
                x_selected = x[start:end, :]
                y_selected = y[start:end]
        train_data = np.concatenate((train_data, np.expand_dims(x_selected, 0)), axis=0)
        train_label = np.concatenate((train_label, np.expand_dims(y_selected,0)), axis=0)

    print("Train : {}".format(train_data.shape))
    print("Train : {}".format(train_label.shape))
    np.nan_to_num(train_data, copy=False)
    return train_data, train_label
