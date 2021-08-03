import numpy as np
import pickle
import os

SEG_N = 1024


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
