import math

import numpy as np
import pandas as pd
import json
import pickle
import seaborn as sns
from utils.BioT import BioTransformer, Epilepsy60Dataset, ImbalancedDataSampler, EvaluateSampler, \
    PatientDiscriminatorDataset, PatientDiscriminatorEvaluationDataset
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.optim import SGD, Adam, AdamW
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from utils.BioT import SEQ_LEN, SEGMENT, ROI
from utils.params import dataset_parameter


def get_data(pretrain_mode=False, dataset='TUSZ'):
    def set_labels(x):

        for seizure_num in range(len(x['onsets'])):
            if x['onsets'][seizure_num][0] == x['offsets'][seizure_num][0]:
                continue
            start = max(x['onsets'][seizure_num][0] // x['fs_FP1'] - 3, 0)
            end = x['offsets'][seizure_num][0] // x['fs_FP1'] + 4
            x['labels'][start:end] = 1

    if dataset == "epilepsiae":
        rootdir = '../input/Epilepsiae_total' if pretrain_mode else '../input/Epilepsiae_info'
    elif dataset == "TUSZ":
        rootdir = "../input/TUSZ_zc"
    else:
        raise Exception("Dataset unknown!")
    df = pd.read_csv(rootdir + '/{}_labels.csv'.format(dataset))
    if not pretrain_mode:
        df['labels_len'] = (df['length'] // df['fs_FP1']) - 4
        df['labels'] = df['labels_len'].apply(lambda x: np.zeros(x, dtype=np.int))
        df['onsets'] = df['onsets'].apply(lambda x: json.loads(x.replace('\n', ',')))
        df['offsets'] = df['offsets'].apply(lambda x: json.loads(x.replace('\n', ',')))
        df.apply(set_labels, axis=1)

    df = df.sort_values(by='patient')
    # df = df.groupby('patient').head(1)

    if dataset == "epilepsiae":
        test_set = [x for x in df['file_name'].tolist() if x.startswith('Patient_1_')]
        excluded = []
        if pretrain_mode:
            validation_set = [x for x in (df.groupby('patient').head(1))['file_name'].tolist() if
                              (x.startswith('Patient_2_') or x.startswith('Patient_3_'))]
        else:
            validation_set = [x for x in df['file_name'].tolist() if x.startswith('Patient_2_')]
    else:  # dataset = TUSZ
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
    feature_size = dataset_parameter[dataset][pretrain_scratch_mode]["feature_size"]
    train_len = dataset_parameter[dataset][pretrain_scratch_mode]["train_len"]
    val_len = dataset_parameter[dataset][pretrain_scratch_mode]["val_len"]
    test_len = dataset_parameter[dataset][pretrain_scratch_mode]["test_len"]
    X = {'train': np.zeros((train_len, feature_size)),
         'test': np.zeros((test_len, feature_size)),
         'val': np.zeros((val_len, feature_size))}
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
    for mode in ['train', 'val', 'test']:
        total_len = 0
        start_index = 0
        for t_file in total_dataset[mode]:
            with open(rootdir + '/{}_zc.pickle'.format(t_file), 'rb') as pickle_file:
                # print(t_file)
                data = pickle.load(pickle_file)
                # if pretrain_mode and data.shape[0] < SEQ_LEN:  # very short files
                #     continue
                total_len += data.shape[0]
                X[mode][start_index:start_index + data.shape[0]] = data
                start_index += data.shape[0]
                y = np.zeros(data.shape[0]) if pretrain_mode else df_file_name.loc[t_file, 'labels']
                if data.shape[0] != y.shape[0]:
                    print("Error in shape of {}: {} and {}\n".format(t_file, data.shape, y.shape))
            pat_num = t_file.split('/')[-1].split('_')[0] if dataset == "TUSZ" else int(t_file.split('_')[1]) - 1

            valid_start = labels[mode].shape[0] + ROI - 1
            if mode == 'test': valid_start = valid_start - 4
            valid_end = labels[mode].shape[0] + y.shape[0]
            minute_labels[mode] = np.concatenate((minute_labels[mode], np.arange(start=valid_start, stop=valid_end, step=ROI)))
            valid_labels[mode] = np.concatenate((valid_labels[mode], np.arange(start=valid_start, stop=valid_end)))

            if pat_num not in pat_start_end[mode]:
                pat_start_end[mode][pat_num] = []
            pat_start_end[mode][pat_num].append((labels[mode].shape[0], labels[mode].shape[0] + y.shape[0]))

            sample_time[mode] = np.concatenate((sample_time[mode], np.arange(start=0, stop=y.shape[0])))
            labels[mode] = np.concatenate((labels[mode], np.expand_dims(y, axis=1)))

        print("total_len ", mode, total_len)
    print(X["train"].shape)
    print(X["val"].shape)
    print(X["test"].shape)
    mean_train = np.mean(X["train"])
    std_train = np.std(X["train"])
    X["train"] = (X["train"] - mean_train) / std_train
    X["val"] = (X["val"] - mean_train) / std_train
    X["test"] = (X["test"] - mean_train) / std_train
    print(valid_labels["test"].shape)
    return X, labels, minute_labels, pat_start_end, sample_time, valid_labels


def train(model, device, save_path: str, learning_rate: float = 1e-5):
    X, labels, valid_labels, _, sample_time, _ = get_data(pretrain_mode=False, dataset='TUSZ')

    # %%
    mode = "train"
    seizure_indices = np.where(labels[mode] == 1)[0]
    non_seizure_indices = np.where(labels[mode] == 0)[0]
    post_ictal_indices = []
    ictal_post_ictal_indices = []
    seizure_end_points = np.where(np.diff(labels[mode][:, 0]).astype(np.int) == -1)[0]
    for post_ictal in seizure_end_points:
        for post_time in range(1, SEQ_LEN):
            if post_ictal + post_time >= len(sample_time[mode]) or \
                    sample_time[mode][post_ictal + post_time] == 0 or \
                    labels[mode][post_ictal + post_time] == 1:
                break
            if post_time <= ROI:
                ictal_post_ictal_indices.append(post_ictal + post_time)
            else:
                post_ictal_indices.append(post_ictal + post_time)

    post_ictal_indices = np.array(post_ictal_indices)
    ictal_post_ictal_indices = np.array(ictal_post_ictal_indices)
    non_seizure_indices = np.setdiff1d(non_seizure_indices, post_ictal_indices, assume_unique=True)
    non_seizure_indices = np.setdiff1d(non_seizure_indices, ictal_post_ictal_indices, assume_unique=True)
    print("{}: {} seizures, {} seizure points, {} non_seizure points and {} post ictal points "
          .format(mode, len(seizure_end_points),
                  len(seizure_indices),
                  len(non_seizure_indices),
                  len(post_ictal_indices)))

    X_train = X["train"]
    X_val = X["val"]
    Y = labels["train"]
    Y_val = labels["val"]
    sample_time_train = sample_time["train"]
    sample_time_val = sample_time["val"]

    train_set = Epilepsy60Dataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y).float(),
                                  torch.from_numpy(sample_time_train).long())
    sampler = ImbalancedDataSampler(torch.from_numpy(seizure_indices).long(),
                                    torch.from_numpy(non_seizure_indices).long(),
                                    torch.from_numpy(post_ictal_indices).long(), overlap=20)
    train_loader = DataLoader(train_set, batch_size=32, sampler=sampler, num_workers=4)

    # classes = {0: 0, 1: 0}

    #     print(sample['y'])
    # for j in range(16):
    # idx = sample['idx'][j]
    # if np.isin(idx, post_ictal_indices, assume_unique=True):
    #     print(idx, "Post ictal", sample['y'][j])
    # elif np.isin(idx, non_seizure_indices, assume_unique=True):
    #     print(idx, "non ictal", sample['y'][j])
    # else:
    #     print(idx, "ictal", sample['y'][j])

    val_set = Epilepsy60Dataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).float(),
                                torch.from_numpy(sample_time_val).long())
    val_sampler = EvaluateSampler(torch.from_numpy(valid_labels['val']).int(), overlap=1)
    val_loader = DataLoader(val_set, shuffle=False, batch_size=16, sampler=val_sampler)

    # Training loop
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
    N_EPOCHS = 20
    train_loss_list = []
    val_loss_list = []

    for epoch in tqdm(range(N_EPOCHS), desc="Training"):
        model.train(True)  # turn on train mode
        train_loss = 0.0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}", position=0, leave=True)):
            optimizer.zero_grad()
            x, y = batch['x'], batch['y']
            x, y = x.to(device), y.to(device)
            x = torch.transpose(x, 0, 1)
            y_hat = model(x)
            loss = criterion(y_hat[-1, :, 0], y.view(-1, ))

            train_loss += loss.detach().cpu().item()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            running_vloss = 0.0
            for i, batch in enumerate(val_loader):
                x, y = batch['x'], batch['y']
                x, y = x.to(device), y.to(device)
                x = torch.transpose(x, 0, 1)
                voutputs = model(x)
                vloss = criterion(voutputs[-1, :, 0], y.view(-1, ))
                running_vloss += vloss.detach().cpu().item()

            avg_vloss = running_vloss / len(val_loader)
            print('LOSS valid {}'.format(avg_vloss))
            val_loss_list.append(avg_vloss)
            if avg_vloss <= np.min(np.array(val_loss_list)):
                torch.save(model.state_dict(), "{}_best".format(save_path))

            scheduler.step()
            lr = scheduler.get_last_lr()[0]
            #
        print(f"Epoch {epoch + 1}/{N_EPOCHS} Training loss: {train_loss / len(train_loader):.2f} ")
        train_loss_list.append(train_loss / len(train_loader))

    torch.save(model.state_dict(), save_path)
    print("Validation_loss_list = ", val_loss_list)
    print("Train_loss_list = ", train_loss_list)


def get_pat_start_end(pat_file_start_end):
    pat_start_end = {"train": [], "val": []}
    for mode in ["train", "val"]:
        for p in pat_file_start_end[mode]:
            if len(pat_file_start_end[mode][p]) == 0:
                continue
            pat_start_end[mode].append((pat_file_start_end[mode][p][0][0], pat_file_start_end[mode][p][-1][-1]))
    return pat_start_end


def pretrain(dataset):
    X, labels, minute_labels, pat_file_start_end, sample_time, valid_labels = get_data(pretrain_mode=True, dataset=dataset)
    print(pat_file_start_end)
    d_feature = 126 if dataset == "TUSZ" else 144
    d_model = 256
    n_heads = 4
    d_hid = 4 * d_model
    seq_len = SEQ_LEN + 3
    segment = SEGMENT
    n_layers = 4
    n_out = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device : ', device)
    model = BioTransformer(d_feature=d_feature, d_model=d_model, n_heads=n_heads, d_hid=d_hid, seq_len=seq_len,
                           n_layers=n_layers,
                           n_out=n_out, device=device, segments=segment).to(device)
    # model.load_state_dict(torch.load("../output/fake_model{}_n{}_{}".format(SEQ_LEN, n_layers, dataset)))

    X_train = X["train"]

    pat_start_end = get_pat_start_end(pat_file_start_end)

    print("Pat Start End: ", pat_start_end)

    train_set = PatientDiscriminatorDataset(torch.from_numpy(X_train).float(), pat_start_end['train'], sample_time['train'])
    sampler = EvaluateSampler(torch.from_numpy(valid_labels['train']).int(), overlap=20)
    train_loader = DataLoader(train_set, batch_size=128, sampler=sampler)

    validation_set = PatientDiscriminatorEvaluationDataset(torch.from_numpy(X["val"]).float(), pat_start_end['val'],
                                                           torch.from_numpy(minute_labels['val']).int(),
                                                           sample_time['val'])
    validation_loader = DataLoader(validation_set, batch_size=16, num_workers=4, shuffle=True)

    # it = iter(train_loader)
    # for i in range(5):
    #     sample = next(it)
    #     for j in range(16):
    #         plt.figure(figsize=(6, 6))
    #         sns.heatmap(sample['x'][j].cpu().numpy().transpose(), cmap="magma_r")
    #         plt.title("{}, LEN : {}".format(sample['y'][j].detach().cpu().item(), sample['len'][j].detach().cpu().item() ))
    #         plt.savefig('../output/{}.png'.format(i*16+j))
    #         plt.close()

    # Training loop
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)
    criterion = BCEWithLogitsLoss()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
    N_EPOCHS = 15
    val_loss_list = []
    save_path = '../output/pretrain_relative_model{}_n{}_{}'.format(SEQ_LEN, n_layers, dataset)
    for epoch in tqdm(range(N_EPOCHS), desc="Training"):
        model.train(True)  # turn on train mode
        train_loss = 0.0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}", position=0, leave=True)):
            optimizer.zero_grad()
            x, y = batch['x'], batch['y']
            x, y = x.to(device), y.to(device)
            x = torch.transpose(x, 0, 1)
            y_hat = model(x)
            loss = criterion(y_hat[-1, :, 0], y.view(-1, ))
            train_loss += loss.detach().cpu().item()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            running_vloss = 0.0
            for i, batch in enumerate(validation_loader):
                x, y = batch['x'], batch['y']
                x, y = x.to(device), y.to(device)
                x = torch.transpose(x, 0, 1)
                voutputs = model(x)
                vloss = criterion(voutputs[-1, :, 0], y.view(-1, ))
                running_vloss += vloss.detach().cpu().item()

            avg_vloss = running_vloss / len(validation_loader)
            print('LOSS valid {}'.format(avg_vloss))
            val_loss_list.append(avg_vloss)
            if avg_vloss <= np.min(np.array(val_loss_list)):
                torch.save(model.state_dict(), "{}_best".format(save_path))

        # scheduler.step()
        # lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch + 1}/{N_EPOCHS} Training loss: {train_loss / len(train_loader):.2f} ")

    torch.save(model.state_dict(), save_path)


def train_scratch(dataset):
    d_feature = 126 if dataset == "TUSZ" else 144
    d_model = 512
    n_heads = 8
    d_hid = 4 * d_model
    seq_len = SEQ_LEN + 3
    segment = SEGMENT
    n_layers = 8
    n_out = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device : ', device)
    model = BioTransformer(d_feature=d_feature, d_model=d_model, n_heads=n_heads, d_hid=d_hid, seq_len=seq_len,
                           n_layers=n_layers, n_out=n_out, device=device, segments=segment).to(device)
    savepath = '../output/model{}_{}_{}_scratch_relative'.format(SEQ_LEN, n_layers, dataset)
    train(model, device, savepath, learning_rate=1e-3)


def finetune():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device : ', device)

    d_feature = 126 #144
    d_model = 256
    n_heads = 4
    d_hid = 4 * d_model
    seq_len = SEQ_LEN + 7
    segment = SEGMENT
    n_layers = 4
    n_out = 1

    model = BioTransformer(d_feature=d_feature, d_model=d_model, n_heads=n_heads, d_hid=d_hid, seq_len=seq_len,
                           n_layers=n_layers,
                           n_out=n_out, device=device, segments=segment).to(device)
    model.load_state_dict(torch.load("../output/pretrain_model300_n4_TUSZ"))

    savepath = '../output/finetuned_model{}_n{}'.format(SEQ_LEN, n_layers)
    model.decoder.weight.data.uniform_(-0.1, 0.1)
    model.decoder.bias.data.uniform_(-0.1, 0.1)
    # model.sep_token.requires_grad = False
    # model.cls_token.requires_grad = False
    # model.encoder.weight.requires_grad = False
    # model.encoder.bias.requires_grad = False
    # for layer_num in range(n_layers):
    #     for param in model.transformer_encoder.layers[layer_num].parameters():
    #         param.requires_grad = False

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(
                name, param.data.shape)
    train(model, device, savepath, learning_rate=1e-4)


def print_results(conf):
    print("Confusion: ", conf)
    conf_normal = conf
    # conf_normal = conf / np.expand_dims(conf.astype(np.float).sum(axis=1), 1)
    sens = conf_normal[1, 1] / (conf_normal[1, 1] + conf_normal[0, 1])
    precision = conf_normal[1, 1] / (conf_normal[1, 1] + conf_normal[1, 0])
    print("Sensitivity: {:.2f}, Precision: {:.2f}".format(sens, precision))


def evaluate(dataset="TUSZ"):
    d_feature = 126 if dataset == "TUSZ" else 144
    d_model = 256
    n_heads = 4
    d_hid = 4 * d_model
    seq_len = SEQ_LEN + 7
    segment = SEGMENT
    n_layers = 4
    n_out = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device : ', device)
    model = BioTransformer(d_feature=d_feature, d_model=d_model, n_heads=n_heads, d_hid=d_hid, seq_len=seq_len,
                           n_layers=n_layers,
                           n_out=n_out, device=device, segments=segment).to(device)
    load_path = '../output/pretrain_model{}_n{}_{}'.format(SEQ_LEN, n_layers, dataset)
    model.load_state_dict(torch.load(load_path))

    print(model)

    X, labels, valid_labels, _, sample_time, _ = get_data(pretrain_mode=False, dataset="TUSZ")
    X_test = X["test"]
    Y = labels["test"]
    sample_time_test = sample_time["test"]

    test_set = Epilepsy60Dataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y).long(),
                                 torch.from_numpy(sample_time_test).long())
    test_sampler = EvaluateSampler(torch.from_numpy(valid_labels['test']).int(), overlap=1)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, sampler=test_sampler)

    model.eval()
    test_predict = []
    test_labels = []

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch['x'], batch['y']
            x, y = x.to(device), y.to(device)
            x = torch.transpose(x, 0, 1)
            outputs = model(x)[-1, :, :]
            # print('output shape: {}'.format(outputs.shape))
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            # print('predicted: {}'.format(predicted))
            test_predict += predicted.tolist()
            test_labels += y.view(-1, ).tolist()
            # print('labels: {}'.format(labels.view(-1,)))
            # correct += (predicted == labels.view(-1,)).sum().item()
            # print('(predicted == labels): {}'.format((predicted == labels.view(-1,))))
            # print('correct :{}'.format( (predicted == labels.view(-1,)).sum().item()))

    # print(f'Accuracy of the network on the test data: {100 * correct // total} %')
    conf = confusion_matrix(test_labels, test_predict)
    print_results(conf)
    print("F1 score: ", f1_score(test_labels, test_predict))


def evaluate_pretraining(dataset='TUSZ'):
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            if len(output) == 2:
                activation[name] = output[0].detach()
            else:
                activation[name] = output.detach()

        return hook

    d_feature = 126 if dataset == "TUSZ" else 144
    d_model = 256
    n_heads = 4
    d_hid = 4 * d_model
    seq_len = SEQ_LEN + 3
    segment = SEGMENT
    n_layers = 4
    n_out = 1
    torch.random.manual_seed(42)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device  = torch.device('cpu')
    print('device : ', device)
    model = BioTransformer(d_feature=d_feature, d_model=d_model, n_heads=n_heads, d_hid=d_hid, seq_len=seq_len,
                           n_layers=n_layers,
                           n_out=n_out, device=device, segments=segment).to(device)

    X, labels, minute_labels, pat_file_start_end, sample_time, valid_labels = get_data(pretrain_mode=True, dataset=dataset)
    X_train = X["train"]

    pat_start_end = get_pat_start_end(pat_file_start_end)

    train_set = PatientDiscriminatorDataset(torch.from_numpy(X_train).float(), pat_start_end['train'],
                                            sample_time['train'])
    sampler = EvaluateSampler(torch.from_numpy(valid_labels['train']).int(), overlap=60)
    train_loader = DataLoader(train_set, batch_size=1, sampler=sampler)
    model.load_state_dict(torch.load('../output/pretrain_relative_model{}_n{}_{}_best'.format(SEQ_LEN, n_layers, dataset)))

    # load_path = '../output/model{}_{}_{}_scratch_best'.format(SEQ_LEN, n_layers, dataset)
    # model.load_state_dict(torch.load(load_path))
    #
    # X, labels, valid_labels, _, sample_time, _ = get_data(pretrain_mode=False, dataset="TUSZ")
    # X_test = X["test"]
    # Y = labels["test"]
    # sample_time_test = sample_time["test"]
    #
    # test_set = Epilepsy60Dataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y).long(),
    #                              torch.from_numpy(sample_time_test).long())
    # test_sampler = EvaluateSampler(torch.from_numpy(valid_labels['test']).int(), overlap=1)
    # test_loader = DataLoader(test_set, batch_size=1, shuffle=False, sampler=test_sampler)

    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)

    hook1 = model.encoder.register_forward_hook(get_activation('layer0_in'))
    for i in range(1, n_layers):
        model.transformer_encoder.layers[i-1].register_forward_hook(get_activation('layer{}_in'.format(i)))
    # model.transformer_encoder.layers[0].self_attn.register_forward_hook(get_activation('self_attn'))
    # model.transformer_encoder.layers[0].norm1.register_forward_hook(get_activation('norm1'))
    # model.encoder.register_forward_hook(get_activation('encoder'))

    it = iter(train_loader)
    print(len(train_loader))
    mmd_label_df = pd.DataFrame(columns=['label', 'mmd'])
    for iteration in range(len(train_loader) - 1):
        batch = next(it)
        x, y = batch['x'], batch['y']
        x, y = x.to(device), y.to(device)
        x = torch.transpose(x, 0, 1)
        outputs = model(x)[0][-1, :, :]
        tokens = model(x)[1]
        plt.subplots(2, 1, figsize=(6, 12))
        plt.subplot(211)
        plt.title("{}, {}".format(y.detach().item(), batch['len'].detach()))
        sns.heatmap(x.squeeze().numpy().transpose(), cmap="magma_r")
        plt.subplot(212)
        sns.heatmap(tokens.detach().squeeze().numpy().transpose(), cmap="magma_r")
        # sns.heatmap(activation['self_attn'].squeeze().numpy().transpose(), cmap="magma_r")
        print(outputs, outputs.shape)

        fig, axes = plt.subplots(n_layers, 3, figsize=(12, 6))
        # fig, axes = plt.subplots(n_layers, n_heads*2, figsize=(12, 6))

        for l in range(n_layers):
            x_in = tokens
            in_proj_weight = model.transformer_encoder.layers[l].self_attn
            q, k, v = in_proj_weight.w_q, in_proj_weight.w_k, in_proj_weight.w_v
            Q = torch.einsum('tbf,hfa->bhta', x_in, q)
            K = torch.einsum('tbf,hfa->bhta', x_in, k)
            V = torch.einsum('tbf,hfa->bhta', x_in, v)
            attn = torch.einsum('bhqa,bhka->bhqk', Q, K).squeeze()
            print(Q.shape, K.shape, V.shape)
            print(attn.shape)

            # for h in range(n_heads):
            #     axes[l, 2*h].plot(attn[h,-1,:-1].detach().numpy())
            #     axes[l, 2*h+1].plot(attn[h,240,:240].detach().numpy())

            # att = torch.matmul(Q, torch.transpose(K, -2, -1))
            # print("Attention: ", att.shape)
            # attn_all = F.normalize(attn / 8.0, dim=-1)
            # attn_all = attn
            # scores = torch.matmul(attn_all, V)
            # print("Scores: ", scores.shape)
            # concat = scores.transpose(0, 1).contiguous().view(-1, d_model)
            # print("Concatenate: ", concat.shape)
            # mha = torch.matmul(concat, out_proj_weight) + out_proj_bias
            # print("MHA: ", mha.shape)
            #
            # mha = activation['self_attn'].squeeze()
            # print("MHA : ", mha.shape)
            # add_norm = x_in + norm_alpha *\
            #            (mha - mha.mean(dim=-1, keepdim=True)) / (mha.std(dim=-1, keepdim=True) + 1e-5) + norm_bias
            # print("Add/Norm: ", add_norm.shape)
            # plt.subplots(2, 1, figsize=(6, 12))
            # plt.subplot(211)
            # plt.title("Norm in Layer ".format(y.detach().item(), batch['len'].detach()))
            # sns.heatmap(concat.detach().squeeze().numpy().transpose(), cmap="magma_r")
            # plt.subplot(212)
            # sns.heatmap(add_norm.detach().numpy().transpose(), cmap="magma_r")
            # for h in range(n_heads):
            #     axes[l, h].imshow(attn_all[h, :, :].detach().numpy())
            important_index = [240, 271, 302]
            for idx, im in enumerate(important_index):
                # for row in range(seq_len):
                #     att[row, : ] = (att[row, : ] - torch.mean(att[row, :])) / torch.std(att[row, :])
                att_softmax = torch.mean(attn, dim=0)
                # attn_all[head, :, :] = att_softmax

                print("layer {}, head {}".format(l, im), att_softmax.shape)
                axes[l, idx].plot(att_softmax[im, :].detach().numpy())
            # attn_rollout = (torch.eye(seq_len) + torch.mean(attn_all, dim=0)) * attn_rollout if l!=0 else torch.mean(attn_all, dim=0)
            # print(attn_rollout.shape)
            # important_index = [240, -1]
            # for idx, im in enumerate(important_index):
            #     attn_rollout[im, im] = 0
            #     axes[l, idx].plot(attn_rollout[im, :].detach().numpy())

        plt.show()


    exit()


    model.eval()
    test_predict = []
    test_labels = []
    # randperm_seg = torch.randperm(SEGMENT)
    # randperm_roi = torch.randperm(SEQ_LEN)
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for batch in tqdm(train_loader, position=0, leave=True):
            x, y = batch['x'], batch['y']
            # x[:, :SEGMENT, :] = x[:, randperm_seg, :]
            x, y = x.to(device), y.to(device)
            x = torch.transpose(x, 0, 1)
            y_hat = model(x)[-1, :, 0]
            predicted = y_hat.data
            # print('predicted: {}'.format(predicted))
            test_predict += predicted.tolist()
            test_labels += y.view(-1, ).tolist()

            same_sample = np.where(y.cpu().numpy() == 0)[0][0]
            diff_sample = np.where(y.cpu().numpy() == 1)[0][0]
            #
            # print(activation['pos_encoder'].shape)
            #
            plt.figure(figsize=(6,6))
            sns.heatmap(activation['encoder'][:,same_sample,:].cpu().numpy().transpose(), cmap="magma_r")
            plt.figure(figsize=(6,6))
            sns.heatmap(activation['pos_encoder'][:,same_sample,:].cpu().numpy().transpose(),  cmap="magma_r")
            # sns.heatmap((activation['pos_encoder'][-61:-1,same_sample,:]- activation['encoder'][:,same_sample,:]  * math.sqrt(512)).cpu().numpy().transpose(),  cmap="magma_r")

            plt.figure(figsize=(3, 6))
            sns.heatmap(activation['encoder'][:, diff_sample, :].cpu().numpy().transpose())
            plt.figure(figsize=(15, 6))
            sns.heatmap(activation['pos_encoder'][:, diff_sample, :].cpu().numpy().transpose())
            plt.show()
            # return

    conf = confusion_matrix(test_labels, test_predict)
    print_results(conf)


def visualize_model():
    x_total = np.zeros((0, 126))
    for t_file in [
        # "../input/Epilepsiae_info/Patient_1_26.mat_zc.pickle",
        # "../input/Epilepsiae_info/Patient_1_37.mat_zc.pickle",
        "../input/TUSZ/00000254_s005_t000.edf_zc.pickle",
        "../input/TUSZ/00000272_s007_t000.edf_zc.pickle",
        "../input/TUSZ/00012679_s003_t010.edf_zc.pickle",
    ]:
        with open(t_file, 'rb') as pickle_file:
            # print(t_file)
            data = pickle.load(pickle_file)
            print(data.shape)
            x_total = np.concatenate((x_total, data))
    x_total = (x_total - 4.41) / 8.6
    d_feature = 126
    d_model = 768
    n_heads = 12
    d_hid = 4 * d_model
    seq_len = SEQ_LEN + 6
    segment = SEGMENT
    n_layers = 12
    n_out = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device : ', device)
    model = BioTransformer(d_feature=d_feature, d_model=d_model, n_heads=n_heads, d_hid=d_hid, seq_len=seq_len,
                           n_layers=n_layers,
                           n_out=n_out, device=device, segments=segment).to(device)
    model.load_state_dict(torch.load('../output/pre_model300_n12_best'))
    print(model)

    start_point = 1438 - SEGMENT
    x = torch.unsqueeze(torch.from_numpy(x_total[start_point:start_point + SEQ_LEN]).float().to(device), dim=0)
    print(x.shape)
    x = torch.transpose(x, 0, 1)
    result = model(x)
    print(result.shape)
    print(result[-1, :, :])
    plt.figure(figsize=(6, 6))
    sns.heatmap(x[:, 0, :].cpu().numpy().transpose(), cmap="magma_r")
    plt.savefig('../output/figures/pat254_272.png')


if __name__ == '__main__':
    # train()
    # pretrain("TUSZ")
    # evaluate()
    train_scratch(dataset="TUSZ")
    # evaluate_pretraining()
    # finetune()
    # visualize_model()
