import math

import numpy as np
import pandas as pd
import json
import pickle
import seaborn as sns
from utils.BioT import BioTransformer, Epilepsy60Dataset, ImbalancedDataSampler, EvaluateSampler,\
    PatientDiscriminatorDataset, PatientDiscriminatorEvaluationDataset
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import SGD, AdamW
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from utils.BioT import SEQ_LEN, SEGMENT, ROI


def get_data(pretrain_mode=False):
    def set_labels(x):
        for seizure_num in range(len(x['onsets'])):
            start = max(x['onsets'][seizure_num][0] // 256 - 3, 0)
            end = x['offsets'][seizure_num][0] // 256 + 4
            x['labels'][start:end] = 1

    rootdir = '../input/Epilepsiae_total' if pretrain_mode else '../input/Epilepsiae_info'
    df = pd.read_csv(rootdir + '/epilepsiae_labels.csv')
    if not pretrain_mode:
        df['labels'] = df['length'].apply(lambda x: np.zeros(x // 256 - 4, dtype=np.int))
        df['onsets'] = df['onsets'].apply(lambda x: json.loads(x.replace('\n', ',')))
        df['offsets'] = df['offsets'].apply(lambda x: json.loads(x.replace('\n', ',')))
        df.apply(set_labels, axis=1)

    df = df.groupby('patient').head(1)
    df = df.sort_values(by='patient')

    test_set = [x for x in df['file_name'].tolist() if x.startswith('Patient_1_')]
    if pretrain_mode:
        validation_set = [x for x in (df.groupby('patient').head(1))['file_name'].tolist() if
                          (x.startswith('Patient_2_') or x.startswith('Patient_3_'))]
    else:  # pretrain mode
        validation_set = [x for x in df['file_name'].tolist() if x.startswith('Patient_2_')]
    training_set = [x for x in df['file_name'].tolist() if (not x in test_set) and (not x in validation_set)]
    df_file_name = df.set_index('file_name')

    X = {'train': np.zeros((0, 144)), 'test': np.zeros((0, 144)), 'val': np.zeros((0, 144))}
    labels = {'train': np.zeros((0, 1)), 'test': np.zeros((0, 1)), 'val': np.zeros((0, 1))}
    valid_labels = {'train': np.zeros(0, dtype=np.int), 'test': np.zeros(0, dtype=np.int), 'val': np.zeros(0, dtype=np.int)}
    dataset = {'train': training_set, 'test': test_set, 'val': validation_set}
    pat_start_end = {'train': {new_list: [] for new_list in range(30)},
                     'test': {new_list: [] for new_list in range(30)},
                     'val': {new_list: [] for new_list in range(30)}}
    for mode in ['train', 'test', 'val']:
        for t_file in dataset[mode]:
            with open(rootdir + '/{}_zc.pickle'.format(t_file), 'rb') as pickle_file:
                # print(t_file)
                data = pickle.load(pickle_file)
                X[mode] = np.concatenate((X[mode], data), axis=0)
                y = np.zeros(data.shape[0]) if pretrain_mode else df_file_name.loc[t_file, 'labels']
            pat_num = int(t_file.split('_')[1]) - 1

            valid_start = labels[mode].shape[0] + SEQ_LEN-1
            valid_end = labels[mode].shape[0] + y.shape[0]
            valid_labels[mode] = np.concatenate((valid_labels[mode],  np.arange(start= valid_start, stop= valid_end)))
            labels[mode] = np.concatenate((labels[mode], np.expand_dims(y, axis=1)))

            pat_start_end[mode][pat_num].append((valid_start, valid_end))

    print(X["train"].shape)
    print(X["val"].shape)
    print(X["test"].shape)
    mean_train = np.mean(X["train"])
    std_train = np.std(X["train"])
    X["train"] = (X["train"] - mean_train) / std_train
    X["val"] = (X["val"] - mean_train) / std_train
    X["test"] = (X["test"] - mean_train) / std_train
    print(valid_labels["train"].shape)
    return X, labels, valid_labels, pat_start_end


def train(model, device, save_path:str, learning_rate:float = 0.01):
    X, labels, valid_labels, _ = get_data()

    # %%
    seizure_indices = np.where(labels['train'] == 1)[0]
    non_seizure_indices = np.where(labels['train'] == 0)[0]
    valid_seizures = np.intersect1d(seizure_indices.astype(np.int32), valid_labels["train"].astype(np.int32))
    valid_non_seizure = np.intersect1d(non_seizure_indices.astype(np.int32), valid_labels["train"].astype(np.int32))

    X_train = X["train"]
    X_val = X["val"]
    Y = labels["train"]
    Y_val = labels["val"]

    print(Y.dtype)
    print(X_train.shape, Y.shape)
    train_set = Epilepsy60Dataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y).long())
    sampler = ImbalancedDataSampler(torch.from_numpy(valid_seizures), torch.from_numpy(valid_non_seizure))
    train_loader = DataLoader(train_set, batch_size=16, sampler=sampler, num_workers=4)

    val_set = Epilepsy60Dataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).long())
    val_sampler = EvaluateSampler(torch.from_numpy(valid_labels['val']).int())
    val_loader = DataLoader(val_set, shuffle=False, sampler=val_sampler, batch_size=16)

    # Training loop
    optimizer = SGD(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
    N_EPOCHS = 20
    train_loss_list = []
    val_loss_list = []

    for epoch in tqdm(range(N_EPOCHS), desc="Training"):
        model.train(True)  # turn on train mode
        train_loss = 0.0
        class_samples = {0: 0, 1: 0}
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}", position=0, leave=True)):
            optimizer.zero_grad()
            x, y = batch['x'], batch['y']
            x1, x2 = x[:,:SEGMENT, :], x[:, SEGMENT:,:]
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            x1 = torch.transpose(x1, 0, 1)
            x2 = torch.transpose(x2, 0, 1)
            y_hat = model(x1, x2)
            loss = criterion(y_hat[-1, :, :], y.view(-1, ))

            train_loss += loss.detach().cpu().item()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            running_vloss = 0.0
            for i, batch in enumerate(val_loader):
                x, y = batch['x'], batch['y']
                x1, x2 = x[:, :SEGMENT, :], x[:, SEGMENT:, :]
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                x1 = torch.transpose(x1, 0, 1)
                x2 = torch.transpose(x2, 0, 1)
                voutputs = model(x1, x2)
                vloss = criterion(voutputs[-1, :, :], y.view(-1, ))
                running_vloss += vloss.detach().cpu().item()

            avg_vloss = running_vloss/ len(val_loader)
            print('LOSS valid {}'.format(avg_vloss))
            val_loss_list.append(avg_vloss)

            scheduler.step()
            lr = scheduler.get_last_lr()[0]

            print(f"Epoch {epoch + 1}/{N_EPOCHS} Training loss: {train_loss/len(train_loader):.2f} ")
            train_loss_list.append(train_loss/len(train_loader))

    torch.save(model, save_path)
    print("Validation_loss_list = ", val_loss_list)
    print("Train_loss_list = ", train_loss_list)


def get_pat_start_end(pat_file_start_end):
    pat_start_end = {"train": [], "val": []}
    for mode in ["train", "val"]:
        for p in range(30):
            if len(pat_file_start_end[mode][p]) == 0:
                continue
            pat_start_end[mode].append((pat_file_start_end[mode][p][0][0], pat_file_start_end[mode][p][-1][-1]))
    return pat_start_end


def pretrain():
    X, labels, valid_labels, pat_file_start_end = get_data(pretrain_mode=True)
    print(pat_file_start_end)
    d_feature = 144
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
    model.load_state_dict(torch.load("../output/pre_model300_n12_2"))

    X_train = X["train"]

    pat_start_end = get_pat_start_end(pat_file_start_end)

    validation_set = PatientDiscriminatorEvaluationDataset(torch.from_numpy(X["val"]).float(), pat_start_end['val'],
                                                           torch.from_numpy(valid_labels['val']).int())
    validation_loader = DataLoader(validation_set, batch_size=16, num_workers=4)

    train_set = PatientDiscriminatorDataset(torch.from_numpy(X_train).float(), pat_start_end['train'])
    sampler = EvaluateSampler(torch.from_numpy(valid_labels['train']).int(), overlap=60)
    train_loader = DataLoader(train_set, batch_size=16, sampler=sampler, num_workers=4)

    # Training loop
    optimizer = AdamW(model.parameters(), lr=1e-5)
    criterion = CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
    N_EPOCHS = 20
    val_loss_list = []
    for epoch in tqdm(range(N_EPOCHS), desc="Training"):
        model.train(True)  # turn on train mode
        train_loss = 0.0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}", position=0, leave=True)):
            optimizer.zero_grad()
            x, y = batch['x'], batch['y']
            # print(np.min(np.sum(np.abs((X_train-x1[0,0].numpy())), axis=1)))
            # x1_index = np.argwhere(np.all(np.abs(X_train-x1[0,0].numpy())<1e-6, axis=1))[0,0]
            # x2_index = np.argwhere(np.all(np.abs(X_train-x2[0,0].numpy())<1e-6, axis=1))[0,0]
            # print(x1_index, x2_index, y.numpy()[0])
            x, y = x.to(device), y.to(device)
            x = torch.transpose(x, 0, 1)
            y_hat = model(x)
            loss = criterion(y_hat[-1, :, :], y.view(-1, ))

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
                vloss = criterion(voutputs[-1, :, :], y.view(-1, ))
                running_vloss += vloss.detach().cpu().item()

            avg_vloss = running_vloss / len(validation_loader)
            print('LOSS valid {}'.format(avg_vloss))
            val_loss_list.append(avg_vloss)
            if avg_vloss <= np.min(np.array(val_loss_list)):
                torch.save(model.state_dict(), '../output/pre_model{}_n{}_best'.format(SEQ_LEN, n_layers))

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch + 1}/{N_EPOCHS} Training loss: {train_loss / len(train_loader):.2f} ")

    torch.save(model.state_dict(), '../output/pre_model{}_n{}_2'.format(SEQ_LEN, n_layers))
    # torch.save(model.state_dict(), '../output/pre_model{}_state_n{}'.format(SEQ_LEN, n_layers))


def train_scratch():
    d_feature = 144
    d_model = 768
    n_heads = 12
    d_hid = 4 * d_model
    seq_len = SEQ_LEN+2
    segment = SEGMENT
    n_layers = 12
    n_out = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device : ', device)
    model = BioTransformer(d_feature=d_feature, d_model=d_model, n_heads=n_heads, d_hid=d_hid, seq_len=seq_len, n_layers=n_layers,
                           n_out=n_out, device=device, segments=segment).to(device)
    savepath = '../output/model{}_n{}'.format(SEQ_LEN, n_layers)
    train(model, device, savepath)


def finetune():
    model = torch.load('../output/pre_model{}_n12'.format(SEQ_LEN))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device : ', device)
    savepath = '../output/finetuned_model{}_n12_frozen12'.format(SEQ_LEN)
    model.decoder.weight.data.uniform_(-0.1, 0.1)
    model.decoder.bias.data.uniform_(-0.1, 0.1)
    model.encoder.weight.requires_grad = False
    model.encoder.bias.requires_grad = False
    model.sep_token.requires_grad = False
    model.class_token.requires_grad = False
    for layer_num in range(12):
        for param in model.transformer_encoder.layers[layer_num].parameters():
            param.requires_grad = False

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(
            name, param.data.shape)
    train(model,device, savepath, learning_rate=0.001)


def print_results(conf):
    print("Confusion: ", conf)
    conf_normal = conf / np.expand_dims(conf.astype(np.float).sum(axis=1), 1)
    sens = conf_normal[1, 1] / (conf_normal[1, 1] + conf_normal[0, 1])
    spec = conf_normal[1, 1] / (conf_normal[1, 1] + conf_normal[1, 0])
    print("Sensitivity: {:.2f}, Specificity: {:.2f}".format(sens, spec))


def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, labels, valid_labels, _ = get_data()
    X_test = X["test"]
    Y = labels["test"]

    test_set = Epilepsy60Dataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y).long())
    test_sampler = EvaluateSampler(torch.from_numpy(valid_labels['test']).int())
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, sampler=test_sampler)

    model = torch.load('../output/model{}_n12'.format(SEQ_LEN))
    model.eval()
    test_predict = []
    test_labels = []

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch['x'], batch['y']
            x1, x2 = x[:, :SEGMENT, :], x[:, SEGMENT:, :]
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            x1 = torch.transpose(x1, 0, 1)
            x2 = torch.transpose(x2, 0, 1)
            outputs = model(x1, x2)[-1,:,:]
            # print('output shape: {}'.format(outputs.shape))
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            # print('predicted: {}'.format(predicted))
            test_predict += predicted.tolist()
            test_labels += y.view(-1,).tolist()
            # print('labels: {}'.format(labels.view(-1,)))
            # correct += (predicted == labels.view(-1,)).sum().item()
            # print('(predicted == labels): {}'.format((predicted == labels.view(-1,))))
            # print('correct :{}'.format( (predicted == labels.view(-1,)).sum().item()))

    # print(f'Accuracy of the network on the test data: {100 * correct // total} %')
    conf = confusion_matrix(test_labels, test_predict)
    print_results(conf)


def evaluate_pretraining():
    # activation = {}
    #
    # def get_activation(name):
    #     def hook(model, input, output):
    #         activation[name] = output.detach()
    #
    #     return hook

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ',device)
    X, labels, valid_labels, pat_file_start_end = get_data(pretrain_mode=True)
    X_train = X["val"]

    pat_start_end = get_pat_start_end(pat_file_start_end)

    validation_set = PatientDiscriminatorEvaluationDataset(torch.from_numpy(X["val"]).float(), pat_start_end['val'],
                                                           torch.from_numpy(valid_labels['val']).int())
    validation_loader = DataLoader(validation_set, batch_size=16, num_workers=4)

    model = torch.load('../output/pre_model{}_n12'.format(SEQ_LEN))
    print(model)
    # model.encoder.register_forward_hook(get_activation('encoder'))
    # model.pos_encoder.register_forward_hook(get_activation('pos_encoder'))

    model.eval()
    test_predict = []
    test_labels = []
    randperm_seg = torch.randperm(SEGMENT)
    # randperm_roi = torch.randperm(SEQ_LEN)
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for batch in tqdm(validation_loader, position=0, leave=True):
            x, y = batch['x'], batch['y']
            x[:,:SEGMENT ,:] = x[:, randperm_seg, :]
            x, y = x.to(device), y.to(device)
            x = torch.transpose(x, 0, 1)
            y_hat = model(x)[-1,:,:]
            _, predicted = torch.max(y_hat.data, 1)
            # print('predicted: {}'.format(predicted))
            test_predict += predicted.tolist()
            test_labels += y.view(-1, ).tolist()

            # same_sample = np.where(y.cpu().numpy() == 0)[0][0]
            # diff_sample = np.where(y.cpu().numpy() == 1)[0][0]
            #
            # print(activation['pos_encoder'].shape)
            #
            # plt.figure(figsize=(6,6))
            # sns.heatmap(activation['encoder'][:,same_sample,:].cpu().numpy().transpose(), cmap="magma_r")
            # plt.figure(figsize=(6,6))
            # sns.heatmap(activation['pos_encoder'][:,same_sample,:].cpu().numpy().transpose(),  cmap="magma_r")
            # # sns.heatmap((activation['pos_encoder'][-61:-1,same_sample,:]- activation['encoder'][:,same_sample,:]  * math.sqrt(512)).cpu().numpy().transpose(),  cmap="magma_r")
            #
            # plt.figure(figsize=(3, 6))
            # sns.heatmap(activation['encoder'][:, diff_sample, :].cpu().numpy().transpose())
            # plt.figure(figsize=(15, 6))
            # sns.heatmap(activation['pos_encoder'][:, diff_sample, :].cpu().numpy().transpose())
            # plt.show()
            # return

    conf = confusion_matrix(test_labels, test_predict)
    print_results(conf)


if __name__ == '__main__':
    # train()
    pretrain()
    # evaluate()
    # train_scratch()
    # evaluate_pretraining()
    # finetune()
