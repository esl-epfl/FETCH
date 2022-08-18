import numpy as np
import scipy.io
import pandas as pd
import json
import pickle
from utils.BioT import BioTransformer, Epilepsy60Dataset, ImbalancedDataSampler
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import SGD, Adam
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix


def set_labels(x):
    for seizure_num in range(len(x['onsets'])):
        start = max(x['onsets'][seizure_num][0] // 256 - 3, 0)
        end = x['offsets'][seizure_num][0] // 256 + 4
        x['labels'][start:end] = 1


def get_data():
    df = pd.read_csv('../input/Epilepsiae_info/epilepsiae_labels.csv')
    df['labels'] = df['length'].apply(lambda x: np.zeros(x // 256 - 4, dtype=np.int))
    df['onsets'] = df['onsets'].apply(lambda x: json.loads(x.replace('\n', ',')))
    df['offsets'] = df['offsets'].apply(lambda x: json.loads(x.replace('\n', ',')))
    df.apply(set_labels, axis=1)

    test_set = [x for x in df['file_name'].tolist() if x.startswith('Patient_1_')]
    validation_set = [x for x in df['file_name'].tolist() if x.startswith('Patient_2_')]
    training_set = [x for x in df['file_name'].tolist() if (not x in test_set) and (not x in validation_set)]
    df_file_name = df.set_index('file_name')

    X = {'train': np.zeros((0, 144)), 'test': np.zeros((0, 144)), 'val': np.zeros((0, 144))}
    labels = {'train': np.zeros((0, 1)), 'test': np.zeros((0, 1)), 'val': np.zeros((0, 1))}
    valid_labels = {'train': np.zeros(0), 'test': np.zeros(0), 'val': np.zeros(0)}
    dataset = {'train': training_set, 'test': test_set, 'val': validation_set}
    for mode in ['train', 'test', 'val']:
        for t_file in dataset[mode]:
            with open('../input/Epilepsiae_info/{}_zc.pickle'.format(t_file), 'rb') as pickle_file:
                data = pickle.load(pickle_file)
                X[mode] = np.concatenate((X[mode], data), axis=0)
            y = df_file_name.loc[t_file, 'labels']
            labels[mode] = np.concatenate((labels[mode], np.expand_dims(y, axis=1)))
            valid_index = np.arange(start=valid_labels[mode].shape[0] + 59,
                                    stop=valid_labels[mode].shape[0] + y.shape[0])
            valid_labels[mode] = np.concatenate((valid_labels[mode], valid_index))
    print(X["train"].shape)
    print(X["val"].shape)
    print(X["test"].shape)
    mean_train = np.mean(X["train"])
    std_train = np.std(X["train"])
    X["train"] = (X["train"] - mean_train) / std_train
    X["val"] = (X["val"] - mean_train) / std_train
    X["test"] = (X["test"] - mean_train) / std_train
    print(valid_labels["train"].shape)
    return X, labels, valid_labels


def train():
    X, labels, valid_labels = get_data()

    d_feature = 144
    d_model = 768
    n_heads = 12
    d_hid = 4 * d_model
    seq_len = 61
    n_layers = 12
    n_out = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device : ', device)
    model = BioTransformer(d_feature=d_feature, d_model=d_model, n_heads=n_heads, d_hid=d_hid, seq_len=seq_len, n_layers=n_layers,
                           n_out=n_out, device=device).to(device)

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
    train_loader = DataLoader(train_set, batch_size=32, sampler=sampler, num_workers=4)

    val_set = Epilepsy60Dataset(torch.from_numpy(X_val).float(), torch.from_numpy(Y_val).long())
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    # Training loop
    optimizer = SGD(model.parameters(), lr=0.01)
    criterion = CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
    N_EPOCHS = 50
    train_loss_list = []
    val_loss_list = []

    for epoch in tqdm(range(N_EPOCHS), desc="Training"):
        model.train(True)  # turn on train mode
        train_loss = 0.0
        class_samples = {0: 0, 1: 0}
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}", position=0, leave=True)):
            optimizer.zero_grad()
            x, y = batch['x'], batch['y']
            x, y = x.to(device), y.to(device)
            x = torch.transpose(x, 0, 1)
            y_hat = model(x)
            loss = criterion(y_hat[-1, :, :], y.view(-1, ))

            train_loss += loss.detach().cpu().item()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        model.train(False)
        running_vloss = 0.0
        for i, vdata in enumerate(val_loader):
            vinputs, vlabels = vdata['x'], vdata['y']
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            vinputs = torch.transpose(vinputs, 0, 1)
            voutputs = model(vinputs)
            vloss = criterion(voutputs[-1, :, :], vlabels.view(-1, ))
            running_vloss += vloss.detach().cpu().item()

        avg_vloss = running_vloss/ len(val_loader)
        print('LOSS valid {}'.format(avg_vloss))
        val_loss_list.append(avg_vloss)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch + 1}/{N_EPOCHS} Training loss: {train_loss/len(train_loader):.2f} ")
        train_loss_list.append(train_loss/len(train_loader))

    torch.save(model, '../output/model60_n{}'.format(n_layers))
    torch.save(model.state_dict(), '../output/model60_state_n{}'.format(n_layers))
    print("Validation_loss_list = ", val_loss_list)
    print("Train_loss_list = ", train_loss_list)


def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X, labels, valid_labels = get_data()
    X_test = X["test"]
    Y = labels["test"]

    test_set = Epilepsy60Dataset(torch.from_numpy(X_test).float(), torch.from_numpy(Y).long())
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False)

    model = torch.load('../output/model60_n16')
    model.eval()
    test_predict = []
    test_labels = []

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data['x'].to(device), data['y'].to(device)
            # print('data shape: {}'.format(images.shape))
            # print('label shape: {}'.format(labels.shape))
            # calculate outputs by running images through the network
            images = torch.transpose(images, 0, 1)
            outputs = model(images)[-1,:,:]
            # print('output shape: {}'.format(outputs.shape))
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            # print('predicted: {}'.format(predicted))
            test_predict += predicted.tolist()
            test_labels += labels.view(-1,).tolist()
            # print('labels: {}'.format(labels.view(-1,)))
            # correct += (predicted == labels.view(-1,)).sum().item()
            # print('(predicted == labels): {}'.format((predicted == labels.view(-1,))))
            # print('correct :{}'.format( (predicted == labels.view(-1,)).sum().item()))

    # print(f'Accuracy of the network on the test data: {100 * correct // total} %')
    print("Confusion: ", confusion_matrix(test_labels, test_predict))


if __name__ == '__main__':
    train()
    # evaluate()
