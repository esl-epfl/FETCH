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
from torch.optim import SGD
from tqdm import tqdm
import torch.nn.functional as F


def set_labels(x):
    for seizure_num in range(len(x['onsets'])):
        start = max(x['onsets'][seizure_num][0] // 256 -3, 0)
        end = x['offsets'][seizure_num][0] // 256 + 4
        x['labels'][start:end] = 1


def train():
    df = pd.read_csv('../input/Epilepsiae_info/epilepsiae_labels.csv')
    df['labels'] = df['length'].apply(lambda x: np.zeros(x//256 - 4, dtype=np.int))
    df['onsets']= df['onsets'].apply(lambda x:json.loads(x.replace('\n', ',')))
    df['offsets']= df['offsets'].apply(lambda x:json.loads(x.replace('\n', ',')))
    df.apply(set_labels, axis=1)

    training_set = [x for x in df['file_name'].tolist() if not x.startswith('Patient_1_')]
    test_set = [x for x in df['file_name'].tolist() if not x in training_set]
    df_file_name = df.set_index('file_name')


    X = {'train': np.zeros((0, 144)), 'test': np.zeros((0, 144))}
    labels = {'train': np.zeros((0, 1)), 'test': np.zeros((0, 1))}
    valid_labels = {'train': np.zeros(0), 'test': np.zeros(0)}
    dataset = {'train': training_set, 'test': test_set}
    for mode in ['train', 'test']:
        for t_file in dataset[mode]:
            with open('../input/Epilepsiae_info/{}_zc.pickle'.format(t_file), 'rb') as pickle_file:
                data = pickle.load(pickle_file)
                X[mode] = np.concatenate((X[mode], data), axis=0)
            y = df_file_name.loc[t_file, 'labels']
            # y_onehot = np.zeros((y.size, 2))
            # y_onehot[np.arange(y.size), y] = 1
            labels[mode] = np.concatenate((labels[mode], np.expand_dims(y,axis=1)))
            valid_index = np.arange(start=valid_labels[mode].shape[0] + 59, stop=valid_labels[mode].shape[0] + y.shape[0])
            valid_labels[mode] = np.concatenate((valid_labels[mode], valid_index))
    print(X["train"].shape)
    mean_train = np.mean(X["train"])
    std_train = np.std(X["train"])
    X["train"] = (X["train"] - mean_train) / std_train
    X["test"] = (X["test"] - mean_train) / std_train
    print(valid_labels["train"].shape)


    d_model = 144
    n_heads = 2
    d_hid = 4*d_model
    seq_len = 60
    n_layers = 8
    n_out = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device : ', device)
    model = BioTransformer(d_model=d_model, n_heads=n_heads, d_hid=d_hid, seq_len=seq_len, n_layers=n_layers, n_out=n_out, device=device).to(device)

    #%%
    seizure_indices = np.where(labels['train']== 1)[0]
    non_seizure_indices = np.where(labels['train'] == 0)[0]
    valid_seizures = np.intersect1d(seizure_indices.astype(np.int32),valid_labels["train"].astype(np.int32))
    valid_non_seizure = np.intersect1d(non_seizure_indices.astype(np.int32),valid_labels["train"].astype(np.int32))

    X_train = X["train"]
    Y = labels["train"]

    print(Y.dtype)
    print(X_train.shape, Y.shape)
    train_set = Epilepsy60Dataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y).long())
    sampler = ImbalancedDataSampler(torch.from_numpy(valid_seizures), torch.from_numpy(valid_non_seizure))
    train_loader = DataLoader(train_set, batch_size=16, sampler=sampler,  num_workers=6)


    # Training loop
    optimizer = SGD(model.parameters(), lr=5.0)
    criterion = CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97)
    N_EPOCHS = 250

    model.train()  # turn on train mode
    for epoch in tqdm(range(N_EPOCHS), desc="Training"):
        train_loss = 0.0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}",  position=0, leave=True)):
            optimizer.zero_grad()
            x, y = batch['x'], batch['y']
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat[:,-1,:], y.view(-1,))

            train_loss += loss.detach().cpu().item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch + 1}/{N_EPOCHS} loss: {train_loss:.2f} lr {lr:.2f}")


    torch.save(model, '../output/model60_n{}'.format(n_layers))
    torch.save(model.state_dict(), '../output/model60_state_n{}'.format(n_layers))


if __name__ == '__main__':
    train()