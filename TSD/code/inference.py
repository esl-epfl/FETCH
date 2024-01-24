import os
import sys
import math
import json
import warnings

import pandas as pd

# Filter out the specific UserWarning related to torchvision
warnings.filterwarnings("ignore", category=UserWarning, message="Failed to load image Python extension")
# TODO solve the CUDA version issue

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import time
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import numpy as np
from tqdm import tqdm
from tuh_dataset import get_dataloader, get_data
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
from sklearn.metrics import confusion_matrix
from tuh_dataset import args as tuh_args
from TSD.code.utils import thresh_max_f1
from TSD.code.utils import get_feasible_ids_with_num_nodes
from TSD.code.utils import create_dataframe


def seed_everything(seed=99):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=99)


def inference(df, train_data, validation_signal, val_label, test_signal, test_label,
              device, model, sigmoid,
              selected_channel_id=tuh_args.selected_channel_id):

    selected_channels = df[df['channel_id'] == selected_channel_id]['channel_list'].values[0]
    print("selected_channels: ", selected_channels)

    _, val_loader, test_loader = \
        get_dataloader(train_data=train_data,
                       train_signal=None, train_label=None,
                       val_data=None, test_data=None,
                       validation_signal=validation_signal, val_label=val_label,
                       test_signal=test_signal, test_label=test_label,
                       batch_size=128,
                       selected_channel_id=selected_channel_id,
                       return_dataset=False,
                       event_base=False, masking=True, random_mask=False,
                       remove_not_used=False)

    val_label_all = torch.zeros(len(val_loader.dataset), dtype=torch.int).to(device)
    val_prob_all = torch.zeros(len(val_loader.dataset), dtype=torch.float).to(device)
    with torch.no_grad():
        for i, (data, label) in enumerate(tqdm(val_loader)):
            start_idx = i * val_loader.batch_size
            end_idx = start_idx + data.size(0)

            val_label_all[start_idx:end_idx] = label
            val_prob = model(data.to(device))
            val_prob = torch.squeeze(sigmoid(val_prob))
            val_prob_all[start_idx:end_idx] = val_prob

    val_label_all = val_label_all.cpu().numpy()
    val_prob_all = val_prob_all.cpu().numpy()

    val_auc = roc_auc_score(val_label_all, val_prob_all)

    test_label_all = torch.zeros(len(test_loader.dataset), dtype=torch.int).to(device)
    test_prob_all = torch.zeros(len(test_loader.dataset), dtype=torch.float).to(device)

    with torch.no_grad():
        for i, (data, label) in enumerate(tqdm(test_loader)):
            start_idx = i * test_loader.batch_size
            end_idx = start_idx + data.size(0)

            test_label_all[start_idx:end_idx] = label
            test_prob = model(data.to(device))
            test_prob = torch.squeeze(sigmoid(test_prob))
            test_prob_all[start_idx:end_idx] = test_prob

    test_label_all = test_label_all.cpu().numpy()
    test_prob_all = test_prob_all.cpu().numpy()

    test_auc = roc_auc_score(test_label_all, test_prob_all)
    return val_auc, test_auc


def extract_epoch(filename):
    return int(filename.split('_')[1])


def get_highest_epoch_file(files):
    if not files:
        return None

    highest_epoch = max(extract_epoch(file) for file in files)
    highest_epoch_files = [file for file in files if extract_epoch(file) == highest_epoch]

    return highest_epoch, highest_epoch_files


def test_models_with_nodes(num_nodes):
    df = get_feasible_ids_with_num_nodes(num_nodes)

    # Create a dataframe to store the results
    results_df = pd.DataFrame(columns=['channel_id', 'val_auc', 'test_auc',
                                       'experiment_name', 'model_name', 'number_nodes'])

    sample_rate = 256
    eeg_type = 'stft'
    device = 'cuda:0' if tuh_args.cuda else 'cpu'
    print("device: ", device)

    root_path = tuh_args.experiment_root
    if tuh_args.global_model:  # Global model
        model_path = os.path.join(root_path, 'global')
        print("Global Model ", model_path)

    else:
        # sample of the model path: output/model_8nodes/best_model.pth
        model_path = os.path.join(root_path, 'model_{}nodes'.format(num_nodes),
                                  'best_model.pth')

    model = torch.load(model_path, map_location=torch.device(device)).float()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.eval()
    sigmoid = nn.Sigmoid()

    (train_data, _, _, _, _,
     validation_signal, val_label,
     test_signal, test_label) = \
        get_data(save_dir=tuh_args.save_directory,
                 balanced_data=True,
                 return_val_test_signal=True,
                 return_train_signal=False)
    validation_signal = validation_signal.to(device)
    test_signal = test_signal.to(device)

    for i, row in tqdm(df.iterrows(), total=len(df), desc='Training '):
        selected_channel_id = row['channel_id']
        print("selected_channel_id: ", selected_channel_id)
        val_auc, test_auc = inference(df, train_data, validation_signal, val_label, test_signal, test_label,
                                      device, model, sigmoid,
                                      selected_channel_id=selected_channel_id)
        print("val_auc: ", val_auc)
        print("test_auc: ", test_auc)
        print("------------------------------------------------------")
        # Placeholder for results
        results = {'channel_id': selected_channel_id,
                   'val_auc': val_auc,
                   'test_auc': test_auc,
                   'experiment_name': 'FETCH',
                   'model_name': 'model_{}nodes'.format(num_nodes),
                   'number_nodes': num_nodes}
        # Concatenate the results
        results_df = pd.concat([results_df, pd.DataFrame(results)], ignore_index=True)

    # Save the results
    results_df.to_csv(os.path.join(root_path, 'model_{}nodes'.format(num_nodes), 'results.csv'), index=False)


if __name__ == '__main__':
    num_nodes = 2
    test_models_with_nodes(num_nodes)
