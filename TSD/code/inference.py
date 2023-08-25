import os
import sys
import math
import json
import warnings

# Filter out the specific UserWarning related to torchvision
warnings.filterwarnings("ignore", category=UserWarning, message="Failed to load image Python extension")
# TODO solve the CUDA version issue

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import time
import torch
from torch import nn
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import numpy as np
from tqdm import tqdm
from tuh_dataset import get_data_loader
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from sklearn.metrics import confusion_matrix
from tuh_dataset import args as tuh_args
from TSD.code.utils import thresh_max_f1


def seed_everything(seed=99):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=99)


def test_sample_base(selected_channel_id=tuh_args.selected_channel_id):
    start_time = time.time()
    save_directory = tuh_args.save_directory
    _, val_loader, test_loader = get_data_loader(2048, save_directory,
                                                 selected_channel_id=selected_channel_id,
                                                 event_base=False, remove_not_used=False)

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
    best_th = thresh_max_f1(val_label_all, val_prob_all)
    validation_time = time.time() - start_time

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

    test_predict_all = np.where(test_prob_all > best_th, 1, 0)
    test_time = time.time() - start_time - validation_time

    with open("../feasible_channels/feasible_8edges.json", 'r') as json_file:
        selected_channels = json.load(json_file)[selected_channel_id]
    # Placeholder for results
    results = {
        "selected_channel_id": selected_channel_id,
        "selected_channels": selected_channels,
        "best_threshold": best_th,
        "accuracy": accuracy_score(test_label_all, test_predict_all),
        "f1_score": f1_score(test_label_all, test_predict_all),
        "auc": roc_auc_score(test_label_all, test_prob_all),
        "val_auc": roc_auc_score(val_label_all, val_prob_all),
        "validation_time": validation_time,
        "test_time": test_time,
        "confusion_matrix": confusion_matrix(test_label_all, test_predict_all).tolist()
    }

    # Save results to a JSON file
    output_filename = "../results/results_{}_{}.json".format("global" if tuh_args.global_model
                                                                              else "Channel_specific",
                                                                              selected_channel_id)
    with open(output_filename, "w") as json_file:
        json.dump(results, json_file, indent=4)


def extract_epoch(filename):
    return int(filename.split('_')[1])


def get_highest_epoch_file(files):
    if not files:
        return None

    highest_epoch = max(extract_epoch(file) for file in files)
    highest_epoch_files = [file for file in files if extract_epoch(file) == highest_epoch]

    return highest_epoch, highest_epoch_files


sample_rate = 256
eeg_type = 'stft'  # 'original', 'bipolar', 'stft'
device = 'cuda:0'
# device = 'cpu'

# model = torch.load('inference_ck_0.9208', map_location=torch.device(device))
root_path = tuh_args.save_directory
if tuh_args.global_model:  # Global model
    model_path = os.path.join(root_path, 'test_STFT8/model_17_0.9159300189983679')
    print("Global Model ", model_path)

else:  # Channel_specific model
    folder_path = os.path.join(root_path, 'test_8ch_{}'.format(tuh_args.selected_channel_id))

    # Check if the folder exists. It means if we have trained a specific model
    if os.path.exists(folder_path):
        all_files = os.listdir(folder_path)

        # Filter files based on the pattern
        matching_files = [file for file in all_files if file.startswith("model")]
        if len(matching_files) > 0:
            # Get the highest epoch and corresponding files
            highest_epoch, highest_epoch_files = get_highest_epoch_file(matching_files)

            model_path = os.path.join(folder_path, highest_epoch_files[0])
            print("Channel-specific Model ", model_path)
        else:
            print("Channel-specific Folder exists but is empty.")
            sys.exit(2)  # Exit with an error status code 2 to show that Channel-specific model does not exist
    else:
        print("Channel-specific Folder does not exist")
        sys.exit(2)  # Exit with an error status code 2 to show that Channel-specific model does not exist


model = torch.load(model_path,  map_location=torch.device(device))
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
model.eval()
sigmoid = nn.Sigmoid()

for i in range(0, 10):
    test_sample_base(i)