import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import warnings

# Filter out the specific UserWarning related to torchvision
warnings.filterwarnings("ignore", category=UserWarning, message="Failed to load image Python extension")
# TODO solve the CUDA version issue

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import json
from channel_possibility import double_banana
from utils import create_dataframe

import tuh_dataset
from vit_pytorch.vit import ViT
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

print(f"Torch: {torch.__version__}")

PATIENCE_EARLY_STOPPING = 10
VAL_EVERY = 1
EPOCHS = 100
GENERAL_MODEL = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")


def seed_everything(seed=99):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train(model_path=None, selected_channel_id=tuh_dataset.args.selected_channel_id):
    seed_everything()

    if model_path is None:
        with open("../feasible_channels/feasible_20edges.json", 'r') as json_file:
            channel_set = json.load(json_file)[selected_channel_id]  # Get all feasible channels and take
            # num of channels based on selected_channel_id
            num_channels = len(channel_set)
        print(f"Number of channels: {num_channels}")
        model = ViT(image_size=(160*num_channels, 15), patch_size=(80, 5), num_classes=1,
                    dim=16, depth=4, heads=4, mlp_dim=4, pool='cls',
                    channels=1, dim_head=4, dropout=0.2, emb_dropout=0.2).to(device)
    else:
        # root_path = '/home/amirshah/EPFL/EpilepsyTransformer/TUSZv2/preprocess'
        # model_path = os.path.join(root_path, 'pretrain_full_channel/model_12_0.9333263693136304')
        # model_path = os.path.join(root_path, 'test_STFT8/model_17_0.9159300189983679')
        model = torch.load(model_path, map_location=torch.device(device))
    sigmoid = nn.Sigmoid()

    # Training settings
    batch_size = 256
    lr = 3e-5
    gamma = 0.7
    tuh_dataset.args.eeg_type = 'stft'

    # loss function
    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([16]).to(device))
    criterion = nn.BCEWithLogitsLoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    # load data
    (train_data, _, _,
     train_signal, train_label,
     validation_signal, val_label,
     test_signal, test_label) = \
        tuh_dataset.get_data(save_dir=tuh_dataset.args.save_directory,
                             balanced_data=True,
                             return_val_test_signal=True,
                             return_train_signal=tuh_dataset.args.server)

    train_loader, val_loader, test_loader = \
        tuh_dataset.get_dataloader(train_data=None if tuh_dataset.args.server else train_data,
                                   val_data=None, test_data=None,
                                   train_signal=train_signal if tuh_dataset.args.server else None,
                                   train_label=train_label,
                                   validation_signal=validation_signal, val_label=val_label,
                                   test_signal=test_signal, test_label=test_label,
                                   batch_size=batch_size,
                                   selected_channel_id=selected_channel_id,
                                   return_dataset=False,
                                   event_base=False, masking=False, random_mask=False, remove_not_used=True)

    best_val_auc = 0.0
    best_val_epoch = 0
    model_directory = os.path.join(tuh_dataset.args.save_directory,
                                   'test_scratch_{}'.format(selected_channel_id))
    if not os.path.exists(model_directory):
        os.mkdir(model_directory)

    for epoch in range(EPOCHS):

        model.train()
        train_label_all = []
        train_prob_all = []
        epoch_train_loss = 0
        train_auc = 0
        if epoch != 0:
            for data, label in tqdm(train_loader, desc='Training '):
                train_label_all.extend(label)

                data = data.to(device)
                label = label.to(device).float()

                prob = model(data)
                prob = torch.squeeze(prob)
                train_prob_all.extend(prob.cpu().detach().numpy())

                loss = criterion(prob, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss / len(train_loader)

            train_auc = roc_auc_score(train_label_all, train_prob_all)

            if epoch % VAL_EVERY != VAL_EVERY - 1:
                continue

        model.eval()
        val_label_all = []
        val_prob_all = []
        epoch_val_loss = 0
        with torch.no_grad():
            for data, label in tqdm(val_loader, desc='Evaluation '):
                val_label_all.extend(label)

                data = data.to(device)
                label = label.to(device).float()

                val_prob = model(data)
                val_prob_sigmoid = torch.squeeze(sigmoid(val_prob))
                val_prob_all.extend(val_prob_sigmoid.cpu().numpy())

                val_loss = criterion(torch.squeeze(val_prob), label)

                epoch_val_loss += val_loss / len(val_loader)

        val_auc = roc_auc_score(val_label_all, val_prob_all)

        print(f"Epoch: {epoch + 1} - train_loss: {epoch_train_loss:.4f} -  train_auc: {train_auc:.4f}; "
              f"val_loss: {epoch_val_loss:.4f} - val_auc: {val_auc:.4f}")

        if best_val_auc < val_auc:
            best_val_auc = val_auc
            best_val_epoch = epoch
            torch.save(model, os.path.join(model_directory, 'model_{}_{}'.format(epoch, val_auc)))

            if not GENERAL_MODEL:
                model.eval()
                test_label_all = []
                test_prob_all = []
                epoch_test_loss = 0
                with torch.no_grad():
                    for data, label in tqdm(test_loader, desc='Testing '):
                        test_label_all.extend(label)

                        data = data.to(device)
                        label = label.to(device).float()

                        test_prob = model(data)
                        test_prob_sigmoid = torch.squeeze(sigmoid(test_prob))
                        test_prob_all.extend(test_prob_sigmoid.cpu().numpy())

                        test_loss = criterion(torch.squeeze(test_prob), label)

                        epoch_test_loss += test_loss / len(test_loader)

                test_auc = roc_auc_score(test_label_all, test_prob_all)
                print(f"test_loss: {epoch_test_loss:.4f} - test_auc: {test_auc:.4f}")

        if epoch > best_val_epoch + PATIENCE_EARLY_STOPPING:
            torch.save(model, os.path.join(model_directory, 'test_model_last_{}'.format(epoch)))
            break

    return best_val_auc


def inference(model_path,
              train_data,
              validation_signal, val_label,
              test_signal, test_label,
              selected_channel_id=tuh_dataset.args.selected_channel_id,
             ):
    seed_everything()
    model = torch.load(model_path, map_location=torch.device(device))
    sigmoid = nn.Sigmoid()

    # Training settings
    batch_size = 256
    lr = 3e-5
    gamma = 0.7
    tuh_dataset.args.eeg_type = 'stft'

    _, val_loader, test_loader = \
        tuh_dataset.get_dataloader(train_data=train_data,
                                   train_signal=None, train_label=None,
                                   val_data=None, test_data=None,
                                   validation_signal=validation_signal, val_label=val_label,
                                   test_signal=test_signal, test_label=test_label,
                                   batch_size=batch_size,
                                   selected_channel_id=selected_channel_id,
                                   return_dataset=False,
                                   event_base=False, masking=False, random_mask=False,
                                   remove_not_used=True)

    model.eval()
    val_label_all = []
    val_prob_all = []
    with torch.no_grad():
        for data, label in tqdm(val_loader, desc='Evaluation '):
            val_label_all.extend(label)

            data = data.to(device)
            val_prob = model(data)
            val_prob_sigmoid = torch.squeeze(sigmoid(val_prob))
            val_prob_all.extend(val_prob_sigmoid.cpu().numpy())

    val_auc = roc_auc_score(val_label_all, val_prob_all)

    print(f"val_auc: {val_auc:.4f}")

    test_label_all = []
    test_prob_all = []
    with torch.no_grad():
        for data, label in tqdm(test_loader, desc='Testing '):
            test_label_all.extend(label)

            data = data.to(device)
            test_prob = model(data)
            test_prob_sigmoid = torch.squeeze(sigmoid(test_prob))
            test_prob_all.extend(test_prob_sigmoid.cpu().numpy())

    test_auc = roc_auc_score(test_label_all, test_prob_all)
    print(f"test_auc: {test_auc:.4f}")
    return val_auc, test_auc


def get_feasible_ids_with_num_nodes(num_nodes):
    def channel_list_to_node_set(x):
        node_set = set()
        edge_lists = [double_banana[a] for a in x]
        for node1, node2 in edge_lists:
            node_set.add(node1)
            node_set.add(node2)
        return len(node_set)
    df = create_dataframe(20)
    df['number_nodes'] = df['channel_list'].apply(channel_list_to_node_set)
    df_num_nodes = df[df['number_nodes'] == num_nodes]
    return df_num_nodes['channel_id'].tolist()


def train_scratch_models():
    channel_ids = get_feasible_ids_with_num_nodes(4)
    print(len(channel_ids))
    # random permutation of channel_ids
    random.shuffle(channel_ids)
    for channel_id in channel_ids[:100]:
        train(selected_channel_id=channel_id, model_path=None)


def inference_scratch_models():
    # get data
    (train_data, _, _, _, _,
     validation_signal, val_label,
     test_signal, test_label) = \
        tuh_dataset.get_data(save_dir=tuh_dataset.args.save_directory,
                             balanced_data=True,
                             return_val_test_signal=True,
                             return_train_signal=False)

    save_directory = tuh_dataset.args.save_directory
    # Read all the directories in save_directory
    model_dirs = [os.path.join(save_directory, d) for d in os.listdir(save_directory)
                  if os.path.isdir(os.path.join(save_directory, d))]
    # filter the directories based on their name test_x_y
    # x is the experiment name, and y is the channel_id
    # x can be SBS, SFS, scratch, or NSGA2

    # Create the dataframe
    df = create_dataframe(20)
    # add three columns to the dataframe as Validation AUC, and Test AUC, and experiment name
    # The experiment name is a list that can have several experiments
    df['val_auc'] = np.nan
    df['test_auc'] = np.nan
    df['experiment_name'] = np.nan

    # iterate over the model directories
    for model_dir in model_dirs:
        if len(model_dir.split('/')[-1].split('_')) != 3:
            continue
        # get the experiment name and channel_id
        experiment_name = model_dir.split('/')[-1].split('_')[1]
        if experiment_name not in ['scratch', 'SBS', 'SFS', 'NSGA']:
            continue
        channel_id = model_dir.split('/')[-1].split('_')[2]

        # get the model path for the best model
        # The model name is similar to model_x_0.7952, where x is the epoch number
        # We need to find the largest number of x as the best model
        model_names = [name for name in os.listdir(model_dir) if name.startswith('model')]
        # get the epoch number
        epoch_numbers = [int(name.split('_')[1]) for name in model_names]
        # get the best epoch number
        best_epoch_number = max(epoch_numbers)
        # get the complete model name based on the best epoch number
        model_name = [name for name in model_names if int(name.split('_')[1]) == best_epoch_number][0]
        best_model_path = os.path.join(model_dir, model_name)
        # get the validation and test AUC
        # val_auc, test_auc = inference(model_path=best_model_path,
        #                               selected_channel_id=int(channel_id))
        val_auc, test_auc = inference(model_path=best_model_path,
                                      train_data=train_data,
                                      validation_signal=validation_signal, val_label=val_label,
                                      test_signal=test_signal, test_label=test_label,
                                      selected_channel_id=int(channel_id))
        # update the dataframe
        df.loc[df['channel_id'] == int(channel_id), 'val_auc'] = val_auc
        df.loc[df['channel_id'] == int(channel_id), 'test_auc'] = test_auc
        # If the experiment name is nan for this specific row, create a list and add the experiment name
        # Otherwise, append the experiment name to the list
        if np.isnan(df.loc[df['channel_id'] == int(channel_id), 'experiment_name'].values[0]):
            df.loc[df['channel_id'] == int(channel_id), 'experiment_name'] = [experiment_name]
        else:
            df.loc[df['channel_id'] == int(channel_id), 'experiment_name'] += [experiment_name]

    # save the updated df
    df.to_csv(os.path.join(save_directory, 'scratch_models.csv'))


if __name__ == '__main__':
    inference_scratch_models()
