# coding=utf-8
import json
import time
import warnings

# Filter out the specific UserWarning related to torchvision
warnings.filterwarnings("ignore", category=UserWarning, message="Failed to load image Python extension")
# TODO solve the CUDA version issue

from TSD.few_shot.prototypical_batch_sampler import PrototypicalBatchSampler
from TSD.few_shot.prototypical_loss import prototypical_loss as loss_fn
from TSD.few_shot.prototypical_loss import get_prototypes, prototypical_evaluation
from TSD.code.parser_util import get_parser
from TSD.code.tuh_dataset import get_data, get_dataloader
from TSD.few_shot.support_set_const import seizure_support_set, non_seizure_support_set
from TSD.code.utils import thresh_max_f1
from TSD.code.utils import create_dataframe, channel_list_to_node_set
from TSD.code.utils import get_feasible_ids_with_num_nodes
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score

from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import os
from vit_pytorch.vit import ViT
import pickle


def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


def init_sampler(opt, labels, mode):
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_query_tr
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_query_val

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=opt.iterations)


def init_dataloader(opt, full_validation=False):
    # load data
    (train_data, _, _, train_signal, train_label,
     validation_signal, val_label, test_signal, test_label) = \
        get_data(save_dir=opt.save_directory,
                 balanced_data=True,
                 return_val_test_signal=True,
                 return_train_signal=opt.server)  # if we use the server, we have enough memory

    tr_dataset, val_dataset, test_dataset = \
        get_dataloader(train_data=None if opt.server else train_data,
                       val_data=None, test_data=None,
                       train_signal=train_signal if opt.server else None,
                       train_label=train_label,
                       validation_signal=validation_signal, val_label=val_label,
                       test_signal=test_signal, test_label=test_label,
                       batch_size=2 * (opt.num_support_tr + opt.num_query_tr),
                       selected_channel_id=opt.selected_channel_id,
                       return_dataset=True,
                       event_base=False, masking=False,
                       random_mask=False, remove_not_used=False)

    tr_sampler = init_sampler(opt, train_label, mode="train")
    tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_sampler=tr_sampler, num_workers=6)
    if full_validation:
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2048, num_workers=6)
    else:
        val_sampler = init_sampler(opt, val_label, mode="val")
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=6)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=2048, num_workers=6)

    return tr_dataloader, val_dataloader, test_dataloader


def init_vit(opt):
    """
    Initialize the ViT
    """
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    model = ViT(image_size=(3200, 15), patch_size=(80, 5), num_classes=16, dim=16, depth=4, heads=4, mlp_dim=4,
                pool='cls',
                channels=1, dim_head=4, dropout=0.2, emb_dropout=0.2).to(device)
    return model


def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)


def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def get_mask(df=None, selected_channel_id=-1):
    MASK = np.ones(20, dtype=np.int32)

    if selected_channel_id == -1:
        if df is None:
            # Create a list of indices
            indices = np.arange(20)
            # Randomly shuffle the indices
            indices = indices[np.random.permutation(20)]

            # Select the first 8 indices and assign 0 to the corresponding MASK elements
            MASK[indices[:8]] = 0
        else:
            # Choose a random row from the dataframe
            # Take channel_list from that row
            df_sample = df.sample(n=1)
            channel_list = df_sample['channel_list'].values[0]
            MASK[channel_list] = 0

    else:
        if df is None:
            with open("../feasible_channels/feasible_20edges.json", 'r') as json_file:
                all_feasible_channel_combination = json.load(json_file)
            present_channels = all_feasible_channel_combination[selected_channel_id]
            MASK[present_channels] = 0
        else:
            df_sample = df[df['channel_id'] == selected_channel_id]
            channel_list = df_sample['channel_list'].values[0]
            MASK[channel_list] = 0

    return MASK


def get_support_set(opt):
    support_set = []
    labels = []
    for label, class_support_set in enumerate([non_seizure_support_set, seizure_support_set]):
        for filename in class_support_set:
            filepath = os.path.join(opt.save_directory,
                                    "task-binary_datatype-train_STFT/",
                                    filename + ".pkl")
            with open(filepath, 'rb') as f:
                data_pkl = pickle.load(f)
                signals = np.asarray(data_pkl['STFT'])
                support_set.append(signals)
                labels.append(label)

    return np.array(support_set), np.array(labels)


def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):
    '''
    Train the model with the prototypical learning algorithm
    '''

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    print("Device", device)

    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')
    best_state = model.state_dict()

    x_support_set, y_support_set = get_support_set(opt)
    x_support_set = torch.tensor(x_support_set).to(device)
    y_support_set = torch.tensor(y_support_set).to(device)

    df = create_dataframe(20)
    df['number_nodes'] = df['channel_list'].apply(channel_list_to_node_set)
    df_num_nodes = df[df['number_nodes'] == opt.num_nodes]
    print("df_num_nodes", df_num_nodes.sample(n=5))

    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x_query_set, y_query_set = batch
            x_query_set, y_query_set = x_query_set.to(device), y_query_set.to(device)

            # concatenate the support set and query set
            x = torch.cat((x_support_set, x_query_set))
            y = torch.cat((y_support_set, y_query_set))

            mask = get_mask(df=df_num_nodes)
            x[:, mask, :, :] = -1  # mask the channels
            x = x.reshape((x.shape[0], 1, -1, x.shape[3]))

            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=opt.num_support_tr)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
        avg_loss = np.mean(train_loss[-opt.iterations:])
        avg_acc = np.mean(train_acc[-opt.iterations:])
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        lr_scheduler.step()
        if val_dataloader is None:
            continue
        val_iter = iter(val_dataloader)
        model.eval()
        for batch in val_iter:
            x_query_set, y_query_set = batch
            x_query_set, y_query_set = x_query_set.to(device), y_query_set.to(device)

            x = torch.cat((x_support_set, x_query_set))
            y = torch.cat((y_support_set, y_query_set))

            mask = get_mask(df=df_num_nodes)
            x[:, mask, :, :] = -1  # mask the channels
            x = x.reshape((x.shape[0], 1, -1, x.shape[3]))

            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=opt.num_support_val)
            val_loss.append(loss.item())
            val_acc.append(acc.item())
        avg_loss = np.mean(val_loss[-opt.iterations:])
        avg_acc = np.mean(val_acc[-opt.iterations:])
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
            avg_loss, avg_acc, postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()

    torch.save(model.state_dict(), last_model_path)

    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join(opt.experiment_root,
                                       name + '.txt'), locals()[name])

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def test(opt, test_dataloader, val_dataloader, model, device,
         df, selected_channel_id):
    """
    Test the model trained with the prototypical learning algorithm
    """

    model.eval()

    x_support_set, y_support_set = get_support_set(opt)  # TODO: move this to the main function
    x_support_set = torch.tensor(x_support_set).to(device)
    y_support_set = torch.tensor(y_support_set).to(device)

    mask = get_mask(df=df, selected_channel_id=selected_channel_id)
    x_support_set[:, mask, :, :] = -1  # mask the channels
    x = x_support_set.reshape((x_support_set.shape[0], 1, -1, x_support_set.shape[3]))
    model_output = model(x)

    prototypes = get_prototypes(model_output, target=y_support_set).to(device)

    val_prob_all = torch.zeros(len(val_dataloader.dataset), dtype=torch.float32).to(device)
    val_label_all = torch.zeros(len(val_dataloader.dataset), dtype=torch.int).to(device)

    for i, batch in enumerate(tqdm(val_dataloader)):
        x, y = batch
        x, y = x.to(device), y.to(device)
        x[:, mask, :, :] = -1  # mask the channels
        x = x.reshape((x.shape[0], 1, -1, x.shape[3]))
        model_output = model(x)
        prob, _ = prototypical_evaluation(prototypes, model_output)

        start_idx = i * val_dataloader.batch_size
        end_idx = start_idx + x.size(0)

        val_prob_all[start_idx:end_idx] = prob.detach()
        val_label_all[start_idx:end_idx] = y.detach()

    val_label_all = val_label_all.cpu().numpy()
    val_prob_all = val_prob_all.cpu().numpy()
    val_auc = roc_auc_score(val_label_all, val_prob_all)

    predict_prob = torch.zeros(len(test_dataloader.dataset), dtype=torch.float32).to(device)
    true_label = torch.zeros(len(test_dataloader.dataset), dtype=torch.int).to(device)

    for i, batch in enumerate(tqdm(test_dataloader)):
        x, y = batch
        x, y = x.to(device), y.to(device)

        x[:, mask, :, :] = -1  # mask the channels
        x = x.reshape((x.shape[0], 1, -1, x.shape[3]))
        model_output = model(x)
        prob, _ = prototypical_evaluation(prototypes, model_output)

        start_idx = i * test_dataloader.batch_size
        end_idx = start_idx + x.size(0)

        predict_prob[start_idx:end_idx] = prob.detach()
        true_label[start_idx:end_idx] = y.detach()

    predict_prob = predict_prob.cpu().numpy()
    true_label = true_label.cpu().numpy()
    test_auc = roc_auc_score(true_label, predict_prob)

    return val_auc, test_auc


def eval():
    """
    Initialize everything and train
    """
    options = get_parser().parse_args()
    num_nodes = options.num_nodes
    df = get_feasible_ids_with_num_nodes(num_nodes)

    # Create a dataframe to store the results
    results_df = pd.DataFrame(columns=['channel_id', 'val_auc', 'test_auc',
                                       'experiment_name', 'model_name', 'number_nodes'])

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    tr_dataloader, val_dataloader, test_dataloader = init_dataloader(options, full_validation=True)
    model = init_vit(options)
    model_path = os.path.join(options.experiment_root, 'model_{}nodes'.format(num_nodes),
                              'best_model.pth')
    model.load_state_dict(torch.load(model_path))

    device = 'cuda:0' if torch.cuda.is_available() and options.cuda else 'cpu'
    print("Device", device)

    # iterate over all rows of the dataframe
    for index, row in df.iterrows():
        selected_channel_id = row['channel_id']
        selected_channels = row['channel_list']

        val_auc, test_auc = test(opt=options,
                                 test_dataloader=test_dataloader,
                                 val_dataloader=val_dataloader,
                                 model=model,
                                 df=df,
                                 device=device,
                                 selected_channel_id=selected_channel_id, )
        print("val_auc: ", val_auc)
        print("test_auc: ", test_auc)
        print("------------------------------------------------------")
        results = {'channel_id': selected_channel_id,
                   'val_auc': val_auc,
                   'test_auc': test_auc,
                   'experiment_name': 'FETCH',
                   'model_name': 'model_{}nodes'.format(num_nodes),
                   'number_nodes': num_nodes}
        # Concatenate the results
        results_df = pd.concat([results_df, pd.DataFrame(results)], ignore_index=True)

        # Save the results
    results_df.to_csv(os.path.join(options.experiment_root,
                                   'model_{}nodes'.format(num_nodes),
                                   'results.csv'), index=False)


def main():
    """
    Initialize everything and train
    """
    options = get_parser().parse_args()
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)

    tr_dataloader, val_dataloader, test_dataloader = init_dataloader(options)

    model = init_vit(options)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    res = train(opt=options,
                tr_dataloader=tr_dataloader,
                val_dataloader=val_dataloader,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler)
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res


if __name__ == '__main__':
    # main()
    eval()
