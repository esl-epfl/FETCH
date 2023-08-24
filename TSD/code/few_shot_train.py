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
from TSD.code.tuh_dataset import get_data_loader
from TSD.few_shot.support_set_const import seizure_support_set, non_seizure_support_set
from TSD.code.utils import thresh_max_f1
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score

from tqdm import tqdm
import numpy as np
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
    ret = get_data_loader(batch_size=2*(opt.num_support_tr + opt.num_query_tr), save_dir=opt.data_root,
                          return_dataset=True, masking=False)
    tr_dataset, val_dataset, test_dataset, tr_label, val_label, test_label = ret

    tr_sampler = init_sampler(opt, tr_label, mode="train")
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


def get_mask(selected_channel_id=-1):
    MASK = np.ones(20, dtype=np.bool)

    if selected_channel_id == -1:
        # Create a list of indices
        indices = np.arange(20)
        # Randomly shuffle the indices
        indices = indices[np.random.permutation(20)]

        # Select the first 8 indices and assign 0 to the corresponding MASK elements
        MASK[indices[:8]] = 0
    else:
        with open("../feasible_channels/feasible_8edges.json", 'r') as json_file:
            all_feasible_channel_combination = json.load(json_file)
        present_channels = all_feasible_channel_combination[selected_channel_id]
        MASK[present_channels] = 0

    return MASK


def get_support_set():
    support_set = []
    labels = []
    for label, class_support_set in enumerate([non_seizure_support_set, seizure_support_set]):
        for filename in class_support_set:
            filepath = os.path.join("../../TUSZv2/preprocess/task-binary_datatype-train_STFT/",
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

    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')

    x_support_set, y_support_set = get_support_set()
    x_support_set = torch.tensor(x_support_set).to(device)
    y_support_set = torch.tensor(y_support_set).to(device)

    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x_query_set, y_query_set = batch
            x_query_set, y_query_set = x_query_set.to(device), y_query_set.to(device)

            x = torch.concatenate((x_support_set, x_query_set))
            y = torch.concatenate((y_support_set, y_query_set))

            mask = get_mask()
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

            x = torch.concatenate((x_support_set, x_query_set))
            y = torch.concatenate((y_support_set, y_query_set))

            mask = get_mask()
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


def test(opt, test_dataloader, val_dataloader, model):
    """
    Test the model trained with the prototypical learning algorithm
    """
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    start_time = time.time()

    model.eval()

    x_support_set, y_support_set = get_support_set()
    x_support_set = torch.tensor(x_support_set).to(device)
    y_support_set = torch.tensor(y_support_set).to(device)

    mask = get_mask(selected_channel_id=opt.selected_channel_id)
    x_support_set[:, mask, :, :] = -1  # mask the channels
    x = x_support_set.reshape((x_support_set.shape[0], 1, -1, x_support_set.shape[3]))
    model_output = model(x)

    prototypes = get_prototypes(model_output, target=y_support_set).to(device)
    mask = get_mask(selected_channel_id=opt.selected_channel_id)

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
    best_th = thresh_max_f1(val_label_all, val_prob_all)
    print("Best Threshold", best_th)
    validation_time = time.time() - start_time

    predict_prob = torch.zeros(len(test_dataloader.dataset), dtype=torch.float32).to(device)
    true_label = torch.zeros(len(test_dataloader.dataset), dtype=torch.int).to(device)

    mask = get_mask(selected_channel_id=opt.selected_channel_id)

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
    test_predict_all = np.where(predict_prob > best_th, 1, 0)
    test_time = time.time() - start_time - validation_time

    with open("../feasible_channels/feasible_8edges.json", 'r') as json_file:
        selected_channels = json.load(json_file)[opt.selected_channel_id]
    # Placeholder for results
    results = {
        "selected_channel_id": opt.selected_channel_id,
        "selected_channels": selected_channels,
        "best_threshold": best_th,
        "accuracy": accuracy_score(true_label, test_predict_all),
        "f1_score": f1_score(true_label, test_predict_all),
        "auc": roc_auc_score(true_label, predict_prob),
        "val_auc": roc_auc_score(val_label_all, val_prob_all),
        "validation_time": validation_time,
        "test_time": test_time,
        "confusion_matrix": confusion_matrix(true_label, test_predict_all).tolist()
    }

    # Save results to a JSON file
    output_filename = "../results/results_channel_adaptation_{}.json".format(opt.selected_channel_id)
    with open(output_filename, "w") as json_file:
        json.dump(results, json_file, indent=4)

    return


def eval():
    """
    Initialize everything and train
    """
    options = get_parser().parse_args()

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    tr_dataloader, val_dataloader, test_dataloader = init_dataloader(options, full_validation=True)
    model = init_vit(options)
    model_path = os.path.join(options.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))

    test(opt=options,
         test_dataloader=test_dataloader,
         val_dataloader=val_dataloader,
         model=model)



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
