import math

import numpy as np
import pickle
import seaborn as sns
from utils.BioT import BioTransformer, Epilepsy60Dataset, ImbalancedDataSampler, EvaluateSampler, \
    PatientDiscriminatorDataset, PatientDiscriminatorEvaluationDataset, TSD, EpilepsyTSD
from utils.data import get_data
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.optim import SGD, Adam, AdamW
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
from utils.BioT import SEQ_LEN, SEGMENT, ROI
from utils.params import feature_noise_threshold
from utils.metrics import thresh_max_f1
from vit_pytorch.vit import ViT


def train(model, device, save_path: str, learning_rate: float = 1e-5, params_lr=None):

    # %%
    # mode = "train"
    # seizure_indices = np.where(labels[mode] == 1)[0]
    # non_seizure_indices = np.where(labels[mode] == 0)[0]
    # post_ictal_indices = []
    # ictal_post_ictal_indices = []
    # seizure_end_points = np.where(np.diff(labels[mode][:, 0]).astype(np.int) == -1)[0]
    # for post_ictal in seizure_end_points:
    #     for post_time in range(1, SEQ_LEN):
    #         if post_ictal + post_time >= len(sample_time[mode]) or \
    #                 sample_time[mode][post_ictal + post_time] == 0 or \
    #                 labels[mode][post_ictal + post_time] == 1:
    #             break
    #         if post_time <= ROI:
    #             ictal_post_ictal_indices.append(post_ictal + post_time)
    #         else:
    #             post_ictal_indices.append(post_ictal + post_time)

    # post_ictal_indices = np.array(post_ictal_indices)
    # ictal_post_ictal_indices = np.array(ictal_post_ictal_indices)
    # non_seizure_indices = np.setdiff1d(non_seizure_indices, post_ictal_indices, assume_unique=True)
    # non_seizure_indices = np.setdiff1d(non_seizure_indices, ictal_post_ictal_indices, assume_unique=True)
    # print("{}: {} seizures, {} seizure points, {} non_seizure points and {} post ictal points "
    #       .format(mode, len(seizure_end_points),
    #               len(seizure_indices),
    #               len(non_seizure_indices),
    #               len(post_ictal_indices)))

    # X_train = X["train"]
    # X_val = X["val"]
    # Y = labels["train"]
    # Y_val = labels["val"]
    # sample_time_train = sample_time["train"]
    # sample_time_val = sample_time["val"]

    # sampler = ImbalancedDataSampler(torch.from_numpy(seizure_indices).long(),
    #                                 torch.from_numpy(non_seizure_indices).long(),
    #                                 torch.from_numpy(post_ictal_indices).long(),
    #                                 post_non_ratio=0.0,
    #                                 overlap=1)

    val_set = EpilepsyTSD(mode="val")
    val_loader = DataLoader(val_set, shuffle=False, batch_size=32)

    # Training loop
    if params_lr == None:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(params_lr, lr=learning_rate)
    # lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
    lr_sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    target_lr = learning_rate

    def set_lr(new_lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = (new_lr / target_lr) * param_group['initial_lr']

    def schedule_lr(iteration):
        if iteration <= 5000:
            set_lr(iteration * target_lr / 5000)

    criterion = BCEWithLogitsLoss()
    N_EPOCHS = 1000
    train_loss_list = []
    val_loss_list = []
    best_f1 = {"value": 0, "epoch": 0, "val_loss": 1000, "auc": 0}
    prev_val_loss = 1000
    patience_count = 0
    for epoch in tqdm(range(N_EPOCHS), desc="Training"):
        train_set = EpilepsyTSD(mode="train")
        train_loader = DataLoader(train_set, batch_size=256, num_workers=4)
        model.train(True)  # turn on train mode
        train_loss = 0.0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}", position=0, leave=True)):
            optimizer.zero_grad()
            # schedule_lr(epoch * len(train_loader) + i)
            x, y = batch['x'], batch['y']
            x = x[:, np.newaxis, :, :]
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat.view(-1,), y.view(-1, ))

            train_loss += loss.detach().cpu().item()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        if epoch % 10 != 9:
            continue

        model.eval()
        with torch.no_grad():
            test_predict = []
            test_labels = []
            running_vloss = 0.0
            for i, batch in enumerate(val_loader):
                x, y = batch['x'], batch['y']
                x = x[:, np.newaxis, :, :]
                x, y = x.to(device), y.to(device)
                voutputs = model(x)
                test_labels += y.view(-1, ).tolist()
                vloss = criterion(voutputs.view(-1,), y.view(-1, ))
                running_vloss += vloss.detach().cpu().item()

                test_predict += torch.sigmoid(voutputs.view(-1,)).tolist()

            avg_vloss = running_vloss / len(val_loader)

            auroc = roc_auc_score(y_true=test_labels, y_score=test_predict)
            best_thresh = thresh_max_f1(y_true=test_labels, y_prob=test_predict)
            test_predict = (np.array(test_predict) > best_thresh) * 1.0
            f1_val = f1_score(test_labels, test_predict)
            print(
                "Best Threshold: {:.2f} -> F1-score: {:.3f}\nAUROC: {:.3f}\nValidation LOSS: {:.2f}".format(best_thresh,
                                                                                                            f1_val,
                                                                                                            auroc,
                                                                                                            avg_vloss))
            val_loss_list.append(avg_vloss)
            if auroc > best_f1["auc"]:
                best_f1["value"] = f1_val
                best_f1["epoch"] = epoch
                best_f1["val_loss"] = avg_vloss
                best_f1["auc"] = auroc
                print("BEST F1!")
                torch.save(model.state_dict(), "{}_best".format(save_path))

        lr_sched.step()
        lr = lr_sched.get_last_lr()[0]
        #
        print(f"Epoch {epoch + 1}/{N_EPOCHS} Training loss: {train_loss / len(train_loader):.2f} Learning Rate {lr} ")
        train_loss_list.append(train_loss / len(train_loader))

        if avg_vloss < prev_val_loss:
            patience_count = 0
        else:
            patience_count += 1
        prev_val_loss = avg_vloss
        if epoch > best_f1["epoch"] + 100 and patience_count > 3:
            break

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
    X, labels, minute_labels, pat_file_start_end, sample_time, valid_labels = get_data(pretrain_mode=True,
                                                                                       dataset=dataset)
    d_feature = 252 if dataset == "TUSZ" else 144
    d_model = 256
    n_heads = 4
    d_hid = 4 * d_model
    seq_len = SEQ_LEN + 3
    segment = SEGMENT
    n_layers = 8
    n_out = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('device : ', device)
    model = BioTransformer(d_feature=d_feature, d_model=d_model, n_heads=n_heads, d_hid=d_hid, seq_len=seq_len,
                           n_layers=n_layers, n_out=n_out, device=device, segments=segment).to(device)
    # model.load_state_dict(torch.load("../output/fake_model{}_n{}_{}".format(SEQ_LEN, n_layers, dataset)))

    X_train = X["train"]

    pat_start_end = get_pat_start_end(pat_file_start_end)

    train_set = PatientDiscriminatorDataset(torch.from_numpy(X_train).float(), pat_start_end['train'],
                                            sample_time['train'])
    sampler = EvaluateSampler(torch.from_numpy(valid_labels['train']).int(), overlap=20)
    train_loader = DataLoader(train_set, batch_size=32, sampler=sampler)

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
    learning_rate = 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    lr_sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    def set_lr(new_lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

    target_lr = learning_rate

    def schedule_lr(iteration):
        iteration = iteration + 1
        if iteration <= 20000:
            set_lr(iteration * target_lr / 20000)

    criterion = BCEWithLogitsLoss()
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)
    N_EPOCHS = 200
    val_loss_list = []
    save_path = '../output/pretrain_model{}_n{}_{}'.format(SEQ_LEN, n_layers, dataset)
    best_f1 = {"value": 0, "epoch": 0}
    for epoch in tqdm(range(N_EPOCHS), desc="Training"):
        model.train(True)  # turn on train mode
        train_loss = 0.0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}", position=0, leave=True)):
            optimizer.zero_grad()
            schedule_lr(epoch * len(train_loader) + i)
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
            test_predict = []
            test_labels = []
            running_vloss = 0.0
            for i, batch in enumerate(validation_loader):
                x, y = batch['x'], batch['y']
                x, y = x.to(device), y.to(device)
                x = torch.transpose(x, 0, 1)
                voutputs = model(x)
                test_predict += voutputs[-1, :, 0].tolist()
                test_labels += y.view(-1, ).tolist()
                vloss = criterion(voutputs[-1, :, 0], y.view(-1, ))
                running_vloss += vloss.detach().cpu().item()

            avg_vloss = running_vloss / len(validation_loader)
            best_thresh = thresh_max_f1(y_true=test_labels, y_prob=test_predict)
            test_predict = (np.array(test_predict) > best_thresh) * 1.0
            f1_val = f1_score(test_labels, test_predict)
            print("Best Threshold: {:.2f} -> F1-score: {:.3f}\nValidation LOSS: {:.2f}".format(best_thresh,
                                                                                               f1_val,
                                                                                               avg_vloss))
            val_loss_list.append(avg_vloss)
            if f1_val > best_f1["value"]:
                best_f1["value"] = f1_val
                best_f1["epoch"] = epoch
                print("BEST F1!")
                torch.save(model.state_dict(), "{}_best".format(save_path))
        if epoch > best_f1["epoch"] + 20:
            break

        lr_sched.step()
        lr = lr_sched.get_last_lr()[0]

        print(f"Epoch {epoch + 1}/{N_EPOCHS} Training loss: {train_loss / len(train_loader):.2f} LR {lr} ")

    torch.save(model.state_dict(), save_path)


def train_scratch(dataset):
    d_feature = 126 * 2 if dataset == "TUSZ" else 144
    d_model = 256
    n_heads = 4
    d_hid = 4 * d_model
    seq_len = SEQ_LEN + 3
    segment = SEGMENT
    n_layers = 12
    n_out = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device : ', device)
    model = BioTransformer(d_feature=d_feature, d_model=d_model, n_heads=n_heads, d_hid=d_hid, seq_len=seq_len,
                           n_layers=n_layers, n_out=n_out, device=device, segments=segment).to(device)
    savepath = '../output/model{}_{}_{}_preprocess_scratch'.format(SEQ_LEN, n_layers, dataset)
    train(model, device, savepath, learning_rate=1e-4)


def train_stft(dataset):
    d_feature = 350
    d_model = 128
    n_heads = 2
    seq_len = 85
    d_hid = 4 * d_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device : ', device)
    # model = TSD(d_feature=d_feature, d_model=d_model, n_heads=n_heads, d_hid=d_hid, device=device,
    #             seq_len=seq_len).to(device)
    model = ViT(image_size=(2100, 14), patch_size=(50, 7), num_classes=1,  dim=128, depth=2,
                heads=2, mlp_dim=512, pool='cls', channels=1,
                dim_head=64, dropout=0.2, emb_dropout=0.2).to(device).double()
    savepath = '../output/model{}_STFT'.format(seq_len)
    train(model, device, savepath, learning_rate=3e-5)


def finetune():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device : ', device)

    d_feature = 126 * 2  # 144
    d_model = 256
    n_heads = 4
    d_hid = 4 * d_model
    seq_len = SEQ_LEN + 5
    segment = SEGMENT
    n_layers = 8
    n_out = 1

    model = BioTransformer(d_feature=d_feature, d_model=d_model, n_heads=n_heads, d_hid=d_hid, seq_len=seq_len,
                           n_layers=n_layers,
                           n_out=n_out, device=device, segments=segment).to(device)
    model.load_state_dict(torch.load("../output/pretrain_model300_n{}_TUSZ_best".format(n_layers)),
                          strict=False)

    savepath = '../output/finetuned_model{}_n{}'.format(SEQ_LEN, n_layers)
    lr_init = 1e-5
    group_params_lr = [
        # {'params': model.sep_token.parameters(), 'lr': lr_init},
        # {'params': model.cls_token.parameters(), 'lr': lr_init},
        {'params': model.transformer_encoder.layers[0].parameters(), 'lr': 2 * lr_init},
        {'params': model.transformer_encoder.layers[1].parameters(), 'lr': 2 * lr_init},
        {'params': model.transformer_encoder.layers[2].parameters(), 'lr': 2 * lr_init},
        {'params': model.transformer_encoder.layers[3].parameters(), 'lr': 2 * lr_init},
        {'params': model.transformer_encoder.layers[4].parameters(), 'lr': 3 * lr_init},
        {'params': model.transformer_encoder.layers[5].parameters(), 'lr': 3 * lr_init},
        {'params': model.transformer_encoder.layers[6].parameters(), 'lr': 3 * lr_init},
        {'params': model.transformer_encoder.layers[7].parameters(), 'lr': 3 * lr_init},
        # {'params': model.transformer_encoder.layers[8].parameters(), 'lr': 4 * lr_init},
        # {'params': model.transformer_encoder.layers[9].parameters(), 'lr': 4 * lr_init},
        # {'params': model.transformer_encoder.layers[10].parameters(), 'lr': 4 * lr_init},
        # {'params': model.transformer_encoder.layers[11].parameters(), 'lr': 4 * lr_init},
        {'params': model.decoder.parameters(), 'lr': 5 * lr_init},
    ]
    # for layer_num in range(n_layers-2):
    #     for param in model.transformer_encoder.layers[layer_num].parameters():
    #         param.requires_grad = False
    model.decoder.weight.data.uniform_(-0.15, 0.15)
    model.decoder.bias.data.zero_()

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(
                name, param.data.shape)
    train(model, device, savepath, learning_rate=lr_init, params_lr=group_params_lr)


def print_results(conf):
    print("Confusion: ", conf)
    conf_normal = conf
    # conf_normal = conf / np.expand_dims(conf.astype(np.float).sum(axis=1), 1)
    sens = conf_normal[1, 1] / (conf_normal[1, 1] + conf_normal[0, 1])
    precision = conf_normal[1, 1] / (conf_normal[1, 1] + conf_normal[1, 0])
    print("Sensitivity: {:.2f}, Precision: {:.2f}".format(sens, precision))


def evaluate(dataset="TUSZ"):
    d_feature = 350
    d_model = 16
    n_heads = 4
    seq_len = 85
    d_hid = 4 * d_model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device : ', device)
    # model = TSD(d_feature=d_feature, d_model=d_model, n_heads=n_heads, d_hid=d_hid, device=device,
    #             seq_len=seq_len).to(device)
    model = ViT(image_size=(2100, 14), patch_size=(50, 7), num_classes=1, dim=128, depth=2,
                heads=2, mlp_dim=512, pool='cls', channels=1,
                dim_head=64, dropout=0.2, emb_dropout=0.2).to(device).double()
    load_path = '../output/model85_STFT_best'
    # load_path = '../output/model{}_{}_{}_scratch_best'.format(300, n_layers, dataset)
    model.load_state_dict(torch.load(load_path), strict=True)

    print(model)

    best_th_init = 0.7
    for mode in ["val", "test"]:
        test_set = EpilepsyTSD(mode=mode)
        test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

        model.eval()
        test_predict = []
        test_labels = []
        fig_cnt = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch['x'], batch['y']
                x = x[:, np.newaxis, :, :]
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                predicted = torch.sigmoid(outputs)
                test_predict += predicted.view(-1,).tolist()
                test_labels += y.view(-1, ).tolist()
                # if mode == "test":
                #     if predicted[0] > best_th_init and y[0] == 1.0:
                #         sns.heatmap(model(x)[1].squeeze().cpu().numpy(), cmap="magma_r")
                #         plt.title("True Positive")
                #         plt.xlabel("Time")
                #         # plt.xticks(np.arange(0, 21, 21/12))
                #         plt.ylabel("FT x CH")
                #         plt.show()
                #     elif predicted[0] < best_th_init and y[0] == 1.0:
                #         sns.heatmap(x.squeeze().cpu().numpy(), cmap="magma_r", vmax=40)
                #         plt.title("False Negative")
                #         plt.xlabel("Time")
                #         plt.xticks(np.arange(0, 21, 21 / 12))
                #         plt.ylabel("FT x CH")
                #         plt.show()
                #     elif predicted[0] < best_th_init and y[0] == 0.0:
                #         sns.heatmap(model(x)[1].squeeze().cpu().numpy(), cmap="magma_r")
                #         plt.title("True Negative")
                #         plt.xlabel("Time")
                #         # plt.xticks(np.arange(0, 21, 21 / 12))
                #         plt.ylabel("FT x CH")
                #         plt.show()
                #     elif predicted[0] > best_th_init and y[0] == 0.0:
                #         sns.heatmap(x.squeeze().cpu().numpy(), cmap="magma_r", vmax=40)
                #         plt.title("False Positive")
                #         plt.xlabel("Time")
                #         plt.xticks(np.arange(0, 21, 21 / 12))
                #         plt.ylabel("FT x CH")
                #         plt.show()

        if mode == "val":
            best_thresh = thresh_max_f1(y_true=test_labels, y_prob=test_predict)
            best_th_init = best_thresh
        else:
            best_thresh = best_th_init

        if test_predict is not None:
            if len(set(test_labels)) <= 2:  # binary case
                auroc = roc_auc_score(y_true=test_labels, y_score=test_predict)

        print("Best Threshold: {}".format(best_thresh))
        average = 'binary'
        test_predict = (np.array(test_predict) > best_thresh).astype(int)
        acc = accuracy_score(y_true=test_labels, y_pred=test_predict)
        f1 = f1_score(y_true=test_labels, y_pred=test_predict, average=average)
        prec = precision_score(
            y_true=test_labels, y_pred=test_predict, average=average)
        recall = recall_score(
            y_true=test_labels, y_pred=test_predict, average=average)

        conf = confusion_matrix(test_labels, test_predict)
        print_results(conf)
        print("F1 score: ", f1)
        print("accuracy: ", acc)
        print("auroc: ", auroc)
        print("recall: ", recall)
        print("precision: ", prec)


def evaluate_pretraining(dataset='TUSZ', visualization=False):
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            if len(output) == 2:
                activation[name] = output[0].detach()
            else:
                activation[name] = output.detach()

        return hook

    d_feature = 126 * 2 if dataset == "TUSZ" else 144
    d_model = 256
    n_heads = 4
    d_hid = 4 * d_model
    seq_len = SEQ_LEN + 3
    segment = SEGMENT
    n_layers = 12
    n_out = 1
    torch.random.manual_seed(62)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('device : ', device)
    model = BioTransformer(d_feature=d_feature, d_model=d_model, n_heads=n_heads, d_hid=d_hid, seq_len=seq_len,
                           n_layers=n_layers,
                           n_out=n_out, device=device, segments=segment).to(device)

    X, labels, minute_labels, pat_file_start_end, sample_time, valid_labels = get_data(pretrain_mode=True,
                                                                                       dataset=dataset)
    X_train = X["train"]

    pat_start_end = get_pat_start_end(pat_file_start_end)

    train_set = PatientDiscriminatorDataset(torch.from_numpy(X_train).float(), pat_start_end['train'],
                                            sample_time['train'])
    sampler = EvaluateSampler(torch.from_numpy(valid_labels['train']).int(), overlap=60)
    train_loader = DataLoader(train_set, batch_size=1, sampler=sampler)

    # load_path = '../output/pretrain_bandpower1e4_model{}_n{}_{}_best'.format(300, n_layers, dataset)
    load_path = '../output/model{}_{}_{}_scratch_bandpower1e4_best'.format(300, n_layers, dataset)

    model.load_state_dict(torch.load(load_path), strict=False)

    validation_set = PatientDiscriminatorEvaluationDataset(torch.from_numpy(X["val"]).float(), pat_start_end['val'],
                                                           torch.from_numpy(minute_labels['val']).int(),
                                                           sample_time['val'])
    validation_loader = DataLoader(validation_set, batch_size=16, num_workers=4, shuffle=True)

    print(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)

    if visualization:
        for i in range(1, n_layers):
            model.transformer_encoder.layers[i - 1].register_forward_hook(get_activation('layer{}_in'.format(i)))

    test_predict = []
    test_labels = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(train_loader if visualization else validation_loader, position=0, leave=True)):
            x, y = batch['x'], batch['y']
            x, y = x.to(device), y.to(device)
            x = torch.transpose(x, 0, 1)
            outputs = model(x)[0][-1, :, 0]
            tokens = model(x)[1]
            test_predict += outputs.tolist()
            test_labels += y.view(-1, ).tolist()

            if visualization:
                print(outputs)
                plt.subplots(2, 1, figsize=(6, 12))
                plt.subplot(211)
                plt.title("{}, {}".format(y.detach().item(), batch['len'].detach()))
                sns.heatmap(x.squeeze().cpu().numpy().transpose(), cmap="magma_r")
                plt.subplot(212)
                sns.heatmap(tokens.detach().squeeze().cpu().numpy().transpose(), cmap="magma_r")
                fig, axes = plt.subplots(n_layers, 2, figsize=(12, 6))

                for l in range(n_layers):
                    x_in = tokens
                    in_proj_weight = model.transformer_encoder.layers[l].self_attn
                    q, k, v = in_proj_weight.w_q, in_proj_weight.w_k, in_proj_weight.w_v
                    Q = torch.einsum('tbf,hfa->bhta', x_in, q)
                    K = torch.einsum('tbf,hfa->bhta', x_in, k)
                    V = torch.einsum('tbf,hfa->bhta', x_in, v)
                    attn = torch.einsum('bhqa,bhka->bhqk', Q, K).squeeze()

                    res_att_mat = torch.mean(attn, dim=0)
                    res_att_mat = res_att_mat + torch.eye(res_att_mat.shape[0]).to(device)
                    res_att_mat = res_att_mat / res_att_mat.sum(axis=-1)[..., None]
                    print(res_att_mat.shape)

                    attn_rollout = torch.matmul(attn_rollout, res_att_mat) if l != 0 else res_att_mat
                    important_index = [SEGMENT, -1]
                    for idx, im in enumerate(important_index):
                        axes[l, idx].plot(attn_rollout[im, :].detach().cpu().numpy())

                plt.show()

    best_thresh = thresh_max_f1(y_true=test_labels, y_prob=test_predict)
    print("Best Threshold: {}".format(best_thresh))
    test_predict = (np.array(test_predict) > best_thresh) * 1.0
    conf = confusion_matrix(test_labels, test_predict)
    print_results(conf)
    print("F1 score: ", f1_score(test_labels, test_predict))


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
    # get_data(pretrain_mode=False, dataset="TUSZ_STFT", vision_based=True)
    # pretrain("TUSZ")
    # evaluate()
    train_stft(dataset="TUSZ_STFT")
    # evaluate_pretraining(visualization=False)
    # finetune()
