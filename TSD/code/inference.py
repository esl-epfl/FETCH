import os

import math

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from sklearn import metrics
import random
from random import shuffle
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from sklearn.metrics import roc_auc_score
import pickle
import numpy as np
from scipy.signal import stft
from tqdm import tqdm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from tuh_dataset import TUHDataset
from epilepsy_performance_metrics.src.timescoring.annotations import Annotation
from epilepsy_performance_metrics.src.timescoring.annotations import Annotation
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def seed_everything(seed=99):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=55)


def get_data_loader(save_directory, batch_size=1):
    file_dir = {'train': os.path.join(save_directory, 'task-binary_datatype-train'),
                'val': os.path.join(save_directory, 'task-binary_datatype-eval'),
                'test': os.path.join(save_directory, 'task-binary_datatype-dev')}
    file_lists = {'train': {'bckg': [], 'seiz': []}, 'val': {'bckg': [], 'seiz': []}, 'test': {'bckg': [], 'seiz': []}}

    for dirname in file_dir.keys():
        filenames = os.listdir(file_dir[dirname])
        for filename in filenames:
            if 'bckg' in filename:
                file_lists[dirname]['bckg'].append(os.path.join(file_dir[dirname], filename))
            elif 'seiz' in filename:
                file_lists[dirname]['seiz'].append(os.path.join(file_dir[dirname], filename))
            else:
                print('------------------------  error  ------------------------')
                exit(-1)

    train_data = file_lists['train']['bckg'] + file_lists['train']['seiz'] * int(
        len(file_lists['train']['bckg']) / len(file_lists['train']['seiz']))
    shuffle(train_data)
    print('len(train_data): {}'.format(len(train_data)))

    val_data = file_lists['val']['bckg'] + file_lists['val']['seiz']
    shuffle(val_data)
    print('len(val_data): {}'.format(len(val_data)))

    test_data = file_lists['test']['bckg'] + file_lists['test']['seiz']
    shuffle(test_data)
    print('len(test_data): {}'.format(len(test_data)))

    train_transforms = transforms.Compose([transforms.ToTensor(), ])

    val_transforms = transforms.Compose([transforms.ToTensor(), ])

    test_transforms = transforms.Compose([transforms.ToTensor(), ])

    train_data = TUHDataset(train_data, transform=train_transforms)
    val_data = TUHDataset(val_data, transform=val_transforms)
    test_data = TUHDataset(test_data, transform=test_transforms)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, val_loader, test_loader


def test():
    print(device)
    save_directory = '/home/amirshah/EPFL/EpilepsyTransformer/TUSZv2/preprocess'
    train_loader, val_loader, test_loader = get_data_loader(save_directory, 32)

    test_label_all = []
    test_prob_all = np.zeros(0, dtype=np.float)

    time_consumes = []
    with torch.no_grad():
        for data, label in tqdm(test_loader):
            test_label_all.extend(label)

            start_time = time.perf_counter()
            test_prob = model(data.to(device))
            time_consumes.append(time.perf_counter() - start_time)

            test_prob = torch.squeeze(sigmoid(test_prob))
            test_prob_all = np.concatenate((test_prob_all, test_prob.cpu().numpy()))

            # annotation_labels = Annotation(label.cpu().numpy(), 1/12)
            # annotation_pred = Annotation(test_prob.cpu().numpy(), 1/12)
            # scores =

    test_auc = roc_auc_score(test_label_all, test_prob_all)
    print(f"test_auc: {test_auc}")
    print('average_time: {}'.format(sum(time_consumes) / len(time_consumes)))



    # def Find_Optimal_Cutoff(FPR, TPR, thresholds):
    #     y = TPR - FPR
    #     Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    #     optimal_threshold = thresholds[Youden_index]
    #     point = [FPR[Youden_index], TPR[Youden_index]]
    #     return optimal_threshold, point
    #
    # fpr, tpr, thresholds = metrics.roc_curve(test_label_all, test_prob_all)
    # print(Find_Optimal_Cutoff(fpr, tpr, thresholds))

  # roc_auc = metrics.auc(fpr, tpr)  # plt.figure(figsize=(6,6))  # plt.title('Validation ROC')  # plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.4f' % roc_auc)  # plt.legend(loc = 'lower right')  # plt.plot([0, 1], [0, 1],'r--')  # plt.xlim([0, 1])  # plt.ylim([0, 1])  # plt.ylabel('True Positive Rate')  # plt.xlabel('False Positive Rate')  # plt.show()


def get_data_loader_multi(save_directory, batch_size=1):
    file_dir = {'eval_multi': os.path.join(save_directory, 'task-multi-classification_datatype-eval'),
                'dev_multi': os.path.join(save_directory, 'task-multi-classification_datatype-dev')}
    file_lists = {
        'eval_multi': {'bckg': [], 'absz': [], 'cpsz': [], 'fnsz': [], 'gnsz': [], 'mysz': [], 'spsz': [], 'tcsz': [],
                       'tnsz': []},
        'dev_multi': {'bckg': [], 'absz': [], 'cpsz': [], 'fnsz': [], 'gnsz': [], 'mysz': [], 'spsz': [], 'tcsz': [],
                      'tnsz': []}}

    for dirname in file_dir.keys():
        filenames = os.listdir(file_dir[dirname])
        for filename in filenames:
            label = ''
            for label in file_lists['eval_multi'].keys():
                if label in filename:
                    file_lists[dirname][label].append(os.path.join(file_dir[dirname], filename))
                    break
            if file_lists[dirname][label][-1] != os.path.join(file_dir[dirname], filename):
                print(' ------------------- error ------------------- {}'.format(label))
                print(filename)
                exit()

    test_transforms = transforms.Compose([transforms.ToTensor(), ])

    eval_multi_data = []
    dev_multi_data = []
    labels = list(file_lists['eval_multi'].keys())
    labels.remove('bckg')

    for label in labels:
        eval_multi_data += file_lists['eval_multi'][label]
        dev_multi_data += file_lists['dev_multi'][label]

        print('eval-{}-{}'.format(label, len(file_lists['eval_multi'][label])))
        print('dev-{}-{}'.format(label, len(file_lists['dev_multi'][label])))

    eval_multi_data = TUHDataset(eval_multi_data, transform=test_transforms)
    dev_multi_data = TUHDataset(dev_multi_data, transform=test_transforms)

    eval_multi_loader = DataLoader(dataset=eval_multi_data, batch_size=batch_size, shuffle=False)
    dev_multi_loader = DataLoader(dataset=dev_multi_data, batch_size=batch_size, shuffle=False)

    return eval_multi_loader, dev_multi_loader


def test_recall():
    save_directory = '/home/michael/workspace/TUH/tuh_eeg_seizure/v1.5.4/second-paper/preprocess'
    eval_multi_loader, dev_multi_loader = get_data_loader_multi(save_directory, 1)

    result = {'bckg': {'total': [], 'correct_prediction': []}, 'absz': {'total': [], 'correct_prediction': []},
              'cpsz': {'total': [], 'correct_prediction': []}, 'fnsz': {'total': [], 'correct_prediction': []},
              'gnsz': {'total': [], 'correct_prediction': []}, 'mysz': {'total': [], 'correct_prediction': []},
              'spsz': {'total': [], 'correct_prediction': []}, 'tcsz': {'total': [], 'correct_prediction': []},
              'tnsz': {'total': [], 'correct_prediction': []}}

    with torch.no_grad():
        for data_loader in [eval_multi_loader, dev_multi_loader]:
            for data, label, label_name in tqdm(data_loader):
                label_name = label_name[0]
                result[label_name]['total'].append(float(label.cpu().numpy()[0]))

                test_prob = model(data.to(device))
                test_prob = torch.squeeze(sigmoid(test_prob))
                test_prob = float(test_prob.cpu().numpy())
                if test_prob >= 0.3901386260986328:
                    result[label_name]['correct_prediction'].append(test_prob)

    for sz_type in ['absz', 'cpsz', 'fnsz', 'gnsz', 'mysz', 'spsz', 'tcsz', 'tnsz']:
        if len(result[sz_type]['total']) == 0:
            print(f"{sz_type} - len: 0")
        else:
            print(
                f"{sz_type} - recall: {float(len(result[sz_type]['correct_prediction'])) / float(len(result[sz_type]['total']))}")


def test_run():
    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    path = '/home/amirshah/EPFL/EpilepsyTransformer/TUSZv2/preprocess/task-binary_datatype-eval'
    test_data = TUHDataset(['{}/{}'.format(path, item) for item in os.listdir(path) if item.endswith('.pkl')], transform=test_transforms)
    test_loader = DataLoader(dataset=test_data, batch_size=math.ceil(len(test_data) / 50), shuffle=False)
    with torch.no_grad():
        for data, label, label_name in tqdm(test_loader):
            test_prob = model(data.to(device))
            test_prob = torch.squeeze(sigmoid(test_prob))
            # print(test_prob.cpu().numpy())


sample_rate = 256
eeg_type = 'stft'  # 'original', 'bipolar', 'stft'
device = 'cuda:0'
# device = 'cpu'

# model = torch.load('inference_ck_0.9208', map_location=torch.device(device))
model = torch.load('/home/amirshah/EPFL/EpilepsyTransformer/TUSZv2/preprocess/test_v2/test_model_0.9374276011157442',
                   map_location=torch.device(device))
model.eval()
sigmoid = nn.Sigmoid()

test()
# test_recall()
# test_run()
