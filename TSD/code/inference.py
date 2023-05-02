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


def seed_everything(seed=99):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=55)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim),
                                 nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList(
                [PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class EEGVT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        patch_height, patch_width = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim), )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


def bipolar_signals_func(signals):
    bipolar_signals = []
    bipolar_signals.append(signals[0] - signals[4])  # fp1-f7
    bipolar_signals.append(signals[1] - signals[5])  # fp2-f8
    bipolar_signals.append(signals[4] - signals[9])  # f7-t3
    bipolar_signals.append(signals[5] - signals[10])  # f8-t4
    bipolar_signals.append(signals[9] - signals[15])  # t3-t5
    bipolar_signals.append(signals[10] - signals[16])  # t4-t6
    bipolar_signals.append(signals[15] - signals[13])  # t5-o1
    bipolar_signals.append(signals[16] - signals[14])  # t6-o2
    bipolar_signals.append(signals[9] - signals[6])  # t3-c3
    bipolar_signals.append(signals[7] - signals[10])  # c4-t4
    bipolar_signals.append(signals[6] - signals[8])  # c3-cz
    bipolar_signals.append(signals[8] - signals[7])  # cz-c4
    bipolar_signals.append(signals[0] - signals[2])  # fp1-f3
    bipolar_signals.append(signals[1] - signals[3])  # fp2-f4
    bipolar_signals.append(signals[2] - signals[6])  # f3-c3
    bipolar_signals.append(signals[3] - signals[7])  # f4-c4
    bipolar_signals.append(signals[6] - signals[11])  # c3-p3
    bipolar_signals.append(signals[7] - signals[12])  # c4-p4
    bipolar_signals.append(signals[11] - signals[13])  # p3-o1
    bipolar_signals.append(signals[12] - signals[14])  # p4-o2

    return bipolar_signals


def spectrogram_unfold_feature(signals):
    nperseg = 250
    noverlap = 50
    freq_resolution = 2
    nfft = sample_rate * freq_resolution
    freqs, times, spec = stft(signals, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, nfft=nfft, boundary=None,
                              padded=False)

    spec = spec[:, :spec.shape[1] - 1, :]
    spec = np.reshape(spec, (-1, spec.shape[2]))
    amp = (np.log(np.abs(spec) + 1e-10)).astype(np.float32)

    return freqs, times, amp


class TUHDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.file_length = len(self.file_list)
        self.transform = transform

    def __len__(self):
        return self.file_length

    def __getitem__(self, idx):
        with open(self.file_list[idx], 'rb') as f:
            data_pkl = pickle.load(f)

            signals = np.asarray(bipolar_signals_func(data_pkl['signals']))
            # print(signals.shape)

            if eeg_type == 'stft':
                f, t, signals = spectrogram_unfold_feature(signals)  # print(signals.shape)  # exit()

            signals = self.transform(signals)
            label = data_pkl['label']
            label = 0. if label == "bckg" else 1.

            patient_id = data_pkl['patient id']
            # bipolar_channel_name = data_pkl['bipolar_channel_name']
            confidence = data_pkl['confidence']

        return signals, label, data_pkl['label']


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

    bckg_data = file_lists['val']['bckg'] + file_lists['test']['bckg']
    shuffle(bckg_data)

    seiz_data = file_lists['val']['seiz'] + file_lists['test']['seiz']
    shuffle(seiz_data)

    val_data = bckg_data[:int(len(bckg_data) / 2)] + seiz_data[:int(len(seiz_data) / 2)]
    shuffle(val_data)
    print('len(val_data): {}'.format(len(val_data)))

    test_data = bckg_data[int(len(bckg_data) / 2):] + seiz_data[int(len(seiz_data) / 2):]
    shuffle(test_data)
    print('len(test_data): {}'.format(len(test_data)))

    train_transforms = transforms.Compose([transforms.ToTensor(), ])

    val_transforms = transforms.Compose([transforms.ToTensor(), ])

    test_transforms = transforms.Compose([transforms.ToTensor(), ])

    train_data = TUHDataset(train_data, transform=train_transforms)
    val_data = TUHDataset(val_data, transform=val_transforms)
    test_data = TUHDataset(test_data, transform=test_transforms)

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def test():
    print(device)
    save_directory = '/home/amirshah/EPFL/EpilepsyTransformer/TUSZv2/preprocess'
    train_loader, val_loader, test_loader = get_data_loader(save_directory, 32)

    test_label_all = []
    test_prob_all = np.zeros(0, dtype=np.float)

    time_consumes = []
    with torch.no_grad():
        for data, label, label_name in tqdm(test_loader):
            test_label_all.extend(label)

            start_time = time.perf_counter()
            test_prob = model(data.to(device))
            time_consumes.append(time.perf_counter() - start_time)

            test_prob = torch.squeeze(sigmoid(test_prob))
            test_prob_all = np.concatenate((test_prob_all, test_prob.cpu().numpy()))

    test_auc = roc_auc_score(test_label_all, test_prob_all)
    print(f"test_auc: {test_auc}")
    print('average_time: {}'.format(sum(time_consumes) / len(time_consumes)))

    #############找阈值
    def Find_Optimal_Cutoff(FPR, TPR, thresholds):
        y = TPR - FPR
        Youden_index = np.argmax(y)  # Only the first occurrence is returned.
        optimal_threshold = thresholds[Youden_index]
        point = [FPR[Youden_index], TPR[Youden_index]]
        return optimal_threshold, point

    fpr, tpr, thresholds = metrics.roc_curve(test_label_all, test_prob_all)
    print(Find_Optimal_Cutoff(fpr, tpr, thresholds))

    # #############画图部分  # roc_auc = metrics.auc(fpr, tpr)  # plt.figure(figsize=(6,6))  # plt.title('Validation ROC')  # plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.4f' % roc_auc)  # plt.legend(loc = 'lower right')  # plt.plot([0, 1], [0, 1],'r--')  # plt.xlim([0, 1])  # plt.ylim([0, 1])  # plt.ylabel('True Positive Rate')  # plt.xlabel('False Positive Rate')  # plt.show()


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


sample_rate = 250
eeg_type = 'stft'  # 'original', 'bipolar', 'stft'
device = 'cuda:0'
# device = 'cpu'

model = torch.load('inference_ck_0.9208', map_location=torch.device(device))
# model = torch.load('/home/amirshah/EPFL/EpilepsyTransformer/TUSZv2/preprocess/test3/test_model_0.867288059115128',
#                    map_location=torch.device(device))
model.eval()
sigmoid = nn.Sigmoid()

test()
# test_recall()
# test_run()
