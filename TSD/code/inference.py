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
from tuh_dataset import TUHDataset, get_data_loader
from epilepsy_performance_metrics.src.timescoring.annotations import Annotation
from epilepsy_performance_metrics.src.timescoring.scoring import EventScoring
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from sklearn.metrics import precision_recall_curve, confusion_matrix
from tuh_dataset import args as tuh_args


def seed_everything(seed=99):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=99)


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


# class TUHDataset(Dataset):
#     def __init__(self, file_list, transform=None):
#         self.file_list = file_list
#         self.file_length = len(self.file_list)
#         self.transform = transform
#
#     def __len__(self):
#         return self.file_length
#
#     def __getitem__(self, idx):
#         with open(self.file_list[idx], 'rb') as f:
#             data_pkl = pickle.load(f)
#
#             signals = np.asarray(bipolar_signals_func(data_pkl['signals'], 6))
#             # print(signals.shape)
#
#             if eeg_type == 'stft':
#                 f, t, signals = spectrogram_unfold_feature(signals)  # print(signals.shape)  # exit()
#
#             signals = self.transform(signals)
#             label = data_pkl['label']
#             label = 0. if label == "bckg" else 1.
#
#             patient_id = data_pkl['patient id']
#             # bipolar_channel_name = data_pkl['bipolar_channel_name']
#             confidence = data_pkl['confidence']
#
#         return signals, label, data_pkl['label']


def thresh_max_f1(y_true, y_prob):
    """
    Find best threshold based on precision-recall curve to maximize F1-score.
    Binary calssification only
    """
    if len(set(y_true)) > 2:
        raise NotImplementedError

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresh_filt = []
    fscore = []
    n_thresh = len(thresholds)
    for idx in range(n_thresh):
        curr_f1 = (2 * precision[idx] * recall[idx]) / \
            (precision[idx] + recall[idx])
        if not (np.isnan(curr_f1)):
            fscore.append(curr_f1)
            thresh_filt.append(thresholds[idx])
    # locate the index of the largest f score
    ix = np.argmax(np.array(fscore))
    best_thresh = thresh_filt[ix]
    return best_thresh


def test_event_base():
    save_directory = '/home/amirshah/EPFL/EpilepsyTransformer/TUSZv2/preprocess'
    train_loader, val_loader, test_loader = get_data_loader(1, save_directory, event_base=True)

    val_label_all = np.zeros(0, dtype=np.int)
    val_prob_all = np.zeros(0, dtype=np.float)
    with torch.no_grad():
        for data, label in tqdm(val_loader):
            data = torch.transpose(data, 0, 1)
            label = torch.transpose(label, 0, 1)
            label = torch.squeeze(label, dim=1)
            label = torch.squeeze(label, dim=1)
            val_label_all = np.concatenate((val_label_all, label.cpu().numpy()))
            val_prob = model(data.to(device))
            val_prob = torch.squeeze(sigmoid(val_prob), dim=1)
            val_prob_all = np.concatenate((val_prob_all, val_prob.cpu().numpy()))

    print(set(val_label_all))
    best_th = thresh_max_f1(val_label_all, val_prob_all)
    print("Best threshold : ", best_th)

    test_label_all = np.zeros(0, dtype=np.int)
    test_prob_all = np.zeros(0, dtype=np.float)

    tp_total, fp_total, fn_total = 0, 0, 0
    total_samples = 0

    with torch.no_grad():
        for data, label in tqdm(test_loader):
            data = torch.transpose(data, 0, 1)
            label = torch.transpose(label, 0, 1)
            label = torch.squeeze(label, dim=1)
            label = torch.squeeze(label, dim=1)
            test_label_all = np.concatenate((test_label_all, label.cpu().numpy()))
            test_prob = model(data.to(device))

            test_prob = torch.squeeze(sigmoid(test_prob), dim=1)
            test_prob_all = np.concatenate((test_prob_all, test_prob.cpu().numpy()))

            test_predict = np.where(test_prob.cpu().numpy() > best_th, 1, 0)

            annotation_ref = Annotation(label.cpu().numpy(), 1/12)
            annotation_hyp = Annotation(test_predict, 1/12)
            param = EventScoring.Parameters(
                toleranceStart=30,
                toleranceEnd=60,
                minOverlap=0,
                maxEventDuration=5 * 60,
                minDurationBetweenEvents=90)
            scores = EventScoring(annotation_ref, annotation_hyp, param)
            tp_total += scores.tp
            fp_total += scores.fp
            fn_total += (scores.refTrue - scores.tp)
            total_samples += scores.numSamples

    print("Total true positive events: ", tp_total)
    print("Total false positive events: ", fp_total)
    print("Total false negative events: ", fn_total)

    print("Sensitivity ", 100.0 * tp_total / (tp_total + fn_total))
    print("Precision ", 100.0 * tp_total / (tp_total + fp_total))
    fp_rate = fp_total / (total_samples / scores.fs / 3600)  # FP per hour
    print("False Alarm Rate : ", fp_rate)


def test_sample_base():
    save_directory = '/home/amirshah/EPFL/EpilepsyTransformer/TUSZv2/preprocess'
    train_loader, val_loader, test_loader = get_data_loader(32, save_directory, event_base=False)

    val_label_all = []
    val_prob_all = np.zeros(0, dtype=np.float)
    with torch.no_grad():
        for data, label in tqdm(val_loader):
            val_label_all.extend(label.cpu().numpy())
            val_prob = model(data.to(device))

            val_prob = torch.squeeze(sigmoid(val_prob))
            val_prob_all = np.concatenate((val_prob_all, val_prob.cpu().numpy()))

    print(set(val_label_all))
    best_th = thresh_max_f1(val_label_all, val_prob_all)
    print("Best threshold : ", best_th)

    test_label_all = []
    test_prob_all = np.zeros(0, dtype=np.float)

    with torch.no_grad():
        for data, label in tqdm(test_loader):
            test_label_all.extend(label)
            test_prob = model(data.to(device))

            test_prob = torch.squeeze(sigmoid(test_prob))
            test_prob_all = np.concatenate((test_prob_all, test_prob.cpu().numpy()))

    test_predict_all = np.where(test_prob_all > best_th, 1, 0)
    print("Test confusion matrix: ", confusion_matrix(test_label_all, test_predict_all))

    print("AUROC result: ", roc_auc_score(test_label_all, test_prob_all))


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
root_path = '/home/amirshah/EPFL/EpilepsyTransformer/TUSZv2/preprocess'
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
    else:
        print("Channel-specific Folder does not exist")
        exit()


model = torch.load(model_path,  map_location=torch.device(device))
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
model.eval()
sigmoid = nn.Sigmoid()

test_sample_base()
# test_recall()
# test_run()
