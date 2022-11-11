from torch import nn, Tensor
import torch
import math
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
from bisect import bisect

torch.manual_seed(1)

SEQ_LEN = 300
SEGMENT = 240
ROI = SEQ_LEN - SEGMENT


class BioTransformer(nn.Module):

    def __init__(self, d_feature, d_model, n_heads, d_hid, n_layers, n_out, seq_len, segments, device):
        super(BioTransformer, self).__init__()
        self.n_heads = n_heads
        self.d_feature = d_feature
        self.d_model = d_model
        self.device = device
        self.seq_len = seq_len

        self.roi_length = torch.tensor([60, 50, 40, 30, 20, 10, 0])
        encoder_layers = TransformerEncoderLayer(d_model, n_heads, d_hid)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.encoder = nn.Linear(d_feature, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1, max_len=seq_len, segment_length=self.roi_length)
        self.decoder = nn.Linear(d_model, n_out)
        self.sigmoid = nn.Sigmoid()

        self.cls_token = nn.Parameter(torch.rand(1, 1, d_model))
        self.sep_token = nn.Parameter(torch.rand(1, 1, d_model))

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, features):
        c, n, h = features.shape
        src = self.encoder(features) #* math.sqrt(self.d_model)
        tokens = src[:SEQ_LEN - self.roi_length[0], :, :]
        for i in range(len(self.roi_length) - 1):
            tokens = torch.cat((tokens, self.sep_token.repeat(1, n, 1),
                                src[SEQ_LEN - self.roi_length[i]: SEQ_LEN - self.roi_length[i+1], :, :]), dim=0)

        tokens = torch.cat((tokens, self.cls_token.repeat(1, n, 1)), dim=0)
        tokens = self.pos_encoder(tokens)
        output = self.transformer_encoder(tokens)
        output = self.decoder(output)
        # output = self.sigmoid(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, segment_length: Tensor = 0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        se = torch.zeros_like(pe)  # segment encoding
        se[:SEQ_LEN-segment_length[0], 0, :] = -1
        for i in range(len(segment_length)-1):
            se[SEQ_LEN-segment_length[i]+(i): SEQ_LEN-segment_length[i+1]+(i+1), 0, :] = -0.7 + 2*i/len(segment_length)
        self.register_buffer('pe', pe)
        self.register_buffer('se', se)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)].expand(-1, x.size(1), -1)
        x = x + self.se[:x.size(0)].expand(-1, x.size(1), -1)
        return self.dropout(x)


class Epilepsy60Dataset(Dataset):
    def __init__(self, x_total, y_total, sample_time):
        self.x_total = x_total
        self.y_total = y_total
        self.sample_time = sample_time

    def __len__(self):
        return self.x_total.shape[0]

    def __getitem__(self, idx):
        if self.sample_time[idx] < SEQ_LEN:
            valid_len = self.sample_time[idx]
            zero_pad = torch.zeros((SEQ_LEN-valid_len-1, self.x_total.shape[1]), dtype=torch.float)
            x60 = torch.cat((zero_pad, self.x_total[idx-valid_len:idx+1, :]), dim=0)
        else:
            valid_len = SEQ_LEN - 1
            x60 = self.x_total[idx - SEQ_LEN + 1:idx + 1, :]
        valid_roi = min(valid_len, ROI)
        y60 = torch.max(self.y_total[idx-valid_roi : idx +1])

        sample = {'x': x60, 'y': y60, 'idx': idx}
        return sample


class ImbalancedDataSampler(Sampler):
    def __init__(self, seizure_indices, non_seizure_indices, post_seizure_indices, post_non_ratio = 0.4, overlap:int =1):
        self.seizure_indices = seizure_indices[::overlap]
        self.non_seizure_indices = non_seizure_indices
        self.post_seizure_indices = post_seizure_indices
        self.num_seizure = len(self.seizure_indices)
        self.num_non_seizure = len(non_seizure_indices)
        self.num_post_seizure = len(post_seizure_indices)
        self.post_seizure_chosen_len = int(post_non_ratio * self.num_seizure)

    def __iter__(self):
        print("Sampler Called")
        sampled_non_seizure_indices = torch.randperm(self.num_non_seizure)[:self.num_seizure-self.post_seizure_chosen_len]
        sampled_post_seizure_indices = torch.randperm(self.num_post_seizure)[:self.post_seizure_chosen_len]

        seizure_non_seizure_cat = torch.cat(
            (self.seizure_indices,
             self.non_seizure_indices[sampled_non_seizure_indices],
             self.post_seizure_indices[sampled_post_seizure_indices]), 0)
        return iter(seizure_non_seizure_cat[torch.randperm(2 * self.num_seizure)])

    def __len__(self):
        return 2 * self.num_seizure


class EvaluateSampler(Sampler):
    def __init__(self, valid_indices, overlap:int =1):
        super().__init__(valid_indices)
        valid_indices_sampled = valid_indices[::overlap]
        self.valid_indices_perm = valid_indices_sampled[torch.randperm(len(valid_indices)//overlap)]

    def __iter__(self):
        return iter(self.valid_indices_perm)

    def __len__(self):
        return len(self.valid_indices_perm)


class PatientDiscriminatorDataset(Dataset):
    def __init__(self, x_total, pat_start_end, sample_time):
        self.x_total = x_total
        self.pat_start_end = pat_start_end
        self.pat_start = [x[0] for x in pat_start_end]
        self.pats = len(pat_start_end)
        self.roi_length = torch.tensor([10, 20, 30, 40, 50, 60])
        self.roi_possible_len = len(self.roi_length)
        self.sample_time = sample_time

    def __len__(self):
        return self.x_total.shape[0]

    def __getitem__(self, idx):
        random_index = torch.randint(low=0, high=self.roi_possible_len, size=(1,))
        roi_random = self.roi_length[random_index]
        seg_random = SEQ_LEN - roi_random
        x_later = self.x_total[idx - roi_random+1:idx+1, :]
        pat_num = bisect(self.pat_start, idx) - 1
        if np.random.uniform(0, 1) > 0.5:
            label = 1.0
            other_pats = np.delete(np.arange(0, self.pats), pat_num)
            pat_sample = np.random.choice(other_pats)
        else:
            label = 0.0
            pat_sample = pat_num

        idx_sample = torch.randint(low=self.pat_start_end[pat_sample][0],
                                   high=self.pat_start_end[pat_sample][1],
                                   size=(1,))
        #
        valid_len = self.sample_time[idx_sample]
        if valid_len < seg_random:
            zero_pad = torch.zeros((seg_random-valid_len-1, self.x_total.shape[1]), dtype=torch.float)
            x_prior = torch.cat((zero_pad, self.x_total[idx_sample-valid_len:idx_sample+1, :]), dim=0)

        else:
            x_prior = self.x_total[idx_sample - seg_random:idx_sample, :]
        x = torch.cat((x_prior, x_later), dim=0)
        sample = {'x': x, 'y': label, 'len': roi_random, 'valid_len': valid_len}
        return sample


class PatientDiscriminatorEvaluationDataset(Dataset):
    def __init__(self, x_total, pat_start_end, valid_indices, sample_time):
        self.x_total = x_total
        self.pat_start_end = pat_start_end
        self.pat_start = [x[0] for x in pat_start_end]
        self.valid_minutes_indices = valid_indices
        self.sample_time = sample_time

    def __len__(self):
        return self.valid_minutes_indices.shape[0] ** 2

    def __getitem__(self, idx):
        idx_later = self.valid_minutes_indices[idx // self.valid_minutes_indices.shape[0]]
        idx_prior = self.valid_minutes_indices[idx % self.valid_minutes_indices.shape[0]]
        x_later = self.x_total[idx_later - ROI+1:idx_later+1, :]
        valid_len = self.sample_time[idx_prior]
        if valid_len < SEGMENT:
            zero_pad = torch.zeros((SEGMENT-valid_len-1, self.x_total.shape[1]), dtype=torch.float)
            x_prior = torch.cat((zero_pad, self.x_total[idx_prior-valid_len:idx_prior+1, :]), dim=0)
        else:
            x_prior = self.x_total[idx_prior - SEGMENT:idx_prior, :]
        pat_num_later = bisect(self.pat_start, idx_later) - 1
        pat_num_prior = bisect(self.pat_start, idx_prior) - 1
        label = 0.0 if pat_num_prior == pat_num_later else 1.0
        x = torch.cat((x_prior, x_later), dim=0)
        sample = {'x': x, 'y': label}
        return sample
