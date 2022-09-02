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

        self.pe = nn.Parameter(torch.randn(seq_len, 1, d_model))
        se_roi = nn.Parameter(torch.randn(1, 1, d_model)).repeat(seq_len - segments,1, 1)
        se_seg = nn.Parameter(torch.randn(1, 1, d_model)).repeat(segments, 1, 1)
        self.se = torch.cat((se_roi, se_seg), dim=0).to(device)

        encoder_layers = TransformerEncoderLayer(d_model, n_heads, d_hid)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.encoder = nn.Linear(d_feature, d_model)
        self.pos_encoder = PositionalEncoding(self.pe, self.se, d_model, dropout=0.1, max_len=seq_len, segment_length=segments)
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

    def forward(self, features_prior, features_later):
        c, n, h = features_prior.shape
        src_prior = self.encoder(features_prior) #* math.sqrt(self.d_model)
        src_later = self.encoder(features_later) #* math.sqrt(self.d_model)
        tokens = torch.cat((src_prior, self.sep_token.expand(-1, n, -1), src_later, self.cls_token.expand(-1, n, -1)),
                           dim=0)
        tokens = self.pos_encoder(tokens)
        output = self.transformer_encoder(tokens)
        output = self.decoder(output)
        # output = self.sigmoid(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, pe, se, d_model: int, dropout: float = 0.1, max_len: int = 5000, segment_length: int = 0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # position = torch.arange(max_len).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # pe = torch.zeros(max_len, 1, d_model)
        # pe[:, 0, 0::2] = torch.sin(position * div_term)
        # pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.pe = pe
        self.se = se
        # self.se = torch.zeros_like(pe)  # segment encoding
        # self.se[:segment_length, :, :] = self.se_seg.expand(segment_length, -1, -1)
        # self.se[segment_length:, :, :] = self.se_roi.expand(max_len-segment_length, -1, -1)
        # self.se = torch.cat((self.se_seg.expand(segment_length, -1, -1),
        #                      self.se_roi.expand(max_len-segment_length, -1, -1)))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)].expand(-1, x.size(1), -1)
        x = x + self.se[:x.size(0)].expand(-1, x.size(1), -1)
        return self.dropout(x)


class Epilepsy60Dataset(Dataset):
    def __init__(self, x_total, y_total):
        self.x_total = x_total
        self.y_total = y_total

    def __len__(self):
        return self.x_total.shape[0]

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        if idx < SEQ_LEN:
            # x60_zero = np.zeros((59-idx, self.x_total.shape[1]), dtype=np.float)
            # x60 = np.concatenate((x60_zero, self.x_total[0:idx+1, :]))
            x60 = self.x_total[0:SEQ_LEN, :]
            y60 = self.y_total[SEQ_LEN]
        else:
            x60 = self.x_total[idx - SEQ_LEN + 1:idx + 1, :]
            y60 = self.y_total[idx]

        sample = {'x': x60, 'y': y60}
        return sample


class ImbalancedDataSampler(Sampler):
    def __init__(self, valid_seizure_indices, valid_non_seizure_indices):
        self.seizure_indices = valid_seizure_indices
        self.non_seizure_indices = valid_non_seizure_indices
        self.num_seizure = len(valid_seizure_indices)
        self.num_non_seizure = len(valid_non_seizure_indices)

    def __iter__(self):
        sampled_non_seizure_indices = torch.randperm(self.num_non_seizure)[:self.num_seizure]
        seizure_non_seizure_cat = torch.cat(
            (self.seizure_indices, self.non_seizure_indices[sampled_non_seizure_indices]), 0)
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
    def __init__(self, x_total, pat_start_end):
        self.x_total = x_total
        self.pat_start_end = pat_start_end
        self.pat_start = [x[0] for x in pat_start_end]
        self.pats = len(pat_start_end)

    def __len__(self):
        return self.x_total.shape[0]

    def __getitem__(self, idx):
        x_later = self.x_total[idx - ROI:idx, :]
        pat_num = bisect(self.pat_start, idx) - 1
        if np.random.uniform(0, 1) > 0.5:
            label = 1
            other_pats = np.delete(np.arange(0, self.pats), pat_num)
            pat_sample = np.random.choice(other_pats)
        else:
            label = 0
            pat_sample = pat_num

        idx_sample = np.random.randint(low=self.pat_start_end[pat_sample][0] + SEGMENT,
                                       high=self.pat_start_end[pat_sample][1])
        x_prior = self.x_total[idx_sample - SEGMENT:idx_sample, :]
        sample = {'x1': x_prior, 'x2': x_later, 'y': label}
        return sample


class PatientDiscriminatorEvaluationDataset(Dataset):
    def __init__(self, x_total, pat_start_end, valid_indices):
        self.x_total = x_total
        self.pat_start_end = pat_start_end
        self.pat_start = [x[0] for x in pat_start_end]
        self.valid_minutes_indices = valid_indices[::60]

    def __len__(self):
        return self.valid_minutes_indices.shape[0] ** 2

    def __getitem__(self, idx):
        idx_later = self.valid_minutes_indices[idx // self.valid_minutes_indices.shape[0]]
        idx_prior = self.valid_minutes_indices[idx % self.valid_minutes_indices.shape[0]]
        x_later = self.x_total[idx_later - ROI:idx_later, :]
        x_prior = self.x_total[idx_prior - SEGMENT:idx_prior, :]
        pat_num_later = bisect(self.pat_start, idx_later) - 1
        pat_num_prior = bisect(self.pat_start, idx_prior) - 1
        label = 0 if pat_num_prior==pat_num_later else 1
        sample = {'x1': x_prior, 'x2': x_later, 'y': label}
        return sample
