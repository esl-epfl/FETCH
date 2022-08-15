from torch import nn, Tensor
import torch
import math
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
import torch.nn.functional as F
torch.manual_seed(1)


class BioTransformer(nn.Module):

    def __init__(self, d_model, n_heads, d_hid, seq_len, n_layers, n_out, device):
        super(BioTransformer, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.device = device
        self.seq_len = seq_len

        encoder_layers = TransformerEncoderLayer(d_model, n_heads, d_hid)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1, max_len=seq_len)
        self.decoder = nn.Linear(d_model, n_out)
        self.sigmoid = nn.Sigmoid()

        self.class_token = nn.Parameter(torch.rand(1, d_model))

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, features):
        n, c, h = features.shape
        tokens = self.pos_encoder(features)
        # tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])
        output = self.transformer_encoder(tokens)
        output = self.decoder(output)
        # output = self.sigmoid(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe
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

        x60 = self.x_total[idx-59:idx+1, :]

        sample = {'x': x60, 'y': self.y_total[idx]}
        return sample


class ImbalancedDataSampler(Sampler):
    def __init__(self, valid_seizure_indices, valid_non_seizure_indices):
        self.seizure_indices = valid_seizure_indices
        self.non_seizure_indices = valid_non_seizure_indices
        self.num_seizure = len(valid_seizure_indices)
        self.num_non_seizure = len(valid_non_seizure_indices)

    def __iter__(self):
        sampled_non_seizure_indices = torch.randperm(self.num_non_seizure)[:self.num_seizure]
        seizure_non_seizure_cat = torch.cat((self.seizure_indices, self.non_seizure_indices[sampled_non_seizure_indices]), 0)
        return iter(seizure_non_seizure_cat[torch.randperm(2*self.num_seizure)])

    def __len__(self):
        return 2*self.num_seizure

