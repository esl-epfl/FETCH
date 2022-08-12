from torch import nn, Tensor
import torch
import math
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
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
        self.decoder = nn.Linear(d_model, n_out)
        self.sigmoid = nn.Sigmoid()

        self.class_token = nn.Parameter(torch.rand(1, d_model))

    def get_positional_embeddings(self):
        result = torch.ones(self.seq_len, self.d_model)
        for i in range(self.seq_len):
            for j in range(self.d_model):
                result[i][j] = np.sin(i / (10000 ** (j / self.d_model))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / self.d_model)))
        return result

    def forward(self, features):
        n, c, h = features.shape
        tokens = features
        # tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])
        tokens += self.get_positional_embeddings().repeat(n, 1, 1).to(self.device)

        output = self.transformer_encoder(tokens)
        output = self.decoder(output)
        # output = self.sigmoid(output)
        return output



class Epilepsy60Dataset(Dataset):
    def __init__(self, x_total, y_total):
        self.x_total = x_total
        self.y_total = y_total

    def __len__(self):
        return self.x_total.shape[0]

    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()

        if idx < 59:
            x60 = torch.zeros((60, 144))
        else:
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
        return iter(torch.cat((self.seizure_indices, self.non_seizure_indices[sampled_non_seizure_indices]), 0))

    def __len__(self):
        return 2*self.num_seizure

