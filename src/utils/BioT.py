from torch import nn, Tensor
import torch
import math
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class BioTransformer(nn.Module):
    def __int__(self, d_model, n_heads, d_hid, seq_len, n_layers, n_out, device=None):
        super().__int__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.device = device
        self.seq_len = seq_len

        encoder_layers = TransformerEncoderLayer(d_model, n_heads, d_hid)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.decoder = nn.Linear(d_model, n_out)

        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def get_positional_embeddings(self):
        result = torch.ones(self.seq_len, self.d_model)
        for i in range(self.seq_len):
            for j in range(self.d_model):
                result[i][j] = np.sin(i / (10000 ** (j / self.d_model))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / self.d_model)))
        return result

    def forward(self, features):
        n, c, w, h = features.shape
        tokens = features
        tokens = torch.stack([torch.vstack((self.class_token, tokens[i])) for i in range(len(tokens))])
        tokens += self.get_positional_embeddings().repeat(n, 1, 1).to(self.device)

        output = self.transformer_encoder(tokens)
        output = self.decoder(output)
        return output
