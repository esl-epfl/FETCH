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

        self.roi_length = torch.tensor([60, 45, 30, 15, 0])
        encoder_layers = TransformerEncoderLayer_(d_model, n_heads, d_hid, relative_positional_distance=30)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        # self.encoder = nn.Linear(d_feature, d_model)
        # self.pos_encoder = PositionalEncoding(d_model, dropout=0.1, max_len=seq_len, segment_length=self.roi_length)
        self.decoder1 = nn.Linear(d_model, 64)
        self.decoder2 = nn.Linear(64, n_out)
        self.decoder_finetune = nn.Linear(d_model, n_out)

        # self.sigmoid = nn.Sigmoid()

        self.cls_token = nn.Parameter(torch.rand(1, 1, d_model))
        self.sep_token = nn.Parameter(torch.rand(1, 1, d_model))
        self.segment_embedding = nn.Parameter(torch.rand(7, 1, d_model))

        self.dropout = nn.Dropout(0.3)
        self.activation = nn.ReLU()

        self.init_weights()

    def init_weights(self) -> None:
        initrange1 = 0.14
        initrange2 = 0.3
        # self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder1.bias.data.zero_()
        self.decoder1.bias.data.zero_()
        self.decoder2.weight.data.uniform_(-initrange1, initrange1)
        self.decoder2.weight.data.uniform_(-initrange2, initrange2)

    def forward(self, features):
        c, n, h = features.shape
        src = torch.cat((features, torch.zeros((c,n,self.d_model - h), device=self.device)), dim=2)
        # src = self.encoder(features) #* math.sqrt(self.d_model)
        segment0 = self.segment_embedding[:1,:,:].repeat(SEQ_LEN - self.roi_length[0], n, 1)
        tokens = src[:SEQ_LEN - self.roi_length[0], :, :] + segment0
        for i in range(len(self.roi_length) - 1):
            segment_i = self.segment_embedding[i:i+1,:,:].repeat(self.roi_length[i] - self.roi_length[i+1], n, 1)
            tokens = torch.cat((tokens, self.sep_token.repeat(1, n, 1),
                                src[SEQ_LEN - self.roi_length[i]: SEQ_LEN - self.roi_length[i+1], :, :] + segment_i),
                               dim=0)

        tokens = torch.cat((tokens, self.cls_token.repeat(1, n, 1)), dim=0)
        # tokens = self.pos_encoder(tokens)
        output = self.transformer_encoder(tokens)
        output = self.decoder_finetune(output)
        # output = self.decoder2(self.dropout(self.activation(self.decoder1(output))))
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
        self.roi_length = torch.tensor([10, 20, 25, 30, 35, 40, 45, 50, 53, 57, 60, 63, 67, 70, 75, 80, 90])
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


class TransformerEncoderLayer_(nn.Module):
    # Adapted from pytorch source
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, relative_positional=True, relative_positional_distance=100):
        super(TransformerEncoderLayer_, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout, relative_positional=relative_positional, relative_positional_distance=relative_positional_distance)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src: torch.Tensor, src_mask = None, src_key_padding_mask= None) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class MultiHeadAttention(nn.Module):
  def __init__(self, d_model=256, n_head=4, dropout=0.1, relative_positional=True, relative_positional_distance=100):
    super().__init__()
    self.d_model = d_model
    self.n_head = n_head
    d_qkv = d_model // n_head
    assert d_qkv * n_head == d_model, 'd_model must be divisible by n_head'
    self.d_qkv = d_qkv

    self.w_q = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
    self.w_k = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
    self.w_v = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
    self.w_o = nn.Parameter(torch.Tensor(n_head, d_qkv, d_model))
    nn.init.xavier_normal_(self.w_q)
    nn.init.xavier_normal_(self.w_k)
    nn.init.xavier_normal_(self.w_v)
    nn.init.xavier_normal_(self.w_o)

    self.dropout = nn.Dropout(dropout)

    if relative_positional:
        self.relative_positional = LearnedRelativePositionalEmbedding(relative_positional_distance, n_head, d_qkv, True)
    else:
        self.relative_positional = None

  def forward(self, x):
    """Runs the multi-head self-attention layer.

    Args:
      x: the input to the layer, a tensor of shape [length, batch_size, d_model]
    Returns:
      A single tensor containing the output from this layer
    """

    q = torch.einsum('tbf,hfa->bhta', x, self.w_q)
    k = torch.einsum('tbf,hfa->bhta', x, self.w_k)
    v = torch.einsum('tbf,hfa->bhta', x, self.w_v)
    logits = torch.einsum('bhqa,bhka->bhqk', q, k) / (self.d_qkv ** 0.5)

    if self.relative_positional is not None:
        q_pos = q.permute(2,0,1,3) #bhqd->qbhd
        l,b,h,d = q_pos.size()
        position_logits, _ = self.relative_positional(q_pos.reshape(l,b*h,d))
        # (bh)qk
        logits = logits + position_logits.view(b,h,l,l)

    probs = F.softmax(logits, dim=-1)
    probs = self.dropout(probs)
    o = torch.einsum('bhqk,bhka->bhqa', probs, v)
    out = torch.einsum('bhta,haf->tbf', o, self.w_o)
    return out


class LearnedRelativePositionalEmbedding(nn.Module):
    # from https://github.com/pytorch/fairseq/pull/2225/commits/a7fb63f2b84d5b20c8855e9c3372a95e5d0ea073
    """
    This module learns relative positional embeddings up to a fixed
    maximum size. These are masked for decoder and unmasked for encoder
    self attention.
    By default the embeddings are added to keys, but could be added to
    values as well.
    Args:
        max_relative_pos (int): the maximum relative positions to compute embeddings for
        num_heads (int): number of attention heads
        embedding_dim (int): depth of embeddings
        unmasked (bool): if the attention is unmasked (for transformer encoder)
        heads_share_embeddings (bool): if heads share the same relative positional embeddings
        add_to_values (bool): compute embeddings to be added to values as well
    """

    def __init__(
            self,
            max_relative_pos: int,
            num_heads: int,
            embedding_dim: int,
            unmasked: bool = False,
            heads_share_embeddings: bool = False,
            add_to_values: bool = False):
        super().__init__()
        self.max_relative_pos = max_relative_pos
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.unmasked = unmasked
        self.heads_share_embeddings = heads_share_embeddings
        self.add_to_values = add_to_values
        num_embeddings = (
            2 * max_relative_pos - 1
            if unmasked
            else max_relative_pos
        )
        embedding_size = (
            [num_embeddings, embedding_dim, 1]
            if heads_share_embeddings
            else [num_heads, num_embeddings, embedding_dim, 1]
        )
        if add_to_values:
            embedding_size[-1] = 2
        initial_stddev = embedding_dim**(-0.5)
        self.embeddings = nn.Parameter(torch.zeros(*embedding_size))
        nn.init.normal_(self.embeddings, mean=0.0, std=initial_stddev)

    def forward(self, query, saved_state=None):
        """
        Computes relative positional embeddings to be added to keys (and optionally values),
        multiplies the embeddings for keys with queries to create positional logits,
        returns the positional logits, along with embeddings for values (optionally)
        which could be added to values outside this module.
        Args:
            query (torch.Tensor): query tensor
            saved_state (dict): saved state from previous time step
        Shapes:
            query: `(length, batch_size*num_heads, embed_dim)`
        Returns:
            tuple(torch.Tensor):
                - positional logits
                - relative positional embeddings to be added to values
        """
        # During inference when previous states are cached
        if saved_state is not None and "prev_key" in saved_state:
            assert not self.unmasked, "This should only be for decoder attention"
            length = saved_state["prev_key"].shape[-2] + 1  # `length - 1` keys are cached,
                                                            # `+ 1` for the current time step
            decoder_step = True
        else:
            length = query.shape[0]
            decoder_step = False

        used_embeddings = self.get_embeddings_for_query(length)

        values_embeddings = (
            used_embeddings[..., 1]
            if self.add_to_values
            else None
        )
        positional_logits = self.calculate_positional_logits(query, used_embeddings[..., 0])
        positional_logits = self.relative_to_absolute_indexing(positional_logits, decoder_step)
        return (positional_logits, values_embeddings)

    def get_embeddings_for_query(self, length):
        """
        Extract the required embeddings. The maximum relative position between two time steps is
        `length` for masked case or `2*length - 1` for the unmasked case. If `length` is greater than
        `max_relative_pos`, we first pad the embeddings tensor with zero-embeddings, which represent
        embeddings when relative position is greater than `max_relative_pos`. In case `length` is
        less than `max_relative_pos`, we don't use the first `max_relative_pos - length embeddings`.
        Args:
            length (int): length of the query
        Returns:
            torch.Tensor: embeddings used by the query
        """
        pad_length = max(length - self.max_relative_pos, 0)
        start_pos = max(self.max_relative_pos - length, 0)
        if self.unmasked:
            with torch.no_grad():
                padded_embeddings = nn.functional.pad(
                    self.embeddings,
                    (0, 0, 0, 0, pad_length, pad_length)
                )
            used_embeddings = padded_embeddings.narrow(-3, start_pos, 2*length - 1)
        else:
            with torch.no_grad():
                padded_embeddings = nn.functional.pad(
                    self.embeddings,
                    (0, 0, 0, 0, pad_length, 0)
                )
            used_embeddings = padded_embeddings.narrow(-3, start_pos, length)
        return used_embeddings

    def calculate_positional_logits(self, query, relative_embeddings):
        """
        Multiplies query with the relative positional embeddings to create relative
        positional logits
        Args:
            query (torch.Tensor): Input tensor representing queries
            relative_embeddings (torch.Tensor): relative embeddings compatible with query
        Shapes:
            query: `(length, batch_size*num_heads, embed_dim)` if heads share embeddings
                   else `(length, batch_size, num_heads, embed_dim)`
            relative_embeddings: `(max_allowed_relative_positions, embed_dim)` if heads share embeddings
                                 else `(num_heads, max_allowed_relative_positions, embed_dim)`
                                 where `max_allowed_relative_positions` is `length` if masked
                                 else `2*length - 1`
        Returns:
            torch.Tensor: relative positional logits
        """
        if self.heads_share_embeddings:
            positional_logits = torch.einsum("lbd,md->lbm", query, relative_embeddings)
        else:
            query = query.view(query.shape[0], -1, self.num_heads, self.embedding_dim)
            positional_logits = torch.einsum("lbhd,hmd->lbhm", query, relative_embeddings)
            positional_logits = positional_logits.contiguous().view(
                positional_logits.shape[0], -1, positional_logits.shape[-1]
            )
        # mask out tokens out of range
        length = query.size(0)
        if length > self.max_relative_pos:
            # there is some padding
            pad_length = length - self.max_relative_pos
            positional_logits[:,:,:pad_length] -= 1e8
            if self.unmasked:
                positional_logits[:,:,-pad_length:] -= 1e8
        return positional_logits

    def relative_to_absolute_indexing(self, x, decoder_step):
        """
        Index tensor x (relative positional logits) in terms of absolute positions
        rather than relative positions. Last dimension of x represents relative position
        with respect to the first dimension, whereas returned tensor has both the first
        and last dimension indexed with absolute positions.
        Args:
            x (torch.Tensor): positional logits indexed by relative positions
            decoder_step (bool): is this is a single decoder step (during inference)
        Shapes:
            x: `(length, batch_size*num_heads, length)` for masked case or
               `(length, batch_size*num_heads, 2*length - 1)` for unmasked
        Returns:
            torch.Tensor: positional logits represented using absolute positions
        """
        length, bsz_heads, _ = x.shape

        if decoder_step:
            return x.contiguous().view(bsz_heads, 1, -1)

        if self.unmasked:
            x = nn.functional.pad(
                x,
                (0, 1)
            )
            x = x.transpose(0, 1)
            x = x.contiguous().view(bsz_heads, length * 2 * length)
            x = nn.functional.pad(
                x,
                (0, length - 1)
            )
            # Reshape and slice out the padded elements.
            x = x.view(bsz_heads, length + 1, 2*length - 1)
            return x[:, :length, length-1:]
        else:
            x = nn.functional.pad(
                x,
                (1, 0)
            )
            x = x.transpose(0, 1)
            x = x.contiguous().view(bsz_heads, length+1, length)
            return x[:, 1:, :]
