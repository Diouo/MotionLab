from typing import List, Optional, Union
from omegaconf import OmegaConf

import numpy as np
import rotary_embedding_torch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.distribution import Distribution


from rfmotion.models.operator.cross_attention import _get_activation_fn, _get_clones, _get_clone
from rfmotion.models.operator.position_encoding import build_position_encoding
from rfmotion.utils.temos_utils import lengths_to_mask, lengths_to_query_mask
from rfmotion.models.operator.rope import RotaryEmbedding


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.d_model = d_model
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def forward_post(self,
                     q,
                     k,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


    def forward(self, q, k, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        return self.forward_post(q, k, src, src_mask, src_key_padding_mask, pos)


class SkipTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm):
        super().__init__()
        self.d_model = encoder_layer.d_model
        self.rope = RotaryEmbedding(dim = self.d_model)
        self.num_layers = num_layers
        self.norm = norm

        assert num_layers % 2 == 1

        num_block = (num_layers-1)//2
        self.input_blocks = _get_clones(encoder_layer, num_block)
        self.middle_block = _get_clone(encoder_layer)
        self.output_blocks = _get_clones(encoder_layer, num_block)
        self.linear_blocks = _get_clones(nn.Linear(2*self.d_model, self.d_model), num_block)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def pos_embed_mid(self, tensor, nframes, nframes_z, offset):
        tensor = tensor.unsqueeze(1)
        if nframes_z is not None:
            q_list = []
            k_list = []

            # mean query
            q1_list = []
            k1_list = []
            for i in range(tensor.shape[0]):
                q1 = self.rope.rotate_queries_or_keys(tensor[i, :, :nframes_z, :].unsqueeze(0), offset[i])
                k1 = self.rope.rotate_queries_or_keys(tensor[i, :, :nframes_z, :].unsqueeze(0), offset[i])
                q1_list.append(q1)
                k1_list.append(k1)
            q1_tensor = torch.cat(q1_list, 0)
            k1_tensor = torch.cat(k1_list, 0)
            q_list.append(q1_tensor)
            k_list.append(k1_tensor)

            # x
            q2 = self.rope.rotate_queries_or_keys(tensor[:, :, nframes_z:-nframes_z, :])
            k2 = self.rope.rotate_queries_or_keys(tensor[:, :, nframes_z:-nframes_z, :])
            q_list.append(q2)
            k_list.append(k2)

            # std query
            q3_list = []
            k3_list = []
            for i in range(tensor.shape[0]):
                q3 = self.rope.rotate_queries_or_keys(tensor[i, :, -nframes_z:, :].unsqueeze(0), offset[i])
                k3 = self.rope.rotate_queries_or_keys(tensor[i, :, -nframes_z:, :].unsqueeze(0), offset[i])
                q3_list.append(q3)
                k3_list.append(k3)
            q3_tensor = torch.cat(q3_list, 0)
            k3_tensor = torch.cat(k3_list, 0)
            q_list.append(q3_tensor)
            k_list.append(k3_tensor)

            q_tensor = torch.cat(q_list, 2).squeeze().permute(1, 0, 2)
            k_tensor = torch.cat(k_list, 2).squeeze().permute(1, 0, 2)

        elif nframes is not None:
            q_list = []
            k_list = []

            # z
            q1_list = []
            k1_list = []
            for i in range(tensor.shape[0]):
                q1 = self.rope.rotate_queries_or_keys(tensor[i, :, :-nframes, :].unsqueeze(0), offset[i])
                k1 = self.rope.rotate_queries_or_keys(tensor[i, :, :-nframes, :].unsqueeze(0), offset[i])
                q1_list.append(q1)
                k1_list.append(k1)
            q1_tensor = torch.cat(q1_list, 0)
            k1_tensor = torch.cat(k1_list, 0)
            q_list.append(q1_tensor)
            k_list.append(k1_tensor)

            # x query
            q2 = self.rope.rotate_queries_or_keys(tensor[:, :, -nframes:, :])
            k2 = self.rope.rotate_queries_or_keys(tensor[:, :, -nframes:, :])
            q_list.append(q2)
            k_list.append(k2)

            q_tensor = torch.cat(q_list, 2).squeeze().permute(1, 0, 2)
            k_tensor = torch.cat(k_list, 2).squeeze().permute(1, 0, 2)

        return q_tensor, k_tensor
    
    def pos_embed(self, tensor, nframes, nframes_z, offset):
        tensor = tensor.unsqueeze(1)
        if nframes_z is not None:
            q_list = []
            k_list = []

            # mean query
            q1 = self.rope.rotate_queries_or_keys(tensor[:, :, :nframes_z, :])
            k1 = self.rope.rotate_queries_or_keys(tensor[:, :, :nframes_z, :])
            q_list.append(q1)
            k_list.append(k1)

            # x
            q2 = self.rope.rotate_queries_or_keys(tensor[:, :, nframes_z:-nframes_z, :])
            k2 = self.rope.rotate_queries_or_keys(tensor[:, :, nframes_z:-nframes_z, :])
            q_list.append(q2)
            k_list.append(k2)

            # std query
            q3 = self.rope.rotate_queries_or_keys(tensor[:, :, -nframes_z:, :])
            k3 = self.rope.rotate_queries_or_keys(tensor[:, :, -nframes_z:, :])
            q_list.append(q3)
            k_list.append(k3)

            q_tensor = torch.cat(q_list, 2).squeeze().permute(1, 0, 2)
            k_tensor = torch.cat(k_list, 2).squeeze().permute(1, 0, 2)

        elif nframes is not None:
            q_list = []
            k_list = []

            # z
            q1 = self.rope.rotate_queries_or_keys(tensor[:, :, :-nframes, :])
            k1 = self.rope.rotate_queries_or_keys(tensor[:, :, :-nframes, :])
            q_list.append(q1)
            k_list.append(k1)

            # x query
            q2 = self.rope.rotate_queries_or_keys(tensor[:, :, -nframes:, :])
            k2 = self.rope.rotate_queries_or_keys(tensor[:, :, -nframes:, :])
            q_list.append(q2)
            k_list.append(k2)

            q_tensor = torch.cat(q_list, 2).squeeze().permute(1, 0, 2)
            k_tensor = torch.cat(k_list, 2).squeeze().permute(1, 0, 2)

        return q_tensor, k_tensor 

    def forward(self, x,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                nframes=None,
                nframes_z=None,
                offset=None):
        xs = []
        for module in self.input_blocks:
            q, k = self.pos_embed(x,nframes,nframes_z,offset)
            x = x.permute(1, 0, 2)
            x = module(q, k, x, src_mask=mask,src_key_padding_mask=src_key_padding_mask, pos=pos)
            x = x.permute(1, 0, 2)
            xs.append(x)

        q, k = self.pos_embed(x,nframes,nframes_z,offset)
        x = x.permute(1, 0, 2)
        x = self.middle_block(q, k, x, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        x = x.permute(1, 0, 2)

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            q, k = self.pos_embed(x,nframes,nframes_z,offset)
            x = x.permute(1, 0, 2)
            x = module(q, k, x, src_mask=mask,src_key_padding_mask=src_key_padding_mask, pos=pos)
            x = x.permute(1, 0, 2)

        x = x.permute(1, 0, 2)
        x = self.norm(x)
        return x


class MldVae(nn.Module):

    def __init__(self,
                 ablation,
                 nfeats: int,
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 **kwargs) -> None:

        super().__init__()

        self.latent_size = latent_dim[0]
        self.latent_dim = latent_dim[-1]
        input_feats = nfeats
        output_feats = nfeats

        self.pe_type = ablation.VAE_PE_TYPE
        self.pe_dim = ablation.VAE_PE_DIM

        encoder_layer = TransformerEncoderLayer(self.latent_dim,num_heads,ff_size,dropout,activation,normalize_before)
        encoder_norm = nn.LayerNorm(self.latent_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.encode_embedding = nn.Linear(input_feats, self.latent_dim)
        self.mean_query = nn.Parameter(torch.randn(1, 1, self.latent_dim))
        self.std_query = nn.Parameter(torch.randn(1, 1, self.latent_dim))

        decoder_norm = nn.LayerNorm(self.latent_dim)
        self.decoder = SkipTransformerEncoder(encoder_layer, num_layers, decoder_norm)
        self.decode_embedding = nn.Linear(self.latent_dim, self.latent_dim)
        self.x_query = nn.Parameter(torch.randn(1, 1, self.latent_dim))
        self.final_layer = nn.Linear(self.latent_dim, output_feats)

    def get_offet_for_rope(self, length, length_z):
        assert len(length) == len(length_z)
        offset = []
        for l, lz in zip(length, length_z):
            offset.append(int((l-lz)//2))
        return offset

    def encode(self, x: Tensor, lengths: List[int], lengths_z: Optional[List[int]] = None) -> Union[Tensor, Distribution]:
        bs = x.shape[0]
        nframes = x.shape[1]
        nframes_z = max(lengths_z)
        offset = self.get_offet_for_rope(lengths, lengths_z)

        # token mask
        x_mask = lengths_to_mask(lengths, x.device, max_len=nframes)
        mean_query_mask = lengths_to_mask(lengths_z, x.device, max_len=nframes_z)
        std_query_mask = lengths_to_mask(lengths_z, x.device, max_len=nframes_z)
        aug_mask = torch.cat((mean_query_mask, x_mask, std_query_mask), 1)

        # input tokens
        x = self.encode_embedding(x)
        mean_query = torch.tile(self.mean_query, (bs, max(lengths_z), 1))
        std_query = torch.tile(self.std_query, (bs, max(lengths_z), 1))
        xseq = torch.cat((mean_query, x, std_query), 1)

        # encode xseq
        dist = self.encoder(xseq,src_key_padding_mask=~aug_mask, nframes_z=nframes_z, offset=offset) # [3*nframes, bs, dim]

        # content distribution
        mu = dist[0:nframes_z, ...]
        mu[~mean_query_mask.T, ...]= 0
        logvar = dist[-nframes_z:, ...]
        logvar[~std_query_mask.T, ...]= 0

        # resampling
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()

        return latent, dist

    def decode(self, z: Tensor, lengths: List[int], lengths_z: Optional[List[int]] = None):
        z=z.permute(1, 0, 2)

        bs = z.shape[1]
        nframes = max(lengths)
        nframes_z = z.shape[0]
        offset = self.get_offet_for_rope(lengths, lengths_z)    

        # token mask 
        x_mask = lengths_to_mask(lengths, z.device, max_len=nframes)
        z_mask = lengths_to_mask(lengths_z, z.device, max_len=nframes_z)
        augmask = torch.cat((z_mask, x_mask), axis=1)

        # input tokens
        z = self.decode_embedding(z.permute(1, 0, 2))
        x_query = torch.tile(self.x_query, (bs, nframes, 1))
        xseq = torch.cat((z, x_query), axis=1)

        # decode xseq
        output = self.decoder(xseq, src_key_padding_mask=~augmask, nframes=nframes, offset=offset)[nframes_z:]
        output = self.final_layer(output)
        output[~x_mask.T] = 0  # zero for padded area
        x = output.permute(1, 0, 2) # [nframes, bs, dim] => [bs, nframes, dim]

        return x
