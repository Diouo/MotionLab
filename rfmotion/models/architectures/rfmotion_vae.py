import math
from typing import List, Optional, Union
from omegaconf import OmegaConf

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions.distribution import Distribution

from rfmotion.models.operator.cross_attention import (
    SkipTransformerEncoder,
    SkipTransformerDecoder,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from rfmotion.models.operator.position_encoding import build_position_encoding
from rfmotion.utils.temos_utils import lengths_to_mask


class RFMotionVae(nn.Module):

    def __init__(self,
                 ablation,
                 nfeats: int,
                 latent_dim: list = [4, 256],
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
        self.query_pos_encoder = build_position_encoding(self.latent_dim, position_embedding=self.pe_type)
        self.query_pos_decoder = build_position_encoding(self.latent_dim, position_embedding=self.pe_type)

        encoder_layer = TransformerEncoderLayer(self.latent_dim,num_heads,ff_size,dropout,activation,normalize_before,)
        encoder_norm = nn.LayerNorm(self.latent_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.encode_embedding = nn.Linear(input_feats, self.latent_dim)
        self.encode_mean_embedding = nn.Linear(self.latent_dim, self.latent_dim)
        self.encode_std_embedding = nn.Linear(self.latent_dim, self.latent_dim)

        decoder_norm = nn.LayerNorm(self.latent_dim)
        self.decoder = SkipTransformerEncoder(encoder_layer, num_layers, decoder_norm)
        self.decode_embedding = nn.Linear(self.latent_dim, self.latent_dim)
        self.decode_mean_embedding = nn.Linear(self.latent_dim, self.latent_dim)
        self.decode_std_embedding = nn.Linear(self.latent_dim, self.latent_dim)
        self.final_layer = nn.Linear(self.latent_dim, output_feats)

    def encode(self, x: Tensor, lengths: List[int], lengths_z: List[int]) -> Union[Tensor, Distribution]:
        bs, nframes_z = x.shape[0], max(lengths_z)
        mask = lengths_to_mask(lengths, x.device)
        mask_z = lengths_to_mask(lengths_z, x.device)

        x = self.encode_embedding(x)  # Embed each human poses into latent vectors
        x = x.permute(1, 0, 2)  # [bs, nframes, latent_dim] -> [nframes, bs, latent_dim]
        mean_queries = torch.zeros(nframes_z, bs, self.latent_dim, device=x.device)
        mean_queries = self.encode_mean_embedding(mean_queries)
        std_queries = torch.zeros(nframes_z, bs, self.latent_dim, device=x.device)
        std_queries = self.encode_std_embedding(std_queries)

        x = torch.cat((mean_queries, std_queries, x), axis=0)
        augmask = torch.cat((mask_z, mask_z, mask), axis=1)
        pos = self.query_pos_encoder(x)

        dist = self.encoder(x,src_key_padding_mask=~augmask,pos=pos)[:nframes_z*2]
        mu = dist[0:nframes_z, ...]
        logvar = dist[nframes_z:, ...]

        # resampling
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        z = dist.rsample()
        return z, dist

    def decode(self, z: Tensor, lengths: List[int], lengths_z: List[int]):
        bs, nframes = z.shape[1], max(lengths)
        mask = lengths_to_mask(lengths, z.device)
        # mask_z = lengths_to_mask(lengths_z, z.device)

        pad = torch.zeros(z.shape[0], bs, self.latent_dim, device=z.device)
        queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)
        queries = self.decode_embedding(queries)
        xseq = torch.cat((z, pad, queries), axis=0)

        z_mask = torch.ones((bs, z.shape[0]),dtype=bool,device=z.device)
        pad_mask = torch.zeros((bs, z.shape[0]),dtype=bool,device=z.device)
        augmask = torch.cat((z_mask, pad_mask, mask), axis=1)
        pos = self.query_pos_decoder(xseq)
        output = self.decoder(xseq, src_key_padding_mask=~augmask, pos=pos)[z.shape[0]*2:]
        output = self.final_layer(output)

        output[~mask.T] = 0 # zero for padded area
        feats = output.permute(1, 0, 2)  # [Sequence, Batch size, Latent Dim] -> [Batch size, Sequence, Latent Dim]
        return feats
