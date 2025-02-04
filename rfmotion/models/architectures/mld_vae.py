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
        self.query_pos_encoder = build_position_encoding(self.latent_dim, position_embedding=self.pe_type, embedding_dim=self.pe_dim)
        self.query_pos_decoder = build_position_encoding(self.latent_dim, position_embedding=self.pe_type, embedding_dim=self.pe_dim)

        encoder_layer = TransformerEncoderLayer(self.latent_dim,num_heads,ff_size,dropout,activation,normalize_before)
        encoder_norm = nn.LayerNorm(self.latent_dim)
        self.encoder = SkipTransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.global_motion_token = nn.Parameter(torch.randn(self.latent_size * 2, self.latent_dim))
        self.encode_embedding = nn.Linear(input_feats, self.latent_dim)

        decoder_norm = nn.LayerNorm(self.latent_dim)
        self.decoder = SkipTransformerEncoder(encoder_layer, num_layers, decoder_norm)
        self.query_embedding = nn.Linear(self.latent_dim, self.latent_dim)
        self.decode_embedding = nn.Linear(self.latent_dim, self.latent_dim)
        self.final_layer = nn.Linear(self.latent_dim, output_feats)


    def encode(self, x: Tensor, lengths: List[int], lengths_z: Optional[List[int]] = None) -> Union[Tensor, Distribution]:
        bs, nframes, nfeats = x.shape
        mask = lengths_to_mask(lengths, x.device)

        # input tokens
        x = x.permute(1, 0, 2)  # [bs, nframes, dim] => [nframes, bs, dim]
        x = self.encode_embedding(x)
        dist = torch.tile(self.global_motion_token[:, None, :], (1, bs, 1)) # Each batch has its own set of tokens
        xseq = torch.cat((dist, x), 0)

        # create a bigger mask, to allow attend to emb
        dist_masks = torch.ones((bs, dist.shape[0]),dtype=bool,device=x.device)
        aug_mask = torch.cat((dist_masks, mask), 1)

        # pos embedding 
        pos = self.query_pos_encoder(xseq)

        # encode xseq
        dist = self.encoder(xseq,src_key_padding_mask=~aug_mask,pos=pos)[:dist.shape[0]]

        # content distribution
        mu = dist[0:self.latent_size, ...]
        logvar = dist[self.latent_size:, ...]

        # resampling
        std = logvar.exp().pow(0.5)
        dist = torch.distributions.Normal(mu, std)
        latent = dist.rsample()

        return latent, dist

    def decode(self, z: Tensor, lengths: List[int], lengths_z: Optional[List[int]] = None):
        mask = lengths_to_mask(lengths, z.device)
        bs, nframes = mask.shape

        # input tokens
        z = self.decode_embedding(z)
        pad =  torch.zeros(z.shape[0], bs, self.latent_dim, device=z.device)
        queries = torch.zeros(nframes, bs, self.latent_dim, device=z.device)
        queries = self.query_embedding(queries)
        z_pad = torch.cat((z, pad), axis=0)
        xseq = torch.cat((z, pad, queries), axis=0)

        # pos embedding 
        pos = self.query_pos_decoder(xseq)

        # mask 
        z_mask = torch.ones((bs, self.latent_size),dtype=bool,device=z.device)
        pad_mask = torch.zeros((bs, self.latent_size),dtype=bool,device=z.device)
        augmask = torch.cat((z_mask, pad_mask, mask), axis=1)

        # decode xseq
        output = self.decoder(xseq, src_key_padding_mask=~augmask, pos=pos)[z.shape[0]*2:]
        output = self.final_layer(output)
        output[~mask.T] = 0  # zero for padded area
        x = output.permute(1, 0, 2) # [nframes, bs, dim] => [bs, nframes, dim]

        return x
