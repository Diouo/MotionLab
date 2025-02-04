import os
from typing import List, Union

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, T5EncoderModel

from rfmotion.models.operator import PositionalEncoding
from rfmotion.utils.temos_utils import lengths_to_mask


class RFMotionTextEncoder(nn.Module):

    def __init__(
        self,
        modelpath: str,
        finetune: bool = False,
        last_hidden_state: bool = True,
        max_length: int = 30,
    ) -> None:

        super().__init__()

        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(modelpath,clean_up_tokenization_spaces = True)
        if "clip" in modelpath:
            self.text_model = AutoModel.from_pretrained(modelpath)
        elif "t5" in modelpath:
            self.text_model = T5EncoderModel.from_pretrained(modelpath)

        # Don't train the model
        if not finetune:
            self.text_model.training = False
            for p in self.text_model.parameters():
                p.requires_grad = False

        # Then configure the model
        if "clip" in modelpath:
            self.text_encoded_dim = self.text_model.config.text_config.hidden_size
            if last_hidden_state:
                self.name = "clip_hidden"
            else:
                self.name = "clip"
        elif "t5" in modelpath:
            self.text_encoded_dim = self.text_model.config.d_model
            self.name = "t5"
        else:
            raise ValueError(f"Model {modelpath} not supported")

    def forward(self, texts: List[str]):
        # get prompt text embeddings
        if self.name in ["clip", "clip_hidden",'t5']:
            text_input_ids = self.tokenizer(texts,padding="max_length",truncation=True,max_length=self.max_length,return_tensors="pt",).input_ids
            # split into max length Clip can handle
            if text_input_ids.shape[-1] > self.tokenizer.model_max_length:
                text_input_ids = text_input_ids[:, :self.tokenizer.model_max_length]

        # use pooled output if latent dim is two-dimensional
        # pooled = 0 if self.latent_dim[0] == 1 else 1 # (bs, seq_len, text_encoded_dim) -> (bs, text_encoded_dim)
        # text encoder forward, clip must use get_text_features
        if self.name == "clip":
            # (batch_Size, text_encoded_dim)
            text_embeddings = self.text_model.get_text_features(text_input_ids.to(self.text_model.device))
            # (batch_Size, 1, text_encoded_dim)
            text_embeddings = text_embeddings.unsqueeze(1)
            return (text_embeddings,)
        elif self.name == "clip_hidden":
            # (batch_Size, text_encoded_dim)
            text_embeddings = self.text_model.get_text_features(text_input_ids.to(self.text_model.device))
            # (batch_Size, 1, text_encoded_dim)
            text_embeddings = text_embeddings.unsqueeze(1)
            # (batch_Size, seq_length , text_encoded_dim)
            text_last_embeddings = self.text_model.text_model(text_input_ids.to(self.text_model.device)).last_hidden_state
            return (text_embeddings, text_last_embeddings)
        elif self.name == "t5":
            text_last_embeddings = self.text_model(text_input_ids.to(self.text_model.device)).last_hidden_state   
            return (text_last_embeddings,)
        else:
            raise NotImplementedError(f"Model {self.name} not implemented")
