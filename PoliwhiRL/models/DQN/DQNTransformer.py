# -*- coding: utf-8 -*-
import torch.nn as nn
from PoliwhiRL.models.CNN.GameBoy import GameBoyOptimizedCNN
from PoliwhiRL.models.transformers.positional_encoding import PositionalEncoding


class TransformerDQN(nn.Module):
    def __init__(self, input_shape, action_size, d_model=128, nhead=8, num_layers=8):
        super(TransformerDQN, self).__init__()
        self.action_size = action_size
        self.input_shape = input_shape
        self.d_model = d_model
        self.cnn = GameBoyOptimizedCNN(input_shape, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=1000)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.fc_out = nn.Linear(d_model, action_size)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.cnn(x)
        x = x.view(batch_size, seq_len, self.d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        q_values = self.fc_out(x)
        return q_values
