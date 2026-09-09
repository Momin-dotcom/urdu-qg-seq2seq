import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(num_embeddings=8000, embedding_dim=256)
        self.dropout=nn.Dropout( p=0.3, inplace=False)
        self.lstm=nn.LSTM( input_size=256, hidden_size=512, num_layers=2,bidirectional=True,batch_first=True)

    def forward(self, x):
        x=self.embedding(x)
        x=self.dropout(x)
        outputs,(h_n,c_n)=self.lstm(x)
        return outputs, (h_n, c_n)

