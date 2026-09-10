import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, decoder_hidden_dim, encoder_hidden_dim, attn_dim):
        super().__init__()
        self.W1 = nn.Linear(decoder_hidden_dim, attn_dim, bias=False)
        self.W2 = nn.Linear(encoder_hidden_dim, attn_dim, bias=False)
        self.v= nn.Linear(attn_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask):
        decoder_hidden_unsqueezed=decoder_hidden.unsqueeze(1)
        energy=torch.tanh(self.W1(decoder_hidden_unsqueezed)+self.W2(encoder_outputs))
        scores=self.v(energy).squeeze(-1)
        scores=scores.masked_fill(~mask,float('-inf'))
        weights=F.softmax(scores,dim=1)
        context=torch.bmm(weights.unsqueeze(1),encoder_outputs).squeeze(1)
        return context, weights
    # -------------------------
# TEST
# -------------------------

batch_size = 2
encoder_seq_len = 5
decoder_hidden_dim = 8
encoder_hidden_dim = 10
attn_dim = 16

attention = BahdanauAttention(
    decoder_hidden_dim,
    encoder_hidden_dim,
    attn_dim
)

decoder_hidden = torch.randn(
    batch_size,
    decoder_hidden_dim
)

encoder_outputs = torch.randn(
    batch_size,
    encoder_seq_len,
    encoder_hidden_dim
)

mask = torch.tensor([
    [True, True, True, True, True],
    [True, True, True, False, False]
])

context, weights = attention(
    decoder_hidden,
    encoder_outputs,
    mask
)

print("Context shape:", context.shape)
print("Weights shape:", weights.shape)

print("\nContext:")
print(context)

print("\nAttention weights:")
print(weights)

print("\nSum of weights:")
print(weights.sum(dim=1))