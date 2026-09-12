import torch
import torch.nn as nn
import torch.optim as optim
from encoder import Encoder
from decoder import Decoder
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import sentencepiece as spm


def load_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            source, target = line.split("\t")
            pairs.append((source, target))
    return pairs


class SentencePieceDataset(Dataset):
    def __init__(self, pairs, sp):
        super().__init__()
        self.pairs = pairs
        self.sp = sp

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        src, tgt = self.pairs[index]
        src_ids = self.sp.encode(src)
        tgt_ids = self.sp.encode(tgt, add_bos=True, add_eos=True)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch):
    src, tgt = zip(*batch)
    padded_srcs = pad_sequence(src, batch_first=True, padding_value=0)
    padded_tgts = pad_sequence(tgt, batch_first=True, padding_value=0)
    return padded_srcs, padded_tgts


def combine_bidirectional(h_or_c, num_layers):
    h_or_c = h_or_c.view(num_layers, 2, h_or_c.size(1), h_or_c.size(2))
    combined = h_or_c[:, 0, :, :] + h_or_c[:, 1, :, :]
    return combined

pad_id = 0
vocab_size = 8000
sp = spm.SentencePieceProcessor(model_file="tokenizer/ur_sp.model")

train_pairs = load_pairs("data/train.tsv")
valid_pairs = load_pairs("data/valid.tsv")
train_dataset = SentencePieceDataset(train_pairs, sp)
valid_dataset = SentencePieceDataset(valid_pairs, sp)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

encoder = Encoder()
decoder = Decoder(vocabulary_size=8000, embedding_dimension=256, hidden_dimensions=512,
                   encoder_hidden_dimension=1024, numOfLayers=2, dropout=0.3, attention_dimension=512)
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
loss_criteria = nn.CrossEntropyLoss(ignore_index=pad_id)

total_parameters = sum(p.numel() for p in encoder.parameters()) + sum(p.numel() for p in decoder.parameters())
print(f"Total parameters: {total_parameters}")
best_valid_loss = float('inf')

for epoch in range(10):
    print(f"Epoch {epoch}")
    encoder.train()
    decoder.train()
    total_train_loss = 0

    for src_batch, tgt_batch in train_loader:
        optimizer.zero_grad()
        encoder_outputs, (h_n, c_n) = encoder(src_batch)
        initial_hidden = combine_bidirectional(h_n, 2)
        initial_cell = combine_bidirectional(c_n, 2)
        mask = (src_batch != 0)
        outputs, hidden, cell = decoder(tgt_batch, encoder_outputs, initial_hidden, initial_cell, mask)
        loss = loss_criteria(outputs.reshape(-1, vocab_size), tgt_batch[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    encoder.eval()
    decoder.eval()
    total_valid_loss = 0

    with torch.no_grad():
        for src_batch, tgt_batch in valid_loader:
            encoder_outputs, (h_n, c_n) = encoder(src_batch)
            initial_hidden = combine_bidirectional(h_n, 2)
            initial_cell = combine_bidirectional(c_n, 2)
            mask = (src_batch != 0)
            outputs, hidden, cell = decoder(tgt_batch, encoder_outputs, initial_hidden, initial_cell, mask)
            loss = loss_criteria(outputs.reshape(-1, vocab_size), tgt_batch[:, 1:].reshape(-1))
            total_valid_loss += loss.item()
    avg_valid_loss = total_valid_loss / len(valid_loader)
    print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, valid_loss={avg_valid_loss:.4f}")

    if avg_valid_loss < best_valid_loss:
        best_valid_loss = avg_valid_loss
        torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()}, "best_checkpoint.pt")
        print("Saved new best checkpoint")