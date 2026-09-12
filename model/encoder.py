import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

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


#padding
def load_pairs(path):
    train_pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            source, target = line.split("\t")
            train_pairs.append((source, target))
    return train_pairs

class SentencePieceDataset(Dataset):
    def __init__(self,pairs,sp):
        super().__init__()
        self.pairs=pairs
        self.sp=sp

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        src, tgt = self.pairs[index]
        src_ids=self.sp.encode(src)            
        tgt_ids=self.sp.encode(tgt)
        return torch.tensor(src_ids, dtype=torch.long),torch.tensor(tgt_ids, dtype=torch.long)

def collate_fn(batch):
    src,tgt=zip(*batch)
    padded_srcs=pad_sequence(src,batch_first=True,padding_value=0)
    padded_tgts =pad_sequence(tgt,batch_first=True,padding_value=0)
    return padded_srcs,padded_tgts

pad_id=0
sp=spm.SentencePieceProcessor(model_file="tokenizer/ur_sp.model")
train_pairs=load_pairs(r"C:\Users\GOGI LAPTOP\OneDrive\Desktop\urdu-qg-seq2seq\data\train.tsv")
dataset=SentencePieceDataset(train_pairs,sp)
loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)




from attention import BahdanauAttention

encoder = Encoder()
attention = BahdanauAttention(decoder_hidden_dim=512, encoder_hidden_dim=1024, attn_dim=512)

