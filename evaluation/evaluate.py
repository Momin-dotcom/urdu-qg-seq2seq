import torch
import torch.nn as nn
import math
import sys
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

sys.path.append("model")
from decode import encoder, decoder, tokenizer, device, pad_id, sos_id, eos_id, max_len, combine_bidirectional

vocab_size = 8000


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


class SentencePieceDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, sp):
        super().__init__()
        self.pairs = pairs
        self.sp = sp

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        src, tgt = self.pairs[index]
        src_ids = self.sp.encode(src, out_type=int)
        tgt_ids = self.sp.encode(tgt, out_type=int, add_bos=True, add_eos=True)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch):
    src, tgt = zip(*batch)
    padded_srcs = pad_sequence(src, batch_first=True, padding_value=pad_id)
    padded_tgts = pad_sequence(tgt, batch_first=True, padding_value=pad_id)
    return padded_srcs, padded_tgts


valid_pairs = load_pairs("data/valid.tsv")
valid_dataset = SentencePieceDataset(valid_pairs, tokenizer)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

# ---- Perplexity nikaalo ----
loss_criteria = nn.CrossEntropyLoss(ignore_index=pad_id)
total_valid_loss = 0

with torch.no_grad():
    for src_batch, tgt_batch in valid_loader:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)
        encoder_outputs, (h_n, c_n) = encoder(src_batch)
        initial_hidden = combine_bidirectional(h_n, 2)
        initial_cell = combine_bidirectional(c_n, 2)
        mask = (src_batch != pad_id)
        outputs, hidden, cell = decoder(tgt_batch, encoder_outputs, initial_hidden, initial_cell, mask)
        loss = loss_criteria(outputs.reshape(-1, vocab_size), tgt_batch[:, 1:].reshape(-1))
        total_valid_loss += loss.item()

avg_valid_loss = total_valid_loss / len(valid_loader)
perplexity = math.exp(avg_valid_loss)

print(f"Validation loss: {avg_valid_loss:.4f}")
print(f"Perplexity: {perplexity:.4f}")


# ---- Greedy decoding (already-tagged sentences ke liye) ----
def encode_tagged_source(tagged_sentence):
    ids = tokenizer.encode(tagged_sentence, out_type=int)
    input_ids = torch.tensor([ids], device=device)
    mask = (input_ids != pad_id)
    return input_ids, mask


def greedy_decode_tagged(tagged_sentence):
    input_ids, mask = encode_tagged_source(tagged_sentence)
    with torch.no_grad():
        encoder_outputs, (h_n, c_n) = encoder(input_ids)
        hidden = combine_bidirectional(h_n, 2)
        cell = combine_bidirectional(c_n, 2)
        input_token = torch.tensor([sos_id], device=device)
        generated_ids = []
        for step in range(max_len):
            logits, hidden, cell, weights = decoder.forward_step(input_token, hidden, cell, encoder_outputs, mask)
            next_id = logits.argmax(dim=-1)
            if next_id.item() == eos_id:
                break
            generated_ids.append(next_id.item())
            input_token = next_id
    return tokenizer.decode(generated_ids)


# ---- Scoring (manual ka diya hua) ----
import sacrebleu
from rouge_score import rouge_scorer


def score(hyps, refs):
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    rl = sum(scorer.score(r, h)["rougeL"].fmeasure
             for h, r in zip(hyps, refs)) / len(refs)
    unk_rate = sum(h.count("\u2047") for h in hyps) / max(
        1, sum(len(h.split()) for h in hyps))
    return {"BLEU-4": bleu, "ROUGE-L": rl, "unk_rate": unk_rate}


hyps = []
refs = []
for source, target in valid_pairs[:50]:
    generated_question = greedy_decode_tagged(source)
    hyps.append(generated_question)
    refs.append(target)

results = score(hyps, refs)
print("BLEU/ROUGE/unk_rate (first 50 examples):", results)