import torch
import torch.nn as nn
import math
import sys
import os
import re
import csv
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

sys.path.append("model")
from decode import (encoder, decoder, tokenizer, device, pad_id, sos_id, eos_id,
                     max_len, combine_bidirectional, beam_search_decode)

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



def extract_plain_and_answer(tagged_source):
    match = re.search(r"<ans>\s*(.*?)\s*</ans>", tagged_source)
    if match is None:
        return tagged_source, ""
    answer = match.group(1)
    plain = re.sub(r"<ans>\s*", "", tagged_source)
    plain = re.sub(r"\s*</ans>", "", plain)
    return plain, answer


#Scoring
import sacrebleu
from rouge_score import rouge_scorer


class WhitespaceTokenizer:
    def tokenize(self, text):
        return text.split()


def score(hyps, refs):
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False, tokenizer=WhitespaceTokenizer())
    rl = sum(scorer.score(r, h)["rougeL"].fmeasure
             for h, r in zip(hyps, refs)) / len(refs)
    unk_rate = sum(h.count("\u2047") for h in hyps) / max(
        1, sum(len(h.split()) for h in hyps))
    return {"BLEU-4": bleu, "ROUGE-L": rl, "unk_rate": unk_rate}

hyps_greedy = []
hyps_beam = []
refs = []

for source, target in valid_pairs[:50]:
    greedy_question = greedy_decode_tagged(source)
    hyps_greedy.append(greedy_question)
    refs.append(target)

    plain_sentence, answer_text = extract_plain_and_answer(source)
    if answer_text:
        beam_question = beam_search_decode(plain_sentence, answer_text)
    else:
        beam_question = ""
    hyps_beam.append(beam_question)

greedy_results = score(hyps_greedy, refs)
beam_results = score(hyps_beam, refs)

print("Greedy BLEU/ROUGE/unk_rate (first 50 examples):", greedy_results)
print("Beam BLEU/ROUGE/unk_rate (first 50 examples):", beam_results)

os.makedirs("results", exist_ok=True)

with open("results/samples.tsv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["source", "reference", "greedy", "beam"])
    for i, (source, target) in enumerate(valid_pairs[:50]):
        writer.writerow([source, target, hyps_greedy[i], hyps_beam[i]])

print("Saved results/samples.tsv")