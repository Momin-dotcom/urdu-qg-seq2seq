import os
import sentencepiece as spm

ANS_OPEN = "<ans>"
ANS_CLOSE = "</ans>"

os.makedirs("tokenizer", exist_ok=True)  # ensure it exists

# --- Load train pairs from Task 1 output ---
train_pairs = []
with open("data/train.tsv", "r", encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        source, target = line.split("\t")
        train_pairs.append((source, target))

print(f"loaded {len(train_pairs)} train pairs")

# --- 2.1: Write corpus file (train split only, sources + targets) ---
with open("tokenizer/sp_corpus.txt", "w", encoding="utf-8") as f:
    for src, tgt in train_pairs:
        f.write(src + "\n")
        f.write(tgt + "\n")

# --- 2.1/2.2: Train SentencePiece, vocab 8k, register special tokens ---
spm.SentencePieceTrainer.train(
    input="tokenizer/sp_corpus.txt",
    model_prefix="tokenizer/ur_sp",
    vocab_size=8000,
    model_type="unigram",
    character_coverage=1.0,          # keep every Urdu character
    user_defined_symbols=[ANS_OPEN, ANS_CLOSE],
    pad_id=0, unk_id=1, bos_id=2, eos_id=3,                  # reproducible across reruns
)

sp = spm.SentencePieceProcessor(model_file="tokenizer/ur_sp.model")
PAD, UNK, BOS, EOS = 0, 1, 2, 3

# --- Round-trip sanity check ---
src, tgt = train_pairs[0]
print(sp.encode(src, out_type=str))       # pieces
print(sp.encode(tgt))                     # ids
print(sp.decode(sp.encode(tgt)) == tgt)   # should be True
