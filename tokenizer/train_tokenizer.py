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
print("\n--- 2.3: Five tokenized examples ---")
for i in range(5):
    src, tgt = train_pairs[i]
    src_pieces = sp.encode(src, out_type=str)
    tgt_pieces = sp.encode(tgt, out_type=str)
    print(f"\nExample {i+1}")
    print("Source:", src)
    print("Source pieces:", src_pieces)
    print("Target:", tgt)
    print("Target pieces:", tgt_pieces)
#commentary
# Most whole Urdu words stay as single pieces (e.g. "ہیوسٹن", "ٹیکساس", "پرفارم",
# "شہرت") rather than being broken down. Urdu already marks word boundaries with
# spaces, so the tokenizer's main job here is deciding which words are frequent
# enough to keep whole versus which need to be split into subwords.True morphological splitting shows up on inflected verb forms: "بڑھی" (grew,
# feminine) splits into "بڑھ" + "ی", separating the verb root from its feminine
# inflectional suffix. This is exactly the case a word-level vocabulary would
# struggle with, since every gender/tense variant of a root would need its own
# entry.The clearest splitting happens on loanwords and mixed-script content. "R&B"
# splits into "R", "&", "B" (essentially one piece per character), since the
# vocabulary wasn't built around Latin script. Transliterated English words split
# similarly: "گرل" (girl) becomes "گر" + "ل", and "بیبی" (baby) splits into two
# "بی" pieces. These aren't real morpheme boundaries, just artifacts of these
# transliterated forms being too rare in the training corpus to earn a full-word
# piece.Numbers ("1990", "2003", "100") stay intact as single pieces rather than being
# split digit by digit, likely because these specific values recur often enough
# across QA pairs (many reference dates) to justify dedicated vocabulary slots.
# Punctuation attached directly to a word with no preceding space merges into
# that word's piece (e.g. "کی۔", where the sentence-final period fuses onto
# "کی"). The <ans>/</ans> tags consistently resolve as their own atomic pieces
# in every example, confirming they were correctly registered as protected
# special tokens rather than being broken apart by the subword algorithm.