from datasets import load_dataset
import csv
import matplotlib.pyplot as plt

ds = load_dataset("uqa/UQA")
print(ds)

ex = ds["train"][0]
print(ex.keys())
print(ex["question"])

def to_answers_dict(example):
    if example["is_impossible"]:
        return {"text": [], "answer_start": []}
    return {"text": [example["answer"]], "answer_start": [example["answer_start"]]}

print(to_answers_dict(ex))

n_total = len(ds["train"])
n_ans = sum(not imp for imp in ds["train"]["is_impossible"])
print(f"train rows: {n_total}, answerable: {n_ans}")

ANS_OPEN, ANS_CLOSE = "<ans>", "</ans>"
SENT_DELIMS = "\u06D4\u061F!"

def split_sentences(text):
    start = 0
    for i, ch in enumerate(text):
        if ch in SENT_DELIMS:
            yield start, i + 1, text[start:i + 1]
            start = i + 1
    if start < len(text):
        yield start, len(text), text[start:]

def make_pair(example, max_src=60, max_tgt=25):
    answers = to_answers_dict(example)
    if len(answers["text"]) == 0:
        return None
    a_start = answers["answer_start"][0]
    a_text = answers["text"][0]
    context = example["context"]

    for s, e, sent in split_sentences(context):
        if s <= a_start < e:
            rel = a_start - s
            if sent[rel:rel + len(a_text)] != a_text:
                return None
            src = (sent[:rel] + " " + ANS_OPEN + " " + a_text + " "
                   + ANS_CLOSE + " " + sent[rel + len(a_text):]).strip()
            src = " ".join(src.split())
            tgt = " ".join(example["question"].split())
            if len(src.split()) > max_src or len(tgt.split()) > max_tgt:
                return None
            return src, tgt
    return None

def build_split(split, out_path):
    pairs = [p for p in map(make_pair, split) if p is not None]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerows(pairs)
    print(f"{out_path}: {len(pairs)} pairs")
    return pairs

train_pairs = build_split(ds["train"], "train.tsv")
valid_pairs = build_split(ds["validation"], "valid.tsv")

print(train_pairs[0])

src_lens = [len(src.split()) for src, tgt in train_pairs]
tgt_lens = [len(tgt.split()) for src, tgt in train_pairs]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(src_lens, bins=30)
axes[0].set_title("Source sentence length (words)")
axes[0].set_xlabel("Words")
axes[0].set_ylabel("Count")

axes[1].hist(tgt_lens, bins=30)
axes[1].set_title("Target question length (words)")
axes[1].set_xlabel("Words")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("length_histograms.png")
plt.close()

print(f"Mean source length: {sum(src_lens)/len(src_lens):.1f} words")
print(f"Mean target length: {sum(tgt_lens)/len(tgt_lens):.1f} words")