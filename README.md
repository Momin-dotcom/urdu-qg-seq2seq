# Urdu Question Generation (Sequence-to-Sequence, Built from Scratch)

**Generative AI — Assignment 01 | Fall 2026**

A from-scratch RNN encoder–decoder (with attention) that reads an Urdu sentence with the answer marked inside it, and generates the question that the marked answer responds to — the same task a teacher performs when turning a paragraph into practice questions.

**Example**
- Input: `دریائے سندھ کی لمبائی تقریباً <ans> 3,180 کلومیٹر </ans> ہے۔`
- Output: `دریائے سندھ کی لمبائی کتنی ہے؟`

No pretrained models, embeddings, or Transformer architectures were used anywhere in this project — the tokenizer, the encoder, the decoder, and the attention mechanism were all built and trained from scratch, as required by the assignment.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Pipeline / How to Reproduce](#pipeline--how-to-reproduce)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Front End](#front-end)
- [Known Limitations](#known-limitations)
- [Team](#team)

---

## Overview

This project implements a sequence-to-sequence question generator for Urdu, following the setting of Du et al. (2017): rather than reading an entire paragraph, the model only sees the single sentence that contains the answer, since a from-scratch RNN trained on ~90k pairs cannot reliably learn to read a full 300-token paragraph.

The project is broken into six tasks, matching the assignment manual:

| Task | Description | Status |
|---|---|---|
| 1 | Data preparation (sentence extraction, `<ans>` tagging, length filtering) | ✅ Complete |
| 2 | Custom SentencePiece tokenizer (vocab size 8k) | ✅ Complete |
| 3 | Encoder–decoder model with attention, trained from scratch | ✅ Complete |
| 4 | Evaluation (BLEU-4, ROUGE-L, perplexity, human evaluation) | ✅ Complete |
| 5 | Web front end (greedy + beam decoding demo) | ✅ Complete |
| 6 | Blog post, LinkedIn post, GitHub repo | ✅ Complete |

---

## Dataset

**UQA: Corpus for Urdu Question Answering** (Arif, Farid, Athar and Raza, LREC-COLING 2024) — a translation of SQuAD 2.0 into Urdu that preserves answer-span character offsets.

- **Source dataset:** [`uqa/UQA`](https://huggingface.co/datasets/uqa/UQA) — ~142k context–question–answer rows
- **Out-of-domain test set:** [`uqa/Wiki-UQA`](https://huggingface.co/datasets/uqa/Wiki-UQA)

### Preprocessing (Task 1)

1. Loaded `uqa/UQA`, kept only rows with a non-empty answer.
2. Found the sentence containing the answer using character offsets (sentence delimiters: Urdu full stop `۔`, Urdu question mark `؟`, `!`).
3. Wrapped the answer in `<ans> ... </ans>` inside that sentence — this became the **source**; the original `question` field became the **target**.
4. Dropped pairs where source > 60 or target > 25 whitespace tokens.
5. Wrote `train.tsv` and `valid.tsv`.

### Dataset statistics

| | Train | Validation | Wiki-UQA |
|---|---|---|---|
| Rows in raw dataset | 124,745 | 16,824 | 210 |
| Answerable rows | 83,018 | 11,169 | 210 |
| Pairs after length filter | 75,067 | 10,018 | 177 |
| Mean source length (words) | 32.6 | 33.2 | — |
| Mean target length (words) | 11.9 | 12.3 | 11.9 |

---

## Repository Structure

```
urdu-qg-seq2seq/
├── app/
│   └── app.py                  # Streamlit/Gradio front end
├── data/
│   ├── prepare_data.py         # Task 1: sentence extraction + tagging
│   ├── train.tsv
│   └── valid.tsv
├── tokenizer/
│   ├── train_tokenizer.py      # Task 2: SentencePiece training
│   ├── sp_corpus.txt
│   ├── ur_sp.model
│   └── ur_sp.vocab
├── model/
│   ├── encoder.py               # Bidirectional 2-layer LSTM encoder
│   ├── attention.py             # Bahdanau/Luong attention
│   ├── decoder.py                # Unidirectional 2-layer LSTM decoder
│   ├── decode.py                 # Greedy + beam search decoding
│   └── train.py                  # Training loop
├── evaluation/
│   ├── evaluate.py                # BLEU-4, ROUGE-L, perplexity, unk-rate
│   └── human_eval.py              # Human evaluation template generator
├── results/
│   ├── figures/
│   │   ├── length_histograms.png
│   │   ├── loss_curve.png
│   │   └── attention_heatmap.png
│   ├── samples.tsv                # 50 UQA validation samples (source/ref/greedy/beam)
│   ├── wiki_samples.tsv           # Wiki-UQA out-of-domain samples
│   ├── human_eval_results.csv     # Both members' Fluency/Relevance/Answerability ratings
│   └── metrics.md                 # All results tables + discussion
├── best_checkpoint.pt             # Best model checkpoint (by validation loss)
└── README.md
```

---

## Setup

```bash
pip install torch sentencepiece datasets sacrebleu rouge-score scikit-learn streamlit
```

Training and evaluation were run on a **T4 GPU (Google Colab)**; data preparation and tokenizer training can run on CPU.

---

## Pipeline / How to Reproduce

**1. Prepare the data**
```bash
python data/prepare_data.py
```
Produces `data/train.tsv` and `data/valid.tsv`.

**2. Train the tokenizer**
```bash
python tokenizer/train_tokenizer.py
```
Produces `ur_sp.model` / `ur_sp.vocab` (SentencePiece, unigram, vocab size 8,000, with `<ans>`/`</ans>` as user-defined symbols).

**3. Train the model**
```bash
python model/train.py
```
Trains for 10 epochs with teacher forcing, Adam optimizer, and padding-aware cross-entropy loss. Saves the checkpoint with the best validation loss to `best_checkpoint.pt`.

**4. Evaluate**
```bash
python evaluation/evaluate.py
```
Computes validation loss/perplexity, runs greedy and beam-search decoding on 50 validation examples, computes BLEU-4/ROUGE-L/unk-rate, and writes `results/samples.tsv`.

**5. Run the front end**
```bash
streamlit run app/app.py
```

---

## Model Architecture

| Component | Configuration |
|---|---|
| Encoder | 2-layer **bidirectional** LSTM |
| Decoder | 2-layer **unidirectional** LSTM with attention |
| Embedding size | 256 |
| Hidden size | 512 |
| Vocabulary size | 8,000 |
| Dropout | 0.3 |
| Trainable parameters | 32,407,872 |
| Optimizer | Adam, lr = 0.001 |
| Batch size | 64 |
| Epochs | 10 |
| Decoding | Greedy and beam search (beam width 3) |

Since the encoder is bidirectional (producing forward and backward hidden states per layer) and the decoder is unidirectional, the encoder's forward and backward final hidden/cell states are combined (summed and passed through a linear projection) before being used to initialize the decoder — otherwise the shapes would not be compatible.

---

## Results

### Automatic metrics

| Split | Decoding | BLEU-4 | ROUGE-L | Perplexity | `<unk>` % |
|---|---|---|---|---|---|
| UQA valid | greedy | 1.63 | 0.150 | 98.36 | 0.0 |
| UQA valid | beam (k=3) | 1.83 | 0.137 | 98.36 | 0.0 |
| Wiki-UQA | greedy | 3.71 | 0.194 | — | 0.0 |
| Wiki-UQA | beam (k=3) | 2.18 | 0.186 | — | 0.0 |

### Human evaluation (50 samples, greedy outputs)

| | Fluency | Relevance | Answerability |
|---|---|---|---|
| Member 1 (% yes) | 10.0% | 6.0% | 2.0% |
| Member 2 (% yes) | 2.0% | 16.0% | 14.0% |
| Cohen's κ | -0.034 | 0.104 | -0.036 |

### Key findings

- The model learned the basic **surface pattern** of Urdu question formation (کیا/کب/کہاں/کون), but frequently selects the **wrong question word** for the marked answer's type — this is the dominant failure mode and is reflected in the low human-evaluation scores.
- Many generated outputs show **token repetition** (e.g. "کے کے کے کے"), which fails Fluency outright; this cascades into automatic failures on Relevance and Answerability, since an incoherent sentence cannot be meaningfully judged on those criteria.
- Beam search did **not** uniformly outperform greedy decoding: it helped slightly on UQA validation (BLEU 1.83 vs 1.63) but hurt on the out-of-domain Wiki-UQA set (BLEU 2.18 vs 3.71) — consistent with beam search converging on "safe," generic sequences that can overlap less with the reference when the underlying model is undertrained.
- The **training/validation loss curve** shows clear overfitting: training loss fell steadily from 5.48 to 2.75 over 10 epochs, while validation loss reached its minimum at epoch 3 (4.56) and then increased — this is why the saved checkpoint corresponds to epoch 3, not the final epoch.
- BLEU-4 scores are below the range reported by Du et al. (2017) on English SQuAD (6–13), and below the manual's expected range for UQA — indicating the model would benefit from more training epochs, more data, or a larger dataset subset than what time constraints allowed.
- Very low (and occasionally negative) Cohen's κ values indicate weak inter-rater agreement — itself a signal that when generated outputs are this inconsistent in quality, "good/bad" judgments become highly subjective between raters.

See [`results/metrics.md`](results/metrics.md) for full tables and discussion, and [`results/samples.tsv`](results/samples.tsv) / [`results/wiki_samples.tsv`](results/wiki_samples.tsv) for qualitative examples.

---

## Front End

A minimal web UI (built with Streamlit) lets a user paste an Urdu sentence, mark the answer span, and see both the greedy and beam-search generated questions from the trained model. See `results/figures/frontend.png` for a screenshot.

---

## Known Limitations

- Trained for only 10 epochs on a subset workflow constrained by GPU quota and time; BLEU-4/ROUGE-L are correspondingly modest.
- The model frequently generates questions with the wrong interrogative word (e.g. asking "when" for a location-type answer).
- Evaluation was run on the first 50 validation examples for decoding-based metrics (BLEU/ROUGE/human eval) due to the computational cost of token-by-token decoding with attention; perplexity was computed on the full validation set.

---

## Team

This is a group assignment completed by two members. Both members contributed to model implementation, training, evaluation, and independently rated the human-evaluation samples (see `results/human_eval_results.csv` for both members' ratings and `results/metrics.md` for Cohen's κ agreement scores).

- Medium blog post: *[link here]*
- LinkedIn post: *[link here]*
