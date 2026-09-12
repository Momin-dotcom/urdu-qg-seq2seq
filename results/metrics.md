# Results — Urdu Question Generation (Seq2Seq)
## Table 1 — Dataset Statistics

| | Train | Validation | Wiki-UQA |
|---|---|---|---|
| Rows in raw dataset | 124745 | 16824 | 210 |
| Answerable rows | 83018 | 11169 | 210 |
| Pairs after length filter | 75067 | 10018 | 177 |
| Mean source length (words) | 32.6 | 33.2 | — |
| Mean target length (words) | 11.9 | 12.3 | 11.9 |

## Table 2 — Model Configuration

| | Value |
|---|---|
| Encoder type | 2-layer bidirectional LSTM |
| Decoder type | 2-layer unidirectional LSTM with attention |
| Embedding size | 256 |
| Hidden size | 512 |
| Vocabulary size | 8000 |
| Trainable parameters | 32,407,872 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 64 |
| Epochs | 10 |
| GPU | T4 (Google Colab) |

## Table 3 — Automatic Metrics

| Split | Decoding | BLEU-4 | ROUGE-L | PPL | unk % |
|---|---|---|---|---|---|
| UQA valid | greedy | 1.63 | 0.150 | 98.36 | 0.0 |
| UQA valid | beam (k=3) | 1.83 | 0.137 | 98.36 | 0.0 |
| Wiki-UQA | greedy | 3.71 | 0.194 | — | 0.0 |
| Wiki-UQA | beam (k=3) | 2.18 | 0.186 | — | 0.0 |

## Table 4 — Human Evaluation (50 samples)

| | Fluency | Relevance | Answerability |
|---|---|---|---|
| Member 1 (% yes) | 10.0 | 6.0 | 2.0 |
| Member 2 (% yes) | 2.0 | 16.0 | 14.0 |
| Cohen's κ | -0.034 | 0.104 | -0.036 |

## Discussion Notes

- The model has learned the basic pattern of question formation (کیا/کب/کہاں/کون — what/when/where/who), but struggles to choose the correct question word for the given answer type — the very low % yes scores in human evaluation confirm this.
- Many outputs showed token repetition (e.g., "کے کے کے کے"), which immediately fails Fluency; this in turn causes Relevance and Answerability to fail automatically as well, since an incoherent sentence cannot be meaningfully judged on those criteria.
- Interesting finding: on UQA validation, beam search (BLEU 1.83) slightly outperformed greedy decoding (BLEU 1.63), but the opposite happened on Wiki-UQA — greedy (BLEU 3.71) outperformed beam (BLEU 2.18). This shows that beam search does not always improve output quality, particularly for undertrained models, where it can converge on a "safe" but generic sequence that overlaps less with the reference.
- BLEU-4 scores are lower than the manual's expected range (6–13), indicating that the model would benefit from additional training/tuning — 10 epochs appear insufficient for a dataset of this size.
- Cohen's κ values are very low (some negative), indicating poor agreement between the two human raters. This is itself a meaningful finding: when model outputs are this weak and inconsistent, judging them as "good/bad" becomes highly subjective and varies rater to rater.
- The training/validation loss curve shows clear overfitting: train_loss decreased steadily (5.48 → 2.75), but valid_loss began increasing after epoch 3 (4.56, the best checkpoint) — meaning the model started memorizing the training data rather than generalizing further. This is why the saved best_checkpoint corresponds to epoch 3, not epoch 9.