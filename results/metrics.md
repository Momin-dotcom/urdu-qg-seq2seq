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
