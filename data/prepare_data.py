from datasets import load_dataset

ds = load_dataset("uqa/UQA")

print(ds)

ex = ds["train"][0]
print(ex.keys())
print(ex["question"])
print(ex["answer"])
print(ex["answer_start"])

n_total = len(ds["train"])
n_ans = sum(not a["is_impossible"] for a in ds["train"])
print(f"train rows: {n_total}, answerable: {n_ans}")