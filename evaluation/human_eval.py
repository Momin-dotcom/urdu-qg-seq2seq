import csv

with open("results/samples.tsv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    rows = list(reader)

with open("results/human_eval_template.csv", "w", encoding="utf-8-sig", newline="") as f:
    fieldnames = [
        "source", "reference", "greedy",
        "Fluency_Member1", "Relevance_Member1", "Answerability_Member1",
        "Fluency_Member2", "Relevance_Member2", "Answerability_Member2"
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=",")
    writer.writeheader()
    for row in rows:
        writer.writerow({
            "source": row["source"],
            "reference": row["reference"],
            "greedy": row["greedy"],
            "Fluency_Member1": "",
            "Relevance_Member1": "",
            "Answerability_Member1": "",
            "Fluency_Member2": "",
            "Relevance_Member2": "",
            "Answerability_Member2": "",
        })

print("Saved results/human_eval_template.csv")