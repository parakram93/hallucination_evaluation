from datasets import load_dataset

dataset = load_dataset(
    "abisee/cnn_dailymail",
    "3.0.0"
)

with open("dataset/corpus.txt", "w",
          encoding="utf-8") as f:

    for sample in dataset["train"].select(range(50000)):
        f.write(sample["article"] + "\n")