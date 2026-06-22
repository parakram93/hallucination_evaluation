print("output1")
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
print("output2")
from datasets import load_dataset
print("output3")
import torch
print("Loading dataset...")
dataset = load_dataset("nyu-mll/multi_nli")

tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/deberta-v3-base"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=3
)
dataset["train"] = dataset["train"].select(range(5000))
dataset["validation_matched"] = dataset["validation_matched"].select(range(1000))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)

def tokenize_function(examples):

    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
        padding="max_length",
        max_length=256
    )
    
tokenized= dataset.map(
    tokenize_function,
    batched=True
)
tokenized = tokenized.remove_columns([
    "premise",
    "hypothesis",
    "promptID",
    "pairID",
    "genre",
    "premise_binary_parse",
    "hypothesis_binary_parse",
    "premise_parse",
    "hypothesis_parse"
])
tokenized = tokenized.rename_column("label", "labels")

tokenized.set_format("torch")

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./deberta_nli",

    learning_rate=2e-5,

    per_device_train_batch_size=8,

    per_device_eval_batch_size=8,

    num_train_epochs=2,

    weight_decay=0.01,

    fp16=False,
    bf16 = True,
    logging_steps = 50,
    save_steps = 1000,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation_matched"]
)

trainer.train()
tokenizer.save_pretrained(
    "hallucination_detect/trained"
)

trainer.save_model("hallucination_detect/trained")