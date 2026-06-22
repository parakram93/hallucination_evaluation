from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained(
    "hallucination_detect/trained"
)

model = AutoModelForSequenceClassification.from_pretrained(
    "hallucination_detect/trained"
)

import torch

label_map = {
    0: "entailment",
    1: "neutral",
    2: "contradiction"
}

def detect_hallucination(context, response):

    inputs = tokenizer(
        context,
        response,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    pred = torch.argmax(logits, dim=-1).item()

    return label_map[pred]

context = "Nepal's capital city is Kathmandu."
response = "Nepal's capital city is Pokhara."

print(detect_hallucination(context, response))