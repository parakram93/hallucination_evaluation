

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from peft import PeftModel
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



DEBERTA_PATH   = "deberta_nli/checkpoint-1000"
CORRECTOR_PATH = "hallucination_detect/bart_lora_corrector/checkpoint-1126"


BASE_MODEL_NAME = "t5-small"   # confirmed



print("Loading DeBERTa NLI model...")
nli_tokenizer = AutoTokenizer.from_pretrained(DEBERTA_PATH)
nli_model     = AutoModelForSequenceClassification.from_pretrained(DEBERTA_PATH)
nli_model     = nli_model.to(device)
nli_model.eval()
print("NLI label mapping:", nli_model.config.id2label)

print("\nLoading corrector model...")
corrector_tokenizer = AutoTokenizer.from_pretrained(CORRECTOR_PATH)
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)
corrector_model = PeftModel.from_pretrained(base_model, CORRECTOR_PATH)
corrector_model = corrector_model.to(device)
corrector_model.eval()
print(" Both models loaded\n")




def classify_nli(premise: str, hypothesis: str) -> str:
    inputs = nli_tokenizer(
        premise, hypothesis,
        return_tensors="pt", truncation=True,
        max_length=256, padding=True
    ).to(device)

    with torch.no_grad():
        outputs = nli_model(**inputs)

    pred_id = outputs.logits.argmax(dim=1).item()
    return nli_model.config.id2label[pred_id]


def evaluate_detection(true_labels: list, pred_labels: list) -> dict:
    labels = list(set(true_labels + pred_labels))
    results = {}
    for label in labels:
        tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p == label)
        fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != label and p == label)
        fn = sum(1 for t, p in zip(true_labels, pred_labels) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        results[label] = {"precision": round(precision, 4),
                          "recall": round(recall, 4),
                          "f1": round(f1, 4)}
    return results


def correct_hallucination(context: str, hallucinated: str) -> str:
    input_text = f"context: {context} [SEP] summary: {hallucinated}"
    inputs = corrector_tokenizer(
        input_text, return_tensors="pt",
        max_length=256, truncation=True
    ).to(device)

    with torch.no_grad():
        output_ids = corrector_model.generate(
            **inputs, max_length=64, num_beams=4,
            early_stopping=True, no_repeat_ngram_size=3,
        )
    return corrector_tokenizer.decode(output_ids[0], skip_special_tokens=True)


def evaluate_correction(generated: list, gold: list) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = [scorer.score(g, gen) for gen, g in zip(generated, gold)]

    avg_rouge1 = sum(s["rouge1"].fmeasure for s in rouge_scores) / len(rouge_scores)
    avg_rouge2 = sum(s["rouge2"].fmeasure for s in rouge_scores) / len(rouge_scores)
    avg_rougeL = sum(s["rougeL"].fmeasure for s in rouge_scores) / len(rouge_scores)

    P, R, F1 = bert_score_fn(generated, gold, lang="en",
                              rescale_with_baseline=True, verbose=False)
    avg_bertscore = F1.mean().item()

    return {"rouge1": round(avg_rouge1, 4), "rouge2": round(avg_rouge2, 4),
            "rougeL": round(avg_rougeL, 4), "bertscore": round(avg_bertscore, 4)}




print("Loading evaluation dataset...")
raw_dataset = load_dataset("pminervini/HaluEval", "summarization")["data"]
raw_split   = raw_dataset.train_test_split(test_size=0.1, seed=42)
raw_val     = raw_split["test"]

N_SAMPLES = 30
sample = raw_val.select(range(min(N_SAMPLES, len(raw_val))))

test_contexts                = list(sample["document"])
test_hallucinated_summaries  = list(sample["hallucinated_summary"])
test_gold_corrections        = list(sample["right_summary"])
print(f"Evaluation set size: {len(test_contexts)}\n")



print("🔍 Running detection evaluation...")
true_detection_labels = []
pred_detection_labels = []

for ctx, hall in zip(test_contexts, test_hallucinated_summaries):
    true_detection_labels.append("CONTRADICTION")  # adjust to match your actual label name
    pred_detection_labels.append(classify_nli(ctx, hall))

detection_results = evaluate_detection(true_detection_labels, pred_detection_labels)
print("\n=== Detection Results ===")
for label, scores in detection_results.items():
    print(f"{label:15} P={scores['precision']:.3f}  R={scores['recall']:.3f}  F1={scores['f1']:.3f}")




print("\n  Running correction evaluation...")
generated_corrections = [
    correct_hallucination(ctx, hall)
    for ctx, hall in zip(test_contexts, test_hallucinated_summaries)
]

correction_results = evaluate_correction(generated_corrections, test_gold_corrections)
print("\n=== Correction Results ===")
for metric, value in correction_results.items():
    print(f"{metric:12}: {value}")




print("\n Full pipeline examples:\n")
for i in range(min(5, len(test_contexts))):
    ctx, hall, gold = test_contexts[i], test_hallucinated_summaries[i], test_gold_corrections[i]
    label = classify_nli(ctx, hall)
    corrected = correct_hallucination(ctx, hall)

    print(f"Example {i+1}")
    print(f"  Hallucinated: {hall}")
    print(f"  NLI Label   : {label}")
    print(f"  Corrected   : {corrected}")
    print(f"  Gold        : {gold}\n")