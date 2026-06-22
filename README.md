# Hallucination Evaluation

Hallucination Evaluation is an NLP project designed to detect and correct factual hallucinations in AI-generated summaries.

## Features

- 🔍 Hallucination detection using DeBERTa
- ✏️ Hallucination correction using T5
- 📄 Document-grounded fact verification
- 📊 Evaluation with Precision, Recall, F1, ROUGE, and BERTScore

## Architecture

- **DeBERTa (Fact Checker):** Classifies summary sentences as TRUE, FALSE, or UNSURE based on the source document.
- **T5 (Editor):** Rewrites hallucinated sentences to make them factually consistent with the source document.

## Purpose

This project was built to explore hallucination mitigation in NLP systems by separating the tasks of factual verification and text correction into two specialized models.
