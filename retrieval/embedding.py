from sentence_transformers import SentenceTransformer
from pathlib import Path
from tokenizers import Tokenizer

import json

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

with open(
    "ingestion/chunks.json",
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)
    
text = [chunk['text'] for chunk in chunks]

embeddings = model.encode(
    text,
    convert_to_numpy=True,
    show_progress_bar=True   
)

import numpy as np

np.save(
    "retrieval/embeddings.npy",
    embeddings
)

print("Embeddings saved.")