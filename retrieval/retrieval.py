import json
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

index = faiss.read_index(
    "retrieval/faiss.index"
)

with open(
    "ingestion/chunks.json",
    "r",
    encoding="utf-8"
) as f:

    chunks = json.load(f)
    
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

def retrieve_evidence(
                     query: str,
                         k: int = 3
                                       ):

    # Convert query into embedding

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype(np.float32)

    # Search FAISS

    distances, indices = index.search(
        query_embedding,
        k
    )

    evidence = []

    # Get retrieved chunks

    for idx in indices[0]:

        chunk_info = {
            "chunk_id": chunks[idx]["chunk_id"],
            "article_id": chunks[idx]["article_id"],
            "text": chunks[idx]["text"]
        }

        evidence.append(
            chunk_info
        )

    return evidence

query = (
    "The president visited Nepal on Tuesday."
)

results = retrieve_evidence(
    query,
    k=3
)
with open(
    "retrieval/evidence.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        ensure_ascii=False,
        indent=4
    )
print("\nRetrieved Evidence:\n")

for i, chunk in enumerate(results, start=1):

    print(f"Evidence {i}")
    print(f"Chunk ID : {chunk['chunk_id']}")
    print(f"Article  : {chunk['article_id']}")
    print(chunk["text"])
    print("-" * 80)