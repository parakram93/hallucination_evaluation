import numpy as np
import faiss #stores vectors + finds nearest neighbours

embeddings = np.load(
    "retrieval/embeddings.npy"
)

dimension = embeddings.shape[1] #faiss must know how many numbers are inside each vector

index = faiss.IndexFlatL2( #this creates empty vector database# l2 because faiss uses euclidean distance to calculate similarity
    dimension
)

index.add( #adds 5000 vectors to invex
    embeddings.astype(
        np.float32
    )
)

faiss.write_index(
    index,
    "retrieval/faiss.index"
)

print(
    f"Indexed {index.ntotal} chunks."
)