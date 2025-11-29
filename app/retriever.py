import numpy as np
import faiss

def create_faiss_index(embeddings):
    """
    Create a FAISS index from chunk embeddings.
    """
    embeddings = np.array(embeddings).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # L2 distance metric
    index.add(embeddings)
    return index

def retrieve(query_emb, index, chunks, top_k=3, threshold=1.0):
    """
    Retrieve relevant PDF chunks for the query embedding.
    
    Args:
        query_emb: 1D numpy array embedding of query
        index: FAISS index of chunk embeddings
        chunks: list of text chunks
        top_k: number of top results to return
        threshold: distance threshold (lower = more similar)
    Returns:
        List of relevant chunks
    """
    query_emb = np.array(query_emb, dtype="float32").reshape(1, -1)

    distances, indices = index.search(query_emb, top_k)
    
    relevant_chunks = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(chunks) and (dist <= threshold or threshold >= 1.0):
            relevant_chunks.append(chunks[idx])
    
    return relevant_chunks
