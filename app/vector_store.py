# app/vector_store.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Initialize the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def create_vector_store(chunks):
    """
    Create a FAISS vector store from text chunks.
    Returns:
        index (faiss.IndexFlatL2): FAISS index
        embeddings (np.ndarray): embeddings of all chunks
    """
    # Encode chunks into embeddings
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    
    # Ensure dtype is float32 for FAISS
    embeddings = embeddings.astype("float32")
    
    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    
    return index, embeddings


def save_vector_store(index, embeddings, path="vector_store.pkl"):
    """
    Save FAISS index and embeddings to disk using pickle
    """
    with open(path, "wb") as f:
        pickle.dump((index, embeddings), f)


def load_vector_store(path="vector_store.pkl"):
    """
    Load FAISS index and embeddings from disk
    """
    with open(path, "rb") as f:
        return pickle.load(f)
