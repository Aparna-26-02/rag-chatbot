from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

def create_embeddings(chunks):
    return model.encode(chunks, convert_to_tensor=True, normalize_embeddings=True)

def retrieve_top_chunk(query, chunks, chunk_embeddings, top_k=1):
    query_embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    hits = util.semantic_search(query_embedding, chunk_embeddings, top_k=top_k)
    top_index = hits[0][0]['corpus_id']
    return chunks[top_index]
