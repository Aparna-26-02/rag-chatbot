# app/qa_chain.py
from sentence_transformers import SentenceTransformer
import requests
from retriever import retrieve

# ========================================
# Initialize embedding model
# ========================================
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ========================================
# Gemini API configuration (Google)
# ========================================
GEMINI_API_KEY = ""
MODEL_NAME = "gemini-2.0-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

# ========================================
# Gemini response function
# ========================================
def get_gemini_response(prompt: str) -> str:
    """
    Send prompt to Gemini API and return generated text.
    """
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 500}
    }

    headers = {"Content-Type": "application/json"}
    url_with_key = f"{GEMINI_API_URL}?key={GEMINI_API_KEY}"

    try:
        response = requests.post(url_with_key, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()

        generated_text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", None)
        )

        return generated_text.strip() if generated_text else "⚠️ No answer generated."
    except requests.exceptions.Timeout:
        return "⚠️ Gemini API request timed out."
    except requests.exceptions.RequestException as e:
        return f"⚠️ Gemini API network error: {str(e)}"
    except Exception as e:
        return f"⚠️ Unexpected error: {str(e)}"

# ========================================
# Main function: Retrieval + Answer generation
# ========================================
def get_answer(query, index, chunks, top_k=3, threshold=1.0):
    """
    Answer questions using context from PDF.
    Falls back to Gemini for general queries.
    """
    query_emb = embedding_model.encode([query])[0].astype("float32")
    relevant_chunks = retrieve(query_emb, index, chunks, top_k=top_k, threshold=threshold)

    if relevant_chunks:
        context = "\n\n".join(relevant_chunks)
        prompt = f"""
        You are a helpful data analysis assistant.
        Answer the user's question using only the information provided in the report below.

        --- REPORT CONTEXT ---
        {context}
        -----------------------

        Question: {query}

        If the answer is not present in the context, reply:
        "The report does not mention this specifically."

        Keep your answer clear, factual, and concise.
        """
        return get_gemini_response(prompt)
    else:
        # No relevant context → fallback to general Gemini knowledge
        return get_gemini_response(query)
