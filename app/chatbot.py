from loader import extract_text_from_pdf
from vector_store import create_vector_store
from qa_chain import get_answer

# Load PDF and split into chunks
pdf_path = "../data/documents/report1.pdf"
text = extract_text_from_pdf(pdf_path)
chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]  # split into ~1000-char chunks

# Create vector store
index, embeddings = create_vector_store(chunks)

# Chat loop
print("Chatbot ready! Ask your question (type 'exit' to quit):")
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    answer = get_answer(query, index, chunks)
    print("Bot:", answer)
