from loader import extract_text_from_pdf, chunk_text

file_path = r"C:\Users\91854\Documents\private\rag-chatbot\app\report1.pdf"

text = extract_text_from_pdf(file_path)
print("Extracted text (first 500 chars):")
print(text[:500])

chunks = chunk_text(text, chunk_size=100)
print("First 2 chunks:")
print(chunks[:2])
