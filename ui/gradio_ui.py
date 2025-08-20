import sys
import os

# Add the parent directory (rag-chatbot) to sys.path so `app/` can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gradio as gr
from app.loader import extract_text_from_pdf, chunk_text
from app.retriever import create_embeddings, retrieve_top_chunk
from app.qa import answer_question
from app.summarizer import summarize

# Global variables
chunks = []
chunk_embeddings = []

def process_pdf(file):
    global chunks, chunk_embeddings
    file_path = file.name
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)
    chunk_embeddings = create_embeddings(chunks)
    return "✅ PDF processed successfully!"

def ask_question(question):
    if not chunks:
        return "Please upload and process a PDF first."
    context = retrieve_top_chunk(question, chunks, chunk_embeddings)
    return answer_question(question, context)

def get_summary():
    if not chunks:
        return "Please upload and process a PDF first."
    full_text = " ".join(chunks[:10])  # summarize first 10 chunks
    return summarize(full_text)

# Gradio interfaces
upload_interface = gr.Interface(
    fn=process_pdf,
    inputs=gr.File(type="file"),
    outputs="text",
    title="📄 Upload PDF",
    description="Upload a textbook or document to process it."
)

qa_interface = gr.Interface(
    fn=ask_question,
    inputs=gr.Textbox(placeholder="Ask a question about the uploaded PDF...", lines=2),
    outputs="text",
    title="❓ Ask a Question",
    description="Ask questions and get context-aware answers."
)

summary_interface = gr.Interface(
    fn=get_summary,
    inputs=[],
    outputs="text",
    title="📝 Get Summary",
    description="Get a short summary of the uploaded PDF."
)

# Combine all interfaces
gr.TabbedInterface(
    [upload_interface, qa_interface, summary_interface],
    ["Upload", "Q&A", "Summary"]
).launch()
