# app/gradio_ui.py
import gradio as gr
from datetime import datetime
from loader import extract_text_from_pdf
from vector_store import create_vector_store
from qa_chain import get_answer

# ----------------------------
# Load PDF and split into chunks
# ----------------------------
pdf_path = r"C:\Users\91854\Documents\private\rag-chatbot\data\Report 1.pdf"
text = extract_text_from_pdf(pdf_path)
chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]  # ~1000-char chunks

# Create vector store
index, embeddings = create_vector_store(chunks)

# ----------------------------
# Chatbot response function
# ----------------------------
def chatbot_response(user_message, chat_history):
    timestamp = datetime.now().strftime("%H:%M")
    chat_history = chat_history + [{"role": "user", "content": f"You ({timestamp}): {user_message}"}]
    chat_history = chat_history + [{"role": "assistant", "content": "Bot is typing..."}]

    answer = get_answer(user_message, index, chunks)

    chat_history[-1] = {"role": "assistant", "content": f"Bot ({timestamp}): {answer}"}
    return chat_history, ""

# ----------------------------
# PDF summary panel
# ----------------------------
pdf_summary = "\n".join(chunks[:5])  # First 5 chunks as preview

# ----------------------------
# Gradio UI
# ----------------------------
with gr.Blocks(title="RAG + Gemini Pro Chatbot") as demo:
    gr.HTML("""
    <style>
    #chatbox .overflow-y-auto { border-radius: 15px; border: 2px solid #4B0082; padding: 10px; background-color:#F7F7F7; }
    #summary_box { border-radius: 10px; border: 2px solid #4B0082; padding: 10px; background-color:#FAFAFA; color:#333; }
    #submit_btn { background-color: #4B0082; color: white; font-weight: bold; border-radius: 10px; transition: 0.3s; }
    #submit_btn:hover { background-color: #6A0DAD; }
    body { background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%); font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    </style>
    """)

    gr.Markdown("<h1 style='color:#4B0082; font-family:Arial; text-align:center;'>📚 RAG + Gemini Pro Chatbot</h1>")
    gr.Markdown("<p style='text-align:center; color:#555;'>Ask questions about your PDF or general topics. Falls back to Gemini Pro if PDF has no relevant info.</p>")

    with gr.Row():
        with gr.Column(scale=2):
            chatbox = gr.Chatbot(elem_id="chatbox", height=500, type="messages")
            with gr.Row():
                user_input = gr.Textbox(placeholder="Type your question here...", label="", lines=2)
                submit_btn = gr.Button("Send", variant="primary", elem_id="submit_btn")

        with gr.Column(scale=1):
            gr.Markdown("<h3 style='color:#4B0082;'>PDF Summary / Preview</h3>")
            summary_box = gr.Textbox(value=pdf_summary, interactive=False, lines=25, elem_id="summary_box")

    submit_btn.click(chatbot_response, inputs=[user_input, chatbox], outputs=[chatbox, user_input])
    user_input.submit(chatbot_response, inputs=[user_input, chatbox], outputs=[chatbox, user_input])

# ----------------------------
# Launch with explicit local server
# ----------------------------
if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)
