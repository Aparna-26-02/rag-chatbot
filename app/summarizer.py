from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "pszemraj/led-large-book-summary"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def summarize(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    summary_ids = model.generate(inputs["input_ids"], max_length=300, min_length=50, length_penalty=2.0, num_beams=4)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
