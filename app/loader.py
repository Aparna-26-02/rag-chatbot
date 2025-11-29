# app/loader.py
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import re
import os

# Point to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def extract_text_from_pdf(file_path):
    """
    Extract text from PDF:
    - Keeps headings and bullet points
    - Uses OCR for scanned pages if normal extraction fails
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    structured_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            print(f"Extracting page {page_num + 1}/{len(pdf.pages)}...")

            page_text = page.extract_text()
            if page_text:
                lines = page_text.split("\n")
                for line in lines:
                    if line.strip().startswith(("-", "*")) or re.match(r"^\d+[\.\)]", line.strip()):
                        structured_text += line.strip() + "\n"
                    else:
                        structured_text += line.strip() + " "
                structured_text += "\n\n"
            else:
                # OCR fallback for scanned PDF pages
                images = convert_from_path(file_path, dpi=300, first_page=page_num+1, last_page=page_num+1)
                for img in images:
                    ocr_text = pytesseract.image_to_string(img)
                    structured_text += ocr_text + "\n\n"

    # Normalize whitespace
    structured_text = re.sub(r'\s+', ' ', structured_text)
    return structured_text.strip()


def chunk_text_by_section(text, max_length=1000):
    """
    Split text into sections/chunks by headings, bullets, or size
    """
    pattern = r'(?=\n\d+[\.\)]|\n[-*])'
    chunks = re.split(pattern, text)
    
    final_chunks = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if len(chunk) <= max_length:
            final_chunks.append(chunk)
        else:
            for i in range(0, len(chunk), max_length):
                final_chunks.append(chunk[i:i+max_length])
    return final_chunks
