import os
import sys
import json
import re

import pdfplumber
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE = 256    # tokens per chunk
CHUNK_OVERLAP = 32  # tokens of overlap between consecutive chunks


def extract_ai_pdf(pdf_path):
    """Extract text from a PDF, preserving layout for code/formulas/bullet points."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(layout=True)
            if page_text:
                text += page_text + "\n\n"
    return text


def clean_ai_text(text):
    """Drop the trailing references/bibliography section and normalize whitespace."""
    text = re.split(r'\n\s*References\s*\n', text, maxsplit=1)[0]
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def chunk_ai_content(text, source, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping token-based chunks for embedding."""
    tokens = TOKENIZER.encode(text)
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(tokens), step):
        token_chunk = tokens[start:start + chunk_size]
        if not token_chunk:
            break
        chunk_text = TOKENIZER.decode(token_chunk, skip_special_tokens=True)
        chunks.append({"text": chunk_text, "source": source})
        if start + chunk_size >= len(tokens):
            break
    return chunks


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python pdfTextExtract.py <pdf_folder>")

    pdf_folder = sys.argv[1]
    all_chunks = []

    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            text = extract_ai_pdf(os.path.join(pdf_folder, pdf_file))
            cleaned = clean_ai_text(text)
            all_chunks.extend(chunk_ai_content(cleaned, source=pdf_file))

    with open("ai_chunks.json", "w") as f:
        json.dump(all_chunks, f)

    print(f"Wrote {len(all_chunks)} chunks to ai_chunks.json")
