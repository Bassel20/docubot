import os

from embeddings import embed_chunks
from pdfTextExtract import chunk_ai_content, clean_ai_text, extract_ai_pdf
from rag import answer

PDF_FOLDER = "Dataset_pdf"


def ingest(pdf_folder=PDF_FOLDER):
    all_chunks = []
    pdfs = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    if not pdfs:
        print(f"No PDFs found in '{pdf_folder}'")
        return

    for pdf_file in pdfs:
        path = os.path.join(pdf_folder, pdf_file)
        text = extract_ai_pdf(path)
        cleaned = clean_ai_text(text)
        chunks = chunk_ai_content(cleaned, source=pdf_file)
        all_chunks.extend(chunks)
        print(f"  {pdf_file}: {len(chunks)} chunks")

    embed_chunks(all_chunks)
    print(f"\nIngested {len(all_chunks)} chunks from {len(pdfs)} PDF(s)\n")


def ask(question):
    print(f"Q: {question}")
    result = answer(question)
    print(f"A: {result['answer']}")
    print(f"Sources: {', '.join(result['sources'])}\n")


if __name__ == "__main__":
    print("=== Ingesting documents ===")
    ingest()

    print("=== Asking questions ===")
    ask("What is the attention mechanism?")
    ask("What are the main advantages of transformer models?")
    ask("How does BERT differ from GPT?")
