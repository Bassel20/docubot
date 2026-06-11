# DocuBot Local

A personal, offline document assistant. like ChatGPT for your own notes, PDFs, and
research papers. It runs entirely on your machine using a local LLM (Mistral via
Ollama) and a local vector database (ChromaDB), so your documents never leave
your computer.

## What it does

- Ingests your documents (PDFs, notes, articles) and stores them in a searchable
  vector index
- Finds the passages most relevant to your question
- Generates an answer using a local LLM, grounded in those passages (RAG)

## Why

- Search legal contracts, study notes, or internal docs without uploading
  anything to the cloud
- Useful for researchers, students, or businesses handling sensitive data
- Works fully offline (e.g., on a flight or in the field)

## Architecture

| Component | File | Role |
| --- | --- | --- |
| PDF ingestion | `pdfTextExtract.py` | Extracts text from PDFs, cleans it, and splits it into overlapping token-based chunks |
| Embeddings | `embeddings.py` *(planned)* | Embeds chunks with `all-MiniLM-L6-v2` and stores them in ChromaDB |
| RAG core | `rag.py` *(planned)* | Retrieves relevant chunks for a question and queries Ollama (Mistral) for an answer |
| API | `app.py` *(planned)* | Flask server exposing a `/query` endpoint |
| Examples | `example_usage.py` *(planned)* | Sample script for ingesting documents and asking questions |

## Setup

1. Install [Ollama](https://ollama.com) and pull the Mistral model:
   ```bash
   ollama pull mistral
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Extract and chunk PDFs from a folder (e.g., `Dataset_pdf/`):

```bash
python pdfTextExtract.py Dataset_pdf/
```

This produces `ai_chunks.json`, a list of `{"text": ..., "source": ...}` chunks
ready to be embedded and stored in ChromaDB.

## Status

Early development. PDF extraction and chunking (`pdfTextExtract.py`) are working.
Embeddings, RAG core, and the API server are not yet implemented.

## Roadmap

- [ ] `embeddings.py`: embed chunks and persist them in ChromaDB
- [ ] `rag.py`: retrieve relevant chunks and query Ollama for an answer
- [ ] `app.py`: Flask API (`POST /query`)
- [ ] `example_usage.py`: end-to-end ingestion + query example
