import json

import chromadb
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = "ai_chunks.json"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 64


def load_chunks(path=CHUNKS_FILE):
    with open(path) as f:
        return json.load(f)


def get_collection(persist_dir=CHROMA_DIR, collection_name=COLLECTION_NAME):
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(name=collection_name)


def embed_chunks(chunks, model_name=EMBEDDING_MODEL, persist_dir=CHROMA_DIR,
                 collection_name=COLLECTION_NAME, batch_size=BATCH_SIZE):
    """Embed chunks with a sentence-transformer and upsert them into ChromaDB."""
    model = SentenceTransformer(model_name)
    collection = get_collection(persist_dir, collection_name)

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        texts = [chunk["text"] for chunk in batch]
        sources = [chunk["source"] for chunk in batch]
        vectors = model.encode(texts).tolist()
        ids = [f"{source}-{start + i}" for i, source in enumerate(sources)]

        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=[{"source": source} for source in sources],
        )

    return collection


if __name__ == "__main__":
    chunks = load_chunks()
    collection = embed_chunks(chunks)
    print(f"Stored {collection.count()} chunks in '{COLLECTION_NAME}' at '{CHROMA_DIR}'")
