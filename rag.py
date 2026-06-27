import sys

import requests
from sentence_transformers import SentenceTransformer

from embeddings import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, get_collection

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"
TOP_K = 5


def retrieve(question, k=TOP_K, model_name=EMBEDDING_MODEL,
             persist_dir=CHROMA_DIR, collection_name=COLLECTION_NAME):
    model = SentenceTransformer(model_name)
    vector = model.encode(question).tolist()
    collection = get_collection(persist_dir, collection_name)
    results = collection.query(query_embeddings=[vector], n_results=k)
    docs = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return docs, sources


def build_prompt(question, docs):
    context = "\n\n---\n\n".join(docs)
    return (
        "Use the following context to answer the question. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )


def query_ollama(prompt, model=OLLAMA_MODEL, url=OLLAMA_URL):
    response = requests.post(url, json={"model": model, "prompt": prompt, "stream": False})
    response.raise_for_status()
    return response.json()["response"]


def answer(question, k=TOP_K, model=OLLAMA_MODEL):
    docs, sources = retrieve(question, k)
    prompt = build_prompt(question, docs)
    response = query_ollama(prompt, model=model)
    unique_sources = list(dict.fromkeys(sources))
    return {"answer": response, "sources": unique_sources}


if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        result = answer(question)
        print("\nAnswer:", result["answer"])
        print("\nSources:", ", ".join(result["sources"]))
    else:
        print("DocuBot — type your question or 'quit' to exit\n")
        while True:
            try:
                question = input("Q: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question:
                continue
            if question.lower() in ("quit", "exit", "q"):
                break
            result = answer(question)
            print(f"\nA: {result['answer']}")
            print(f"Sources: {', '.join(result['sources'])}\n")
