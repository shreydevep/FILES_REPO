import json
import chromadb
from chromadb.config import Settings
from pathlib import Path

# === CONFIG ===
JSON_INPUT_PATH = "data/input_docs.json"
OUTPUT_DIR = "output"
USE_CHROMA = False  # Set to True to use ChromaDB
CHROMA_QUERY = "Enter your query here"
PROMPT_TEMPLATE = "Answer the following questions using the provided context:\n\n"

# === Output Files ===
Path(OUTPUT_DIR).mkdir(exist_ok=True)
PROMPT_FILE = f"{OUTPUT_DIR}/prompt.txt"
CONTEXT_FILE = f"{OUTPUT_DIR}/context.txt"
QUESTIONS_FILE = f"{OUTPUT_DIR}/questions.txt"
RAW_DOCS_FILE = f"{OUTPUT_DIR}/raw_docs.json"

# === JSON Loader ===
def load_from_json(path):
    with open(path, 'r') as f:
        return json.load(f)

# === Chroma Query ===
def query_chroma(query_text: str, top_k=5):
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))
    collection = client.get_or_create_collection(name="docs")
    results = collection.query(query_texts=[query_text], n_results=top_k)
    return [{"content": doc, "metadata": meta}
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])]

# === Extract Context & Questions ===
def extract_parts(docs):
    context = "\n\n---\n\n".join(doc['content'] for doc in docs)
    questions = [doc['metadata'].get("question", "").strip() for doc in docs if doc['metadata'].get("question")]
    return context, questions

# === Save Helpers ===
def save_text(path, text):
    with open(path, 'w') as f:
        f.write(text)
    print(f"✅ Saved: {path}")

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved: {path}")

# === Main ===
def main():
    if USE_CHROMA:
        docs = query_chroma(CHROMA_QUERY)
    else:
        docs = load_from_json(JSON_INPUT_PATH)

    context, questions = extract_parts(docs)

    # Save files separately
    save_text(PROMPT_FILE, PROMPT_TEMPLATE)
    save_text(CONTEXT_FILE, context)
    save_text(QUESTIONS_FILE, "\n".join(questions))
    save_json(RAW_DOCS_FILE, docs)

if __name__ == "__main__":
    main()
