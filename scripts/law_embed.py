import json
import argparse
import os
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm

ROOT_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_FILE       = os.path.join(ROOT_DIR, "data", "data.json")
VECTOR_DB       = os.path.join(ROOT_DIR, "vectorstores", "law")
COLLECTION_NAME = "indonesian_law_rag"
EMBED_MODEL     = "bge-m3"
BATCH_SIZE      = 32


def load_chunks(json_path: str) -> list[Document]:
    documents = []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for article in data:

        base_metadata = article.get("metadata", {})

        for chunk in article.get("chunks", []):

            if chunk.get("chunk_type") != "combined":
                continue

            doc = Document(
                page_content=chunk.get("content", ""),
                metadata={
                    "article_id": article.get("article_id"),
                    "article_number": article.get("article_number"),
                    "chunk_id": chunk.get("chunk_id"),
                    "chunk_type": chunk.get("chunk_type"),
                    "source_article": chunk.get("source_article"),
                    "source": base_metadata.get("source"),
                    "book": base_metadata.get("book"),
                    "chapter": base_metadata.get("chapter"),
                }
            )

            documents.append(doc)

    return documents

def main():
    parser = argparse.ArgumentParser(description="Embed chunks to ChromaDB")
    parser.add_argument("--reset", action="store_true", help="Reset vector database before embedding")
    args = parser.parse_args()


    print(f"\nLoading chunks from: {JSON_FILE}")
    try:
        docs = load_chunks(JSON_FILE)
        print(f"   Successfully loaded {len(docs)} chunks.")
    except FileNotFoundError:
        print(f"   File not found: {JSON_FILE}")
        return
    except Exception as e:
        print(f"   Error reading JSON: {e}")
        return

    if not docs:
        print("   No valid chunks to embed.")
        return

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    if args.reset:
        print("\nResetting ChromaDB collection...")
        import shutil
        if os.path.exists(VECTOR_DB):
            shutil.rmtree(VECTOR_DB)
        print("   Old database deleted.")

    print(f"\nSaving to ChromaDB: {VECTOR_DB} (Collection: {COLLECTION_NAME})")

    vectorstore = None
   

    for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Embedding Progress", unit="batch"):
        batch_docs = docs[i : i + BATCH_SIZE]

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch_docs,
                embedding=embeddings,
                persist_directory=VECTOR_DB,
                collection_name=COLLECTION_NAME
            )
        else:
            vectorstore.add_documents(batch_docs)




if __name__ == "__main__":
    main()
