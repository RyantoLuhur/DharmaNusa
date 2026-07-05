import json
import argparse
import os
import shutil
from typing import List

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm


ROOT_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_JSON_FILE = os.path.join(ROOT_DIR, "data", "bali.json")
VECTOR_DB       = os.path.join(ROOT_DIR, "vectorstores", "culture")
COLLECTION_NAME = "balinese_culture_rag"
EMBED_MODEL     = "bge-m3"
BATCH_SIZE      = 32


def load_documents(json_path: str) -> List[Document]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs: List[Document] = []

    for ceremony in data.get("ceremonies", []):
        ceremony_name = ceremony.get("ceremony_name", "")
        category = ceremony.get("category", "")
        title = ceremony.get("title", "")

        ceremony_meta = ceremony.get("metadata", {}) or {}
        canonical_id = ceremony_meta.get("id", "")

        for section_name, section_text in ceremony.get("text", {}).items():
            if not section_text:
                continue

            docs.append(
                Document(
                    page_content=f"{title} - {section_name.replace('_', ' ').title()}\n\n{section_text}",
                    metadata={
                        "id": canonical_id,
                        "type": "ceremony",
                        "category": category,
                        "ceremony_name": ceremony_name,
                        "title": title,
                        "section": section_name,
                        "frequency": ceremony_meta.get("frequency", ""),
                        "tags": ",".join(ceremony_meta.get("tags", []) or []),
                        "domain": "culture",
                    },
                )
            )

    for taboo in data.get("taboos", []):
        title = taboo.get("title", "")
        category = taboo.get("category", "")
        text = taboo.get("text", "")

        metadata = taboo.get("metadata", {}) or {}
        canonical_id = metadata.get("id", "")

        content_parts = [
            f"TABOO: {title}",
            f"\n{text}",
        ]

        if metadata.get("rationale"):
            content_parts.append(f"\nRationale: {metadata['rationale']}")

        if metadata.get("consequence"):
            content_parts.append(f"\nConsequence: {metadata['consequence']}")

        if metadata.get("protocol_guide"):
            content_parts.append(f"\nProtocol: {metadata['protocol_guide']}")

        docs.append(
            Document(
                page_content="\n".join(content_parts),
                metadata={
                    "id": canonical_id,
                    "type": "taboo",
                    "category": category,
                    "title": title,
                    "rules_no": str(taboo.get("rules", "")),
                    "severity": metadata.get("severity", "high"),
                    "related_ceremony": ",".join(taboo.get("related_ceremony", []) or []),
                    "domain": "culture",
                },
            )
        )

    return docs


def init_vectorstore(reset: bool, embeddings: OllamaEmbeddings) -> Chroma:
    if reset and os.path.exists(VECTOR_DB):
        shutil.rmtree(VECTOR_DB)
        print("Old vector DB removed")

    return Chroma(
        persist_directory=VECTOR_DB,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
    )


def embed_documents(docs: List[Document], vectorstore: Chroma):
    for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Embedding"):
        batch = docs[i : i + BATCH_SIZE]
        vectorstore.add_documents(batch)

    print(f"Embedded {len(docs)} documents")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=DEFAULT_JSON_FILE)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    print("\nStarting Embedding Pipeline")
    print(f"Model: {EMBED_MODEL}")
    print(f"DB: {VECTOR_DB}")
    print(f"Collection: {COLLECTION_NAME}")

    docs = load_documents(args.json)

    print(f"\nLoaded documents: {len(docs)}")
    print(f"- Ceremonies: {sum(1 for d in docs if d.metadata['type'] == 'ceremony')}")
    print(f"- Taboos: {sum(1 for d in docs if d.metadata['type'] == 'taboo')}")

    missing = [i for i, d in enumerate(docs) if not d.metadata.get("id")]
    if missing:
        print(f"\nWARNING: {len(missing)} document(s) missing 'id' metadata!")
    else:
        print(f"All {len(docs)} documents have canonical `id` metadata")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vectorstore = init_vectorstore(args.reset, embeddings)

    embed_documents(docs, vectorstore)


    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
