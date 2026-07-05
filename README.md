# Indonesian Legal & Cultural Assistant

A RAG-powered chatbot that helps **international tourists, expats, and digital nomads** navigate Indonesian law and Balinese cultural norms — so they stay safe, informed, and respectful.

---

## Features

- **Dual-domain RAG** — queries both a law corpus and a Balinese culture corpus simultaneously, returning two focused answers per question
- **Hybrid Retrieval** — combines dense vector search (ChromaDB) with sparse BM25 keyword matching for high-recall candidate generation
- **Cross-Encoder Reranking** — BAAI/bge-reranker-v2-m3 scores and reranks candidates for precision
- **Direct Article Lookup** — exact metadata match when users reference a specific article/pasal number (e.g. "Article 458")
- **Gemini + Ollama** — uses Google Gemini Flash by default; falls back to local Ollama automatically if unavailable
- **Flask Web UI** — clean browser-based chat interface

---

## Project Structure

```
.
├── app.py                   # Flask entry point — serves UI and /ask endpoint
├── .env                     # API keys (not committed to git)
│
├── core/
│   ├── __init__.py
│   ├── vector.py            # ChromaDB, BM25, reranker, HybridRetriever
│   └── rag.py               # RAG logic: translation, retrieval, LLM call, response
│
├── scripts/
│   ├── bali_embed.py        # One-time script to embed Balinese culture data
│   └── law_embed.py         # One-time script to embed Indonesian law data
│
├── templates/
│   └── index.html           # Frontend chat UI
│
├── data/
│   ├── data.json            # Indonesian law corpus source
│   └── bali.json            # Balinese ceremony & taboo source
│
├── pdf/                     # Source PDF documents (law texts)
│
└── vectorstores/
    ├── law/                 # Persisted ChromaDB — law domain
    └── culture/             # Persisted ChromaDB — culture domain
```

---

## Tech Stack

| Layer | Component |
|---|---|
| **LLM (primary)** | Google Gemini Flash via `langchain-google-genai` |
| **LLM (fallback)** | Ollama `qwen2.5:3b` |
| **Embeddings** | Ollama `bge-m3` |
| **Vector Store** | ChromaDB (persisted locally) |
| **Sparse Retrieval** | BM25Okapi via `rank_bm25` |
| **Reranker** | `BAAI/bge-reranker-v2-m3` via `sentence-transformers` |
| **Orchestration** | LangChain (`langchain-core`, `langchain-ollama`, `langchain-chroma`) |
| **Web Server** | Flask |
| **Frontend** | Vanilla HTML / CSS / JS |

---

## Getting Started

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally
- Required Ollama models:

```bash
ollama pull bge-m3
ollama pull qwen2.5:3b
```

### Installation

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd <repo-folder>

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install flask langchain langchain-core langchain-ollama langchain-chroma \
            langchain-google-genai chromadb rank_bm25 sentence-transformers \
            numpy python-dotenv
```

### Configure API Key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

Get a free key at [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey).

If no key is provided, the app falls back to Ollama automatically.

### Run

```bash
python app.py
```

Open your browser at **http://127.0.0.1:5000**

---

## How It Works

### Retrieval Pipeline

```
User Question
     |
     v
[Direct Article Lookup]  <-- if article/pasal number detected in query
     |  (fallback if no exact match)
     v
[Hybrid Candidate Generation]
     |- Vector Search   (ChromaDB top-10)
     |- BM25 Search     (BM25Okapi top-10)
     `- Deduplicated by canonical article ID
     |
     v
[Cross-Encoder Reranker]
     BAAI/bge-reranker-v2-m3 -> top-5 docs
     |
     v
[Build Context]  (max 16,000 chars)
     |
     v
[LLM — Gemini Flash / Ollama]
     |- Law answer    (from law context)
     `- Culture answer (from culture context)
```

Each question is answered from **both** domains independently. The frontend displays both perspectives side by side.

### Knowledge Domains

| Domain | Contents |
|---|---|
| **Law** | Indonesian Penal Code, Immigration Law (UU No. 6/2011, UU No. 18/2025) |
| **Culture** | Balinese ceremonies & taboos — Nyepi, Galungan, Ngaben, temple etiquette, offerings, etc. |

---

## API

### `POST /ask`

**Request:**
```json
{ "question": "Can I fly a drone near a temple in Bali?" }
```

**Response:**
```json
{
  "answer_law": "...",
  "answer_culture": "...",
  "docs_law": 5,
  "docs_culture": 5,
  "llm_used": "gemini",
  "time": 2.84
}
```

---

## Configuration

Key constants to tune:

**`core/vector.py`**
```python
EMBED_MODEL    = "bge-m3"
VECTOR_TOP_K   = 10
BM25_TOP_K     = 10
RERANK_TOP_K   = 5
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
```

**`core/rag.py`**
```python
GEMINI_MODEL      = "gemini-3.1-flash-lite"
OLLAMA_MODEL      = "qwen2.5:3b"
TEMPERATURE       = 0.1
MAX_CONTEXT_CHARS = 16000
```

---

## Re-Embedding Data

If you update the source JSON files, re-run the embedding scripts from the project root:

```bash
# Embed law corpus
python scripts/law_embed.py

# Embed culture corpus
python scripts/bali_embed.py

# Reset and re-embed from scratch
python scripts/law_embed.py --reset
python scripts/bali_embed.py --reset
```

---

## License

Educational and research use.

---

## Acknowledgements

- [Google Gemini](https://deepmind.google/technologies/gemini/) — primary LLM
- [Ollama](https://ollama.com/) — local LLM fallback & embeddings
- [BAAI](https://huggingface.co/BAAI) — `bge-m3` embeddings & `bge-reranker-v2-m3` reranker
- [LangChain](https://www.langchain.com/) — orchestration
- [ChromaDB](https://www.trychroma.com/) — vector persistence
