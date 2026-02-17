# 𓁟 Thoth — Private Knowledge Agent

Thoth is a **local-first, privacy-focused knowledge agent** that combines Retrieval-Augmented Generation (RAG) with multi-source information retrieval. It lets you upload your own documents, ask questions in a conversational chat interface, and get cited answers drawn from your documents, Wikipedia, Arxiv, and the web — all powered by a locally-running LLM via Ollama.

### Why "Thoth"?

In ancient Egyptian mythology, **Thoth** (𓁟) was the god of wisdom, writing, and knowledge — the divine scribe who recorded all human understanding. He was credited with inventing hieroglyphs, maintaining the library of the gods, and serving as the impartial judge of truth. Naming a private knowledge agent after Thoth felt fitting: like its namesake, this tool is built to gather, organize, and faithfully retrieve knowledge — while keeping everything under your control, running locally on your own machine.

---

## Features

### Chat & Conversation Management
- **Multi-turn conversational Q&A** with full message history
- **Persistent conversation threads** stored in a local SQLite database
- **Auto-naming** — threads are automatically renamed to the first question asked
- **Thread switching** — resume any previous conversation seamlessly
- **Thread deletion** — remove conversations you no longer need

### Model Selection
- **Dynamic model switching** — choose any Ollama-supported model from the Settings panel in the sidebar
- **Curated model list** — includes popular models (Llama, Qwen, Gemma, Mistral, DeepSeek, Phi, etc.) alongside any models you've already downloaded
- **Automatic download** — selecting a model you haven't downloaded yet triggers an in-app download with a live progress indicator
- **First-run setup** — if the default model isn't available, the app automatically downloads it on startup
- **Local indicators** — models are marked with ✅ (downloaded) or ⬇️ (needs download) in the selector

### Intelligent Context Retrieval
- **Smart context assessment** — an LLM-powered node decides whether additional context is needed before searching
- **Accumulated context** — context from multiple queries within a thread builds up rather than being replaced
- **Configurable retrieval sources** — toggle each retrieval backend on/off from the Settings panel:
  | Source | Description |
  |--------|-------------|
  | **📄 Documents** | FAISS vector similarity search over your indexed files |
  | **🌐 Wikipedia** | Real-time Wikipedia article retrieval |
  | **📚 Arxiv** | Academic paper search via the Arxiv API |
  | **🔍 Web Search** | Live web search via the Tavily Search API |
- **Context compression** — retrieved content is compressed by the LLM to keep only relevant information while preserving source citations

### Document Management
- **Upload & index** PDF, DOCX, DOC, and TXT files
- **Automatic chunking** with `RecursiveCharacterTextSplitter` (4000-char chunks, 200-char overlap)
- **FAISS vector store** with persistent local storage
- **Embedding model**: `Qwen/Qwen3-Embedding-0.6B` via HuggingFace
- **Duplicate detection** — already-processed files are skipped
- **Clear all** — one-click reset of the entire vector store and processed files list

### Source Citation
Every piece of information in an answer is cited:
- `(Source: document.pdf)` for uploaded documents
- `(Source: https://en.wikipedia.org/...)` for Wikipedia
- `(Source: https://arxiv.org/abs/...)` for Arxiv papers
- `(Source: https://...)` for web search results
- `(Source: Internal Knowledge)` when the LLM uses its own training data

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend (app.py)              │
│  ┌──────────┐   ┌─────────────────────┐   ┌────────────┐  │
│  │ Sidebar  │   │    Chat Interface   │   │  Document  │  │
│  │ Threads  │   │   (Q&A Messages)    │   │  Manager   │  │
│  └──────────┘   └─────────────────────┘   └────────────┘  │
└────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────┐
│                 LangGraph RAG Pipeline (rag.py)            │
│                                                            │
│   START ──▶ needs_context ──┬──▶ get_context ──▶ generate  │
│                             │                    _answer   │
│                             └──────────────────▶ generate  │
│                                                  _answer   │
│                                                    │       │
│                                                    ▼       │
│                                                   END      │
└────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
  ┌────────────┐    ┌──────────────┐     ┌──────────────┐
  │  Ollama    │    │   Retrievers │     │   SQLite     │
  │  LLM      │    │  (FAISS,     │     │  Checkpointer│
  │(qwen3-vl) │    │   Wiki,      │     │  (threads.db)│
  └────────────┘    │   Arxiv,Web) │     └──────────────┘
                    └──────────────┘
```

### LangGraph State Machine

The RAG pipeline is implemented as a LangGraph `StateGraph` with three nodes:

1. **`needs_context`** — Asks the LLM whether the current question can be answered with existing accumulated context or if new retrieval is needed. Returns `Yes`/`No`.
2. **`get_context`** — Queries all four retrieval backends in parallel, combines the results, and uses the LLM to compress the context down to only relevant information (preserving source citations).
3. **`generate_answer`** — Formats the system prompt, accumulated context, and user question into a final prompt and generates the answer with citations.

A conditional edge routes from `needs_context` to either `get_context` or directly to `generate_answer`.

---

## Project Structure

```
Thoth/
├── app.py                  # Streamlit frontend — UI, chat, document upload
├── rag.py                  # LangGraph RAG pipeline — nodes, edges, state
├── documents.py            # Document loading, chunking, FAISS vector store
├── models.py               # LLM configuration (Ollama)
├── threads.py              # Thread/conversation management (SQLite)
├── api_keys.py             # API key configuration
├── processed_files.json    # Tracks which files have been indexed (auto-generated)
├── threads.db              # SQLite database for thread metadata (auto-generated)
├── vector_store/           # FAISS index files (auto-generated)
│   ├── index.faiss
│   └── index.pkl
└── README.md
```

### Module Descriptions

| File | Purpose |
|------|---------|
| **`app.py`** | Streamlit application with three-panel layout: sidebar (threads + settings), center (chat), right (documents). Handles UI state, file uploads, model selection, retrieval source toggles, and invokes the RAG graph. |
| **`rag.py`** | Defines the LangGraph state machine with `SessionState`, retriever initialization, context compression, and answer generation. Also supports a CLI mode via `__main__`. |
| **`documents.py`** | Manages document ingestion: loading (PDF/DOCX/TXT), text splitting, embedding with `Qwen/Qwen3-Embedding-0.6B`, FAISS storage, and processed file tracking. |
| **`models.py`** | LLM model management — listing, downloading, and switching Ollama models at runtime. |
| **`threads.py`** | SQLite-backed thread metadata (create, list, rename, delete) and LangGraph `SqliteSaver` checkpointer for persisting conversation state. |
| **`api_keys.py`** | Sets environment variables for external API keys (Tavily). |

---

## Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.com/)** installed and running locally
- **Tavily API Key** for web search (set in `api_keys.py`)

> **Note:** You no longer need to manually pull a model — the app will automatically download the default model (`qwen3:8b`) on first run if it isn't available.

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/thoth.git
   cd thoth
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**
   ```bash
   # Windows
   .venv\Scripts\activate

   # macOS / Linux
   source .venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   pip install streamlit langchain-community langchain-core langchain-classic langchain-huggingface langchain-ollama langgraph faiss-cpu torch transformers pypdf python-docx unstructured
   ```

5. **Configure API keys**

   Edit `api_keys.py` and set your Tavily API key:
   ```python
   os.environ["TAVILY_API_KEY"] = "your-tavily-api-key"
   ```

6. **Ensure Ollama is running**
   ```bash
   ollama serve
   ```

---

## Usage

### Web Interface (Streamlit)

```bash
streamlit run app.py
```

This opens the Thoth web UI in your browser with:
- **Left sidebar**: Create, switch, and delete conversation threads; Settings panel at the bottom for model selection and retrieval source toggles
- **Center**: Chat interface for asking questions
- **Right panel**: Upload and manage documents

### CLI Mode

```bash
python rag.py
```

This starts an interactive terminal session where you can select/create threads and ask questions directly.

---

## How It Works

1. **User asks a question** in the chat interface.
2. The **`needs_context` node** evaluates whether the accumulated context from previous turns is sufficient or if new retrieval is needed.
3. If new context is needed, the **`get_context` node** queries the enabled sources (configurable via Settings):
   - FAISS vector store (uploaded documents)
   - Wikipedia API
   - Arxiv API
   - Tavily web search
4. Retrieved content is **compressed** by the LLM to remove irrelevant information while preserving source citations.
5. The compressed context is **appended** to the existing context (not replaced).
6. The **`generate_answer` node** combines the system prompt, all accumulated context, and the question to produce a cited answer.
7. The full conversation state is **checkpointed** in SQLite, enabling thread persistence across sessions.

---

## Configuration

### LLM Model
Select a model directly from the **⚙️ Settings** panel in the sidebar. You can also change the default model in `models.py`:
```python
DEFAULT_MODEL = "qwen3:8b"  # Change to any Ollama-supported model
```

### Embedding Model
Change the embedding model in `documents.py`:
```python
embedding_model = HuggingFaceEmbeddings(
    model_name="Qwen/Qwen3-Embedding-0.6B"  # Change to any HuggingFace embedding model
)
```

### Chunking Parameters
Adjust text splitting in `documents.py`:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,      # Characters per chunk
    chunk_overlap=200     # Overlap between chunks
)
```

### Retriever Settings
Modify the number of documents retrieved in `rag.py`:
```python
document_retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Top-k results
```

---

## Supported File Types

| Extension | Loader |
|-----------|--------|
| `.pdf` | `PyPDFLoader` |
| `.docx` | `UnstructuredWordDocumentLoader` |
| `.doc` | `UnstructuredWordDocumentLoader` |
| `.txt` | `TextLoader` |

---

## License

This project is licensed under the [MIT License](LICENSE).
