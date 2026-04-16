# DocuMind: PDF RAG & Extraction Pipeline
---

## Project Description
DocuMind is an intelligent Retrieval-Augmented Generation (RAG) pipeline and document interaction platform. 
It processes large-scale PDF documentation, cleans the extracted text, and generates a structured corpus. 
Using this corpus, it builds a highly accurate semantic search engine using FAISS and OpenAI embeddings. 

Users can interact with the corpus through a web-based Gradio interface. The system features smart comparison logic (allowing cross-referencing between multiple entities), robust prompt safety checks, semantic text chunking, inline PDF highlight generation, and real-time API cost tracking.

## Getting Started

### 1. Prerequisites
- **Python 3.9+**
- **uv** package manager (`pip install uv`)
- An active **OpenAI API Key**

### 2. Environment Setup
Create a `.env` file in the root directory of the project and add your OpenAI API key:
```env
OPENAI_API_KEY=your_api_key_here
```

### 3. Installation
We use `uv` for fast dependency management. Install the required dependencies:
```bash
uv pip install -r requirements.txt
```

---

## Steps to Run

### Running the Application
To launch the interactive Gradio web server, simply run:
```bash
uv run python app/app.py
```
This handles starting the UI where you can:
1. **Upload PDFs**: Directly via the interface.
2. **Build Index**: Automatically chunks text semantically and builds the FAISS vector index.
3. **Interactive RAG Chat**: Ask questions or compare concepts directly against the indexed documents. The UI will optionally display the exact PDF page highlighting the source of the answer.
4. **Multi-Agent Analysis (Human in Loop)**: Request comprehensive Markdown reports in a separate tab. A dedicated AI Researcher scours your vector DB (via tool integrations) and DuckDuckGo, while an AI Writer drafts content. A human operator can review the preview, request AI revisions via a feedback loop, and approve the final Markdown draft which persists cleanly to disk.

### Backend Scripts (Optional)
If you need to strictly test the extraction backend independently:
```bash
uv run python scripts/extract.py
```
