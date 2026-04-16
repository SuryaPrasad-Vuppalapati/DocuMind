# DocuMind RAG Pipeline - Project Instructions & Architecture

This document provides a comprehensive overview of the DocuMind RAG project, detailing its architecture, the reasoning behind various components, and an explanation of the core functions driving the application.

## Overview
DocuMind is an intelligent Retrieval-Augmented Generation (RAG) platform built to securely extract, index, and query large PDF documents. It features a two-pronged approach to interacting with data:
1. **Interactive RAG (Standard)**: A standard chat interface where users can ask questions about the uploaded PDFs, and the system retrieves the most relevant chunks, generates an answer, and shows the specific PDF page highlight containing the source information.
2. **Multi-Agent Analysis (Human in Loop)**: An advanced tab featuring multiple AI agents working collaboratively. A "Researcher" agent uses tools to autonomously query the FAISS index (via FastMCP) and search DuckDuckGo, while a "Writer" synthesizes those findings. You can revise the drafts in a feedback loop before approving them to be saved directly to your disk.

## Core Backend Architecture (Scripts)

The backend pipeline (in `scripts/`) is responsible for processing documents without a UI.

### 1. Extractor & Cleaner (`scripts/extract.py`)
PDF documents contain raw text often polluted by headers, footers, ligatures, and watermarks. 
- **`process_pdf(file_path)`**: Uses `PyMuPDF` to read raw text page-by-page.
- **`LLMCleaner.clean_text(text)`**: Uses an LLM (typically a cheaper model like `gpt-4o-mini`) strictly instructed to fix typos and formatting while preserving the original meaning. It acts as an OCR/text-correction refinement step.

### 2. Chunker (`scripts/rag/chunker.py`)
Text needs to be split into sizes that fit into LLM context windows and FAISS embeddings.
- **`SemanticChunker.chunk(corpus)`**: Instead of blindly splitting every 1000 characters (which can cut sentences in half), this function splits text intelligently based on paragraph/sentence boundaries (regex-based). It also creates small overlaps to prevent cutting off context between chunks.

### 3. Embedder & Indexer (`scripts/rag/embedder.py` & `indexer.py`)
- **`OpenAIEmbedder.embed_batch(texts)`**: Uses OpenAI (`text-embedding-3-small`) to convert text strings into dense vector representations.
- **`FAISSIndexer.build_index(embeddings)`**: Consumes the numerical vectors and creates an optimized FAISS L2 index. This makes semantic similarity searches extremely fast.

### 4. Answer Generator Orchestration (`scripts/rag/answer_generator.py`)
This is the brain of the Standard Interactive RAG.
- **`QueryFormatter.format(query)`**: Takes a raw query and uses an LLM to expand acronyms (e.g. `ml -> machine learning`), corrects typos, and outputs 3 different variations of the question to cast a wider search net.
- **`SessionMemoryManager`**: Manages rolling chat history. It tracks the latest questions and intelligently summarizes older context so the LLM doesn't lose the thread of the conversation while keeping token usage low.
- **`IntentClassifier.classify(query)`**: Classifies user queries as "meta" (asking about previous chat), "injection" (trying to jailbreak the bot), or "document" (normal query).
- **`AnswerGenerator.generate()`**: Given the user query and the retrieved FAISS chunks, it strictly forces the LLM to output a JSON object containing an answer, confidence scores, and strict line-by-line citations matching the source chunks. It is aggressively prompted not to hallucinate.
- **`AnswerComparator.pick_best()`**: If the system generated answers based on the 3 different query variants, this asks the LLM to pick the highest quality, most factually supported answer among them.

---

## Frontend Architecture (Gradio App)

The application (`app/app.py` and `app/agents.py`) links the backend RAG pipeline to a user-friendly UI.

### 1. The Standard RAG Interface (`app/app.py` Tab 1)
- **`respond(message, chat_history)`**: The primary Gradio chat callback. It takes user input, triggers the Pipeline (`ag.generate()`), parses the JSON response, outputs it to the chat, and uses PyMuPDF to render an image of the specific PDF page with the cited text highlighted in yellow.
- **`build_index_ui(progress)`**: Triggers the `extraction -> chunking -> embedding -> indexing` sequence when the user hits "Build Corpus Index".
- **`format_answer(json_data)`**: Parses the strict JSON output from `AnswerGenerator` (which contains `[1]`, `[2]` citation markers) and converts it into readable Markdown for the chat window.

### 2. Multi-Agent Analysis (`app/agents.py`, `app/mcp_server.py`)
The second tab uses a Model Context Protocol (FastMCP) approach. Instead of static RAG, an AI is given tools it can run by itself in a loop.
- **`mcp_server.py`**: Registers Python functions as tools using `@mcp.tool()`. 
  - `search_database(query)`: Lets the AI search the FAISS vector database.
  - `web_search(query)`: Lets the AI search DuckDuckGo for external context.
  - `create_markdown_report(...)`: Saves formatted strings directly to `data/reports/` as `.md` files.
- **`Agent` class (`app/agents.py`)**: A wrapper around the OpenAI Chat Completions API. It listens for `tool_calls` in the API response. If the AI decides it needs to run `search_database`, the `Agent.chat()` loop pauses, runs the actual Python tool, appends the result to the messages array, and recurses until the AI gives a final text answer.
- **`agent_generate_draft()`**: The UI callback that prompts the "Researcher Agent" to gather data, then passes that context to the "Synthesizer Agent" to write a Markdown draft.
- **`agent_process_feedback()`**: Prompts the Synthesizer Agent to revise its draft based on human text input.
- **`agent_approve_draft()`**: Extracts the markdown Heading 1 (`# Title`), generates a valid filename, calls `create_markdown_report` to save it to disk, and completely clears the UI (resetting textboxes and hiding previews).

## Directory Structure
- **`app/`**: Contains UI logic (`app.py`), Agent logic (`agents.py`), and the MCP tool definitions (`mcp_server.py`).
- **`data/`**: The runtime storage directory. Holds `chunks.json`, `my_index.faiss` (the vector DB), uploaded `.pdf`s, generated `reports/`, and users' `session_memory/`.
- **`scripts/rag/`**: The core RAG operations (chunking, embedding, indexing, retrieval, QA generation).