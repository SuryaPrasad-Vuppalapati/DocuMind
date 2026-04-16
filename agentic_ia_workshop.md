# Agentic IA Hackathon — Kaggle Week

---

## What is an Agent?

An AI agent is an autonomous software system powered by a Large Language Model (LLM) that can perceive its environment, make decisions, and take actions to achieve a specific goal.

**3 Pillars:**

- **Reasoning (LLM)** — Breaks down complex goals into step-by-step plans
- **Memory (Context)** — The system's ability to remember past interactions
- **Tools (Action)** — The "hands" of the agent; how the LLM interacts with the real world through APIs, web scrapers, etc.

---

## Where Agents Are Not Well Suited

- Agents are not suited for highly sensitive tasks because they can fail via hallucination or wrong answers
- Agents boost productivity only when reviewing their work is faster than doing the work yourself
- Like machine learning models: *all models are false, but some are useful*

---

## MCP (Model Context Protocol)

MCP acts as an adapter between LLMs and external tools/systems. It provides:

- A standard system that simplifies development
- An added layer of security
- Communication via **JSON-RPC 2.0** between client and server

**History:**
- Pre-2023: Early developers relied on frameworks and strict prompt engineering
- 2023: OpenAI pioneered tool use in GPT-3.5/4.0
- Anthropic (Claude) and Google (Gemini) followed with native tool use
- 2024: Anthropic open-sourced MCP, solving the N×M integration problem

**Known Issues:**
- Too many overlapping tools can confuse the LLM
- Natural language can feel limiting for complex use cases
- MCP is being challenged by newer technologies

**Getting started:** Use [fastMCP](https://gofastmcp.com/getting-started/quickstart) — build tools as simple Python functions and run a server locally.

---

## Agentic Use Cases

- **Agentic Chatbot** — Ask a question; the agent uses documentation lookup and database queries to answer
- **Human-in-the-Loop Workflow** — The agent chains multiple steps with your feedback at each stage
  - Example: Pull server logs → detect bug → propose fix → generate pull request

---

## Agentic AI & Data Quality

LLMs are more likely to hallucinate when handling messy, unordered data.

- Easier for LLMs to work in a clean codebase
- Easier for LLMs to respond using well-ordered, documented databases
- **Data Engineering is still at the heart of the process** — garbage in, garbage out

---

## Multi-Agent Architecture

The key is to reduce the amount of information any single agent must process.

| Agent | Role |
|---|---|
| Supervisor Agent | Routes the request to the best-suited agent |
| Data Retrieval Agent | Fetches the required data |
| Policy Agent | Checks whether the requested action is permitted |
| Execution Agent | Executes a specific task |

> Trying to do too much with a single agent leads to errors.

---

## AI Security

The attack vector is no longer just malicious code — it's **malicious conversation**.

- **Prompt Injection** — Injecting instructions via user input
- **Data Leakage** — Sensitive data exposed through model outputs
- **Blindly Trusting the Output** — Acting on unverified AI responses

---

## Testing

Traditional software testing relies on exact matches. AI agents produce open-ended, non-deterministic outputs — a different approach is needed.

- **LLM as Judge** — Use a capable model to automatically review and score agent outputs
- **Set your Grade** — The Judge follows strict rules: accuracy, tone, hallucination, tool usage
- **Python packages:** `DeepEval`, `Ragas`, `TruLens`

---

## LLM Observability

- **Tracing** — Visualize the exact sequence of sub-agents, tool calls, and prompts for each request
- **Token & Cost Tracking** — Monitor financial cost and latency in real-time
- **Context Debugging** — See exactly what data was injected into the prompt at the moment of hallucination

---

## Hackathon Project Steps

### Step 1 — Build the MCP RAG Server (`app/mcp_server.py`)

Build an MCP server that exposes one or more live web search tools (your corpus may be outdated).

**Data sources to consider:**
- [Tavily](https://www.tavily.com)
- `duckduckgo-search` Python package
- Scraping APIs (BeautifulSoup, etc.)

**Agent pipeline:**
1. **Agent 1 — Internal Researcher** — finds the answer in the corpus
2. **Agent 2 — External Fact-Checker** — uses MCP to search the live internet to verify
3. **Agent 3 — Synthesizer** — combines both outputs into a final response

---

### Step 2 — Assemble your Multi-Agent Team (`app/agents.py`)

Update your MCP server to expose a `create_markdown_report` tool that saves a `.md` file to disk.

**Human-in-the-Loop flow:**
- Agents research and draft a response
- Human approves the draft (HITL)
- Action Agent publishes the document (Slack, directory, etc.)

**Routing options:**
- Add a Supervisor/Router Agent (LLM-driven)
- Let the Synthesizer act as the router (delegation)
- Let the human decide (rule-based / no AI)

---

### Step 3 — Even More Complex Multi-Agent Logic (`app/main.py`)

Update your MCP Server to expose an `add_to_database` tool. This lets an agent dynamically update the knowledge base when new information arrives.

---

### Step 4 — Handle PDF Files in Your RAG

Add an option to load PDF files directly into your database. Your model should be able to convert a PDF to a readable format via MCP and ingest it into the database.

---

### Step 5 — Integrate Your RAG into a Real Application via Signal

Connect your RAG to a messaging app (Signal, WhatsApp, Teams, Slack, email) for end-to-end LLM-powered conversations.

- [`signal-cli`](https://github.com/AsamK/signal-cli)
- [`signalbot`](https://pypi.org/project/signalbot/) Python package

---

### Go Even Further

- Generate graphs and specific files using MCP + `matplotlib` / Mermaid (Visualizer Agent)
