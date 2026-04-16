import os
import json
import logging
from openai import OpenAI
from typing import List, Dict, Callable, Optional
from dotenv import load_dotenv
from duckduckgo_search import DDGS

load_dotenv()

# Add project root to path so we can import scripts.rag
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts.rag.pipeline import RAGPipeline

# Initialize the OpenAI client natively
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Initialize RAG Pipeline globally for the agents
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
pipeline = RAGPipeline(
    corpus_path=os.path.join(data_dir, "corpus.json"),
    index_path=os.path.join(data_dir, "my_index.faiss"),
    chunks_path=os.path.join(data_dir, "chunks.json"),
)

class Agent:
    def __init__(self, name: str, instructions: str, tools: Optional[List[Callable]] = None):
        self.name = name
        self.instructions = instructions
        self.tools = tools or []
        # Convert tools to OpenAI format
        self.tool_definitions = []
        for tool in self.tools:
            # simple single arg query tool builder
            # (Note for more complex signatures we'd use inspect standard but this is simple)
            props = {}
            required = []
            
            # hardcode duck duck param
            if tool.__name__ == "add_to_database":
                props = {
                    "text": {"type": "string"},
                    "source": {"type": "string"}
                }
                required = ["text"]
            elif tool.__name__ == "create_markdown_report":
                 props = {
                    "report_content": {"type": "string"},
                    "filename": {"type": "string"}
                }
                 required = ["report_content"]
            else:
                props = {"query": {"type": "string"}}
                required = ["query"]
                
            self.tool_definitions.append({
                "type": "function",
                "function": {
                    "name": tool.__name__,
                    "description": tool.__doc__,
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": required
                    }
                }
            })

    def chat(self, messages: List[Dict]) -> str:
        sys_message = {"role": "system", "content": self.instructions}
        current_messages = [sys_message] + messages

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=current_messages,
            tools=self.tool_definitions if self.tool_definitions else None,
        )

        message = response.choices[0].message
        
        if message.tool_calls:
            current_messages.append(message)
            for tool_call in message.tool_calls:
                tool_func = next((t for t in self.tools if t.__name__ == tool_call.function.name), None)
                if tool_func:
                    kwargs = json.loads(tool_call.function.arguments)
                    tool_result = tool_func(**kwargs)
                    current_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_func.__name__,
                        "content": str(tool_result)[:2000]
                    })
            return self.chat(current_messages[1:])
        return message.content

# ---- Define Tools for the Agents ----

def search_knowledge_base(query: str) -> str:
    """Use this tool to search the INTERNAL document database for facts."""
    print(f"\n[Tool] Searching Internal DB: '{query}'")
    try:
        answer, results = pipeline.ask(query=query, k=3)
        formatted = f"Internal Setup:\n{answer}\nEvidence:\n"
        for i, chunk in enumerate(results, start=1):
            formatted += f"[{i}] {chunk.get('source', '')} - {chunk.get('text', '')[:200]}...\n"
        return formatted
    except Exception as e:
        return f"Error: {e}"

def live_web_search(query: str) -> str:
    """Use this tool to search the live internet for factual verification or recent info via duckduckgo."""
    print(f"\n[Tool] Searching Live Web: '{query}'")
    try:
        results = DDGS().text(query, max_results=3)
        if not results: return "No current web results."
        output = "Live Web Results:\n"
        for i, res in enumerate(results, start=1):
            output += f"{i}. {res.get('title')} - {res.get('body')} (URL: {res.get('href')})\n"
        return output
    except Exception as e:
        return f"Web Search Error: {e}"
        
def create_markdown_report(report_content: str, filename: str = "report.md") -> str:
    """Saves a generated report to the disk."""
    filepath = os.path.join(os.path.dirname(__file__), "..", "data", filename)
    try:
        with open(filepath, "w") as f:
            f.write(report_content)
        return f"Saved report to {filepath}"
    except Exception as e:
        return f"Failed to save: {e}"
        
def add_to_database(text: str, source: str = "Agent Update") -> str:
    """Dynamically update the knowledge base with new information."""
    print(f"\n[Tool] Adding to Database from {source}")
    try:
        pipeline.load_if_needed()
        pipeline.loaded_chunks.append({"source": source, "page": 1, "text": text})
        return f"Added new info into the database cache."
    except Exception as e:
        return f"Failed to add: {e}"

# ---- Define the Multi-Agent Team ----

# 1. Agent 1 - Internal Researcher
researcher_agent = Agent(
    name="Internal Researcher",
    instructions="You are an expert Document Researcher. Find factual data using `search_knowledge_base` inside our corpus.",
    tools=[search_knowledge_base]
)

# 2. Agent 2 - External Fact-Checker
factchecker_agent = Agent(
    name="External Fact-Checker",
    instructions="You verify data externally. Use `live_web_search` to verify or find new up-to-date data. Also use `add_to_database` if you find something useful to save for the future.",
    tools=[live_web_search, add_to_database]
)

# 3. Agent 3 - Synthesizer
synthesizer_agent = Agent(
    name="Synthesizer",
    instructions="Compile notes from the Internal Researcher and External Fact Checker into a clear markdown response. Use `create_markdown_report` to save your final draft if asked.",
    tools=[create_markdown_report]
)

def run_agent_team(user_query: str) -> str:
    """Orchestrates the multi-agent workflow."""
    print(f"\n--- Assigning task to {researcher_agent.name} ---")
    internal_notes = researcher_agent.chat([{"role": "user", "content": user_query}])
    print(f"\n[Internal Findings]:\n{internal_notes}")

    print(f"\n--- Assigning task to {factchecker_agent.name} ---")
    verification_notes = factchecker_agent.chat([{"role": "user", "content": f"Verify / Expand this data via live search:\n\n{internal_notes}"}])
    print(f"\n[External Verification]:\n{verification_notes}")

    print(f"\n--- Assigning task to {synthesizer_agent.name} ---")
    final_draft = synthesizer_agent.chat([{"role": "user", "content": f"Please synthesize an official answer based on:\n\nINTERNAL:\n{internal_notes}\n\nEXTERNAL VERIFICATION:\n{verification_notes}"}])
    return final_draft
