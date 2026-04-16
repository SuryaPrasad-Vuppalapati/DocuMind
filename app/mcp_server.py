import os
import sys
import tempfile
from duckduckgo_search import DDGS

# Add project root to path so we can import scripts.rag
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from mcp.server.fastmcp import FastMCP
from scripts.rag.pipeline import RAGPipeline

# Initialize the MCP Server
mcp = FastMCP("DocuMind-RAG-Server")

# Initialize the RAG pipeline globally so tools can use it
data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
pipeline = RAGPipeline(
    corpus_path=os.path.join(data_dir, "corpus.json"),
    index_path=os.path.join(data_dir, "my_index.faiss"),
    chunks_path=os.path.join(data_dir, "chunks.json"),
)

@mcp.tool()
def search_documents(query: str, k: int = 3) -> str:
    """
    Search the INTERNAL document corpus for relevant information.
    
    Args:
        query: The search query to look for in the documents.
        k: The number of results to return.
    """
    try:
        answer, results = pipeline.ask(query=query, k=k)
        output = f"RAG Generated Answer:\n{answer}\n\nSources:\n"
        for i, chunk in enumerate(results, start=1):
            _source = chunk.get("source", "Unknown")
            _text = chunk.get("text", "")[:300]
            output += f"{i}. Source: {_source}\n   TextSnippet: {_text}...\n\n"
        return output
    except Exception as e:
        return f"Error searching documents: {str(e)}"

@mcp.tool()
def live_web_search(query: str, max_results: int = 3) -> str:
    """
    Search the live internet for factual verification or real-time data using DuckDuckGo.
    
    Args:
        query: The search query.
        max_results: The maximum number of web results.
    """
    try:
        results = DDGS().text(query, max_results=max_results)
        if not results:
            return "No web search results found."
        
        output = "Live Web Search Results:\n"
        for i, res in enumerate(results, start=1):
            title = res.get('title', '')
            body = res.get('body', '')
            url = res.get('href', '')
            output += f"{i}. {title}\n   {body}\n   URL: {url}\n\n"
        return output
    except Exception as e:
        return f"Error performing web search: {str(e)}"

@mcp.tool()
def create_markdown_report(report_content: str, filename: str = "report.md") -> str:
    """
    Saves a generated report to the disk as a markdown file.
    
    Args:
        report_content: The markdown content of the report.
        filename: The name of the file to save it to.
    """
    dir_path = os.path.join(os.path.dirname(__file__), "..", "data", "reports")
    os.makedirs(dir_path, exist_ok=True)
    filepath = os.path.join(dir_path, filename)
    try:
        with open(filepath, "w") as f:
            f.write(report_content)
        return f"Report saved successfully to {filepath}"
    except Exception as e:
        return f"Failed to save report: {str(e)}"

@mcp.tool()
def add_to_database(text: str, source: str = "Dynamic Update") -> str:
    """
    Dynamically update the knowledge base when new information arrives.
    
    Args:
        text: The content to ingest.
        source: The name/title of the source content.
    """
    try:
        # Save temp pdf/text to ingest into corpus, or directly append to corpus.json
        pipeline.load_if_needed()
        new_doc = {"source": source, "page": 1, "text": text}
        pipeline.loaded_chunks.append(new_doc)
        
        # NOTE: A robust implementation would rebuild the index here.
        # For simplicity in this demo, we add it to the chunks array dynamically.
        # It's currently partially implemented.
        return f"Successfully added new information from {source} into the database cache."
    except Exception as e:
        return f"Failed to add to database: {str(e)}"

if __name__ == "__main__":
    print("Starting DocuMind MCP Server...", file=sys.stderr)
    mcp.run(transport='stdio')
