import os
import sys
import json
import shutil
import pymupdf
import base64
import warnings
from PIL import Image
from typing import List, Tuple

import gradio as gr

# Suppress underlying library warnings (Pandas, DuckDuckGo, Gradio deprecations)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from rag.pipeline import RAGPipeline
from extract import extract_from_pdfs
from rag.cost import get_total_cost
from app.agents import researcher_agent, factchecker_agent, synthesizer_agent


def get_header_html():
    logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            b64_image = base64.b64encode(f.read()).decode("utf-8")
        img_html = f'<img src="data:image/png;base64,{b64_image}" style="height: 65px; margin-right: 15px;" />'
    else:
        img_html = ""
    return f"""
    <div style="display: flex; align-items: center;">
        {img_html}
        <div style="display: flex; flex-direction: column; justify-content: center;">
            <h1 style="margin: 0; padding: 0; font-size: 2.2em; font-weight: bold; line-height: 1.1;">DocuMind</h1>
            <p style="margin: 0; padding: 0; color: #666; font-size: 1.1em; line-height: 1.2;">Document Explorer with RAG</p>
        </div>
    </div>
    """


DEFAULT_CORPUS = os.path.join(ROOT_DIR, "data", "corpus.json")
DEFAULT_INDEX = os.path.join(ROOT_DIR, "data", "my_index.faiss")
DEFAULT_CHUNKS = os.path.join(ROOT_DIR, "data", "chunks.json")
PDF_DIR = os.path.join(ROOT_DIR, "data", "pdfs")
SAMPLE_JSON = os.path.join(ROOT_DIR, "data", "sample.json")
REPORT_MD = os.path.join(SCRIPTS_DIR, "EXTRACTION_REPORT.md")


GLOBAL_PIPELINE = None
CURRENT_SESSION_ID = None


def get_pipeline():
    global GLOBAL_PIPELINE
    if GLOBAL_PIPELINE is None:
        GLOBAL_PIPELINE = RAGPipeline(
            corpus_path=DEFAULT_CORPUS,
            index_path=DEFAULT_INDEX,
            chunks_path=DEFAULT_CHUNKS,
        )
    return GLOBAL_PIPELINE


def create_new_session():
    global CURRENT_SESSION_ID
    pipeline = get_pipeline()
    CURRENT_SESSION_ID = pipeline.create_session()
    return CURRENT_SESSION_ID


def get_session_id():
    global CURRENT_SESSION_ID
    if CURRENT_SESSION_ID is None:
        create_new_session()
    return CURRENT_SESSION_ID

def get_available_sessions() -> List[Tuple[str, str]]:
    memory_dir = os.path.join(ROOT_DIR, "data", "session_memory")
    sessions = []
    if not os.path.exists(memory_dir):
        return sessions
    for f in os.listdir(memory_dir):
        if f.endswith('.json'):
            filepath = os.path.join(memory_dir, f)
            try:
                with open(filepath, 'r') as fp:
                    data = json.load(fp)
                sid = data.get("session_id", f.replace(".json", ""))
                created_at = data.get("created_at", "")
                
                # find first user query
                first_query = None
                for m in data.get("conversation_history", []):
                    if m.get("role") == "user":
                        first_query = m.get("content", "").strip()
                        break
                
                if first_query:
                    # Truncate to make a clean topic heading
                    display_name = (first_query[:45] + "...") if len(first_query) > 45 else first_query
                    sessions.append((created_at, display_name, sid))
            except Exception:
                pass
                
    # Sort by created_at descending
    sessions.sort(key=lambda x: x[0], reverse=True)
    return [(name, sid) for _, name, sid in sessions]

def load_session(session_id: str):
    global CURRENT_SESSION_ID
    CURRENT_SESSION_ID = session_id
    memory_file = os.path.join(ROOT_DIR, "data", "session_memory", f"{session_id}.json")
    chat_history = []
    if os.path.exists(memory_file):
        try:
            with open(memory_file, "r") as f:
                data = json.load(f)
            for msg in data.get("conversation_history", []):
                chat_history.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        except Exception:
            pass
            
    return chat_history


def ensure_pipeline_ready() -> Tuple[bool, str]:
    if not os.path.exists(DEFAULT_INDEX) or not os.path.exists(DEFAULT_CHUNKS):
        return False, "Index not found. Click 'Build Index' first (requires OPENAI_API_KEY)."
    return True, "Ready"


def build_index_ui() -> dict:
    try:
        pipeline = get_pipeline()
        total = pipeline.build()
        msg_out = f"Index built successfully. Total chunks: {total}"
    except Exception as exc:
        msg_out = f"Build failed: {exc}"
    return gr.update(value=[{"role": "assistant", "content": msg_out}])


def handle_pdf_upload(file_objs):
    if not file_objs:
        return gr.update(value=[{"role": "assistant", "content": "No files uploaded."}])
    if not isinstance(file_objs, list):
        file_objs = [file_objs]
    os.makedirs(PDF_DIR, exist_ok=True)
    filenames = []
    for file_obj in file_objs:
        filename = os.path.basename(file_obj.name)
        destination = os.path.join(PDF_DIR, filename)
        shutil.copy(file_obj.name, destination)
        filenames.append(filename)
    try:
        extract_from_pdfs(PDF_DIR, DEFAULT_CORPUS, SAMPLE_JSON, REPORT_MD)
    except Exception as exc:
        return gr.update(value=[{"role": "assistant", "content": f"Extraction failed: {exc}"}])
    chat_update = build_index_ui()
    filenames_str = ", ".join(filenames)
    new_content = (
        f"Successfully uploaded and extracted: **{filenames_str}**\n\n"
        f"{chat_update['value'][0]['content']}"
    )
    chat_update["value"][0]["content"] = new_content
    return chat_update


def get_pdf_page_image(filename: str, page_num: int, text_to_highlight: str = ""):
    pdf_path = os.path.join(PDF_DIR, filename)
    if not os.path.exists(pdf_path):
        return None
    try:
        doc = pymupdf.open(pdf_path)
        page = doc[page_num - 1]
        if text_to_highlight:
            words = text_to_highlight.split()
            phrases = [" ".join(words[i:i + 8]) for i in range(0, len(words), 8)]
            for phrase in phrases:
                if len(phrase) > 4:
                    text_instances = page.search_for(phrase)
                    for inst in text_instances:
                        page.add_highlight_annot(inst)
        pix = page.get_pixmap(dpi=150, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    except Exception as e:
        print(f"Error getting page image: {e}")
        return None


def get_cost_markdown():
    return f"**API Cost Used:** ${get_total_cost():.4f} / $5.00"


# ──────────────────────────────────────────────────────
# Format the JSON answer for display
# ──────────────────────────────────────────────────────

def format_answer(raw_json: str) -> Tuple[str, bool]:
    """Returns (formatted_text, is_out_of_scope)."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return raw_json, False

    answer_type = data.get("type", "relevant")
    content = data.get("content", "")
    citations = data.get("citations", [])

    if answer_type == "out-of-scope":
        return "**OUT_OF_CONTEXT**", True

    type_labels = {
        "relevant": "📄 Answer",
        "cross-reference": "🔗 Cross-Reference",
        "comparison": "⚖️ Comparison",
        "negation": "🚫 Negation",
        "yes-no": "✅ Yes/No",
        "ambiguous": "❓ Clarification Needed",
        "multi-part": "📋 Multi-Part Answer",
        "meta": "💬 From Conversation",
    }
    badge = type_labels.get(answer_type, "📄 Answer")

    parts = [f"**{badge}**\n", content]
    if citations:
        parts.append("\n\n---\n**Sources:**")
        for cite in citations:
            cid = cite.get("id", "?")
            source = cite.get("source", "unknown")
            page = cite.get("page", "?")
            parts.append(f"- [{cid}] {source}, page {page}")
    return "\n".join(parts), False


# ──────────────────────────────────────────────────────
# Main respond
# ──────────────────────────────────────────────────────

def respond(message_data, chat_history):
    mode = "Fast"
    if isinstance(message_data, dict):
        text = message_data.get("text", "").strip()
        files = message_data.get("files", [])
    else:
        text = str(message_data).strip()
        files = []

    if files:
        chat_history.append({"role": "user", "content": f"Uploaded {len(files)} file(s)."})
        chat_history.append({"role": "assistant", "content": "Processing and building index..."})
        yield gr.update(value={"text": "", "files": []}), chat_history, gr.update(), gr.update(), get_cost_markdown(), gr.update()
        
        os.makedirs(PDF_DIR, exist_ok=True)
        filenames = []
        for f in files:
            path = f if isinstance(f, str) else getattr(f, 'path', getattr(f, 'name', str(f)))
            filename = os.path.basename(path)
            destination = os.path.join(PDF_DIR, filename)
            shutil.copy(path, destination)
            filenames.append(filename)
            
        try:
            extract_from_pdfs(PDF_DIR, DEFAULT_CORPUS, SAMPLE_JSON, REPORT_MD)
            pipeline = get_pipeline()
            total = pipeline.build()
            upload_status = f"Successfully uploaded and indexed: **{', '.join(filenames)}** (Total chunks: {total})."
            chat_history[-1]["content"] = upload_status
            yield gr.update(), chat_history, gr.update(), gr.update(), get_cost_markdown(), gr.update()
        except Exception as exc:
            chat_history[-1]["content"] = f"Extraction failed: {exc}"
            yield gr.update(), chat_history, gr.update(), gr.update(), get_cost_markdown(), gr.update()
            if not text:
                return

    if not text:
        return

    message = text
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": "Evaluating request..."})
    yield gr.update(value={"text": "", "files": []}), chat_history, gr.update(), gr.update(), get_cost_markdown(), gr.update()

    ready, msg = ensure_pipeline_ready()
    if not ready:
        chat_history[-1]["content"] = msg
        yield "", chat_history, gr.update(value=None), gr.update(visible=False), get_cost_markdown(), gr.update()
        return

    try:
        pipeline = get_pipeline()
        session_id = get_session_id()
        pipeline.load_if_needed()
        ag = pipeline.answer_generator

        # ── 1. Classify intent ────────────────────────────
        intent = ag.classify_intent(message, session_id)

        # ── 2. For document queries: expand & retrieve ────
        results = []
        pdf_images = []
        if intent == "document":
            chat_history[-1]["content"] = "Analyzing intent & formatting query..."
            yield gr.update(value={"text": "", "files": []}), chat_history, gr.update(), gr.update(), get_cost_markdown(), gr.update()
            
            if mode == "Fast":
                try:
                    # Still use the LLM to format and resolve context/pronouns,
                    # but only use the primary variant to save time and API costs.
                    query_variants = [ag.expand_query(message, session_id)[0]]
                except Exception:
                    query_variants = [message]
            else:
                query_variants = ag.expand_query(message, session_id)

            chat_history[-1]["content"] = "Embedding & searching knowledge base..."
            yield gr.update(value={"text": "", "files": []}), chat_history, gr.update(), gr.update(), get_cost_markdown(), gr.update()

            seen, merged = set(), []
            for variant in query_variants:
                hits = pipeline.retriever.retrieve(
                    query=variant,
                    index=pipeline.loaded_index,
                    chunks=pipeline.loaded_chunks,
                    method="vector",
                    k=10,
                )
                for chunk in hits:
                    key = (chunk["source"], chunk["page"], chunk["text"][:80])
                    if key not in seen:
                        seen.add(key)
                        merged.append(chunk)
            merged.sort(key=lambda c: c.get("score", 0), reverse=True)
            results = merged[:10]

            for chunk in results:
                if chunk.get("score", 0) > 0.0:
                    img = get_pdf_page_image(chunk["source"], chunk["page"], text_to_highlight=chunk["text"])
                    if img:
                        pdf_images.append(img)

        chat_history[-1]["content"] = "Generating final response..."
        yield gr.update(value={"text": "", "files": []}), chat_history, gr.update(), gr.update(), get_cost_markdown(), gr.update()

        # ── 3. Generate answer ────────────────────────────
        raw_json = ag.generate(
            query=message,
            retrieved_chunks=results,
            session_id=session_id,
            intent=intent,
            mode=mode,
        )

        # ── 4. Format & display ──────────────────────────
        formatted, is_out_of_scope = format_answer(raw_json)

        chat_history[-1]["content"] = formatted

        if is_out_of_scope or not pdf_images:
            yield "", chat_history, gr.update(value=None), gr.update(visible=False), get_cost_markdown(), gr.update(choices=get_available_sessions(), value=session_id)
        else:
            yield "", chat_history, gr.update(value=pdf_images), gr.update(visible=True), get_cost_markdown(), gr.update(choices=get_available_sessions(), value=session_id)

    except Exception as exc:
        chat_history[-1]["content"] = f"Error: {exc}"
        yield "", chat_history, gr.update(value=None), gr.update(visible=False), get_cost_markdown(), gr.update()


# ──────────────────────────────────────────────────────
# Agent Workflow Setup
# ──────────────────────────────────────────────────────
AGENT_WIP_STATE = {}

def agent_generate_draft(query):
    if not query.strip():
        return "Please enter a query.", ""
    
    # Run researcher
    yield "Researching internal facts from the database...", ""
    research_notes = researcher_agent.chat([{"role": "user", "content": query}])
    
    # Run fact checker
    yield f"Internal search complete. Fact-checking externally...\n\n{research_notes[:150]}...", ""
    verification_notes = factchecker_agent.chat([{"role": "user", "content": f"Verify this data via live internet search and optionally update DB:\n\n{research_notes}"}])
    
    # Run synthesizer
    yield f"Fact Check Complete.\n\n{verification_notes[:150]}...\n\nSynthesizing final response...", ""
    draft = synthesizer_agent.chat([{"role": "user", "content": f"Please synthesize an official answer based on:\n\nINTERNAL:\n{research_notes}\n\nEXTERNAL VERIFICATION:\n{verification_notes}\n\nIMPORTANT: Just output the draft text alone. Do NOT use the `create_markdown_report` tool."}])
    
    AGENT_WIP_STATE["current_draft"] = draft
    yield "Draft Complete. Awaiting human review.", draft

def agent_process_feedback(feedback, current_draft):
    if not feedback.strip():
        yield "Please provide feedback instructions.", current_draft, gr.update()
        return
    
    yield "Processing human feedback and rewriting draft...", current_draft, gr.update(value="")
    
    prompt = f"The human operator provided this feedback on your draft:\n{feedback}\n\nPlease output the revised draft only. Do not use the `create_markdown_report` tool right now. Original draft:\n{current_draft}"
    revised_draft = synthesizer_agent.chat([{"role": "user", "content": prompt}])
    
    AGENT_WIP_STATE["current_draft"] = revised_draft
    yield "Revision Complete.", revised_draft, gr.update()

def agent_approve_draft(current_draft):
    from mcp_server import create_markdown_report
    import re
    
    if not current_draft.strip():
        return gr.update(), "No draft found to approve.", gr.update(), gr.update(), gr.update(), gr.update()
        
    match = re.search(r'^#\s+(.+)', current_draft, re.MULTILINE)
    if match:
        base_name = match.group(1).strip()
        filename = re.sub(r'[^a-zA-Z0-9_\-]', '', base_name.replace(' ', '_')) + ".md"
    else:
        filename = "approved_report.md"
        
    res = create_markdown_report(current_draft, filename)
    status_msg = f"✅ Draft approved and saved as '{filename}'! Ready for next task."
    
    return (
        "",                            # agent_query_input
        status_msg,                    # agent_status_txt
        "",                            # agent_draft_box
        gr.update(value="", visible=False),  # agent_preview_box
        "",                            # agent_feedback_input
        False                          # preview_state
    )


# ──────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────

with gr.Blocks(title="DocuMind") as demo:
    with gr.Row(equal_height=True):
        with gr.Column(scale=4):
            gr.HTML(get_header_html())
        with gr.Column(scale=1, min_width=100):
            cost_display = gr.Markdown(get_cost_markdown())

    with gr.Tabs():
        # TAB 1: Standard Chat UI
        with gr.Tab("Interactive RAG (Standard)"):
            with gr.Row(equal_height=True):
                new_session_btn = gr.Button("New Session", variant="secondary")
                session_dropdown = gr.Dropdown(
                    label="Session",
                    choices=get_available_sessions(),
                    value=CURRENT_SESSION_ID,
                    interactive=True,
                    allow_custom_value=True,
                    scale=3
                )
                clear_session_btn = gr.Button("Clear Session", variant="secondary")
                build_btn = gr.Button("Build Corpus Index", variant="primary")
            
            with gr.Row():
                with gr.Column(scale=1):
                    chatbot = gr.Chatbot()
                    
                    with gr.Row():
                        msg = gr.MultimodalTextbox(
                            show_label=False, 
                            placeholder="Enter message or upload file...", 
                            file_types=[".pdf"],
                            container=False, 
                            scale=1
                        )
                    clear = gr.ClearButton([msg, chatbot])
                with gr.Column(scale=1, visible=False) as pdf_col:
                    pdf_viewer = gr.Gallery(label="Highlighted Source Pages", object_fit="contain", interactive=False)

            def load_selected_session(session_id):
                if not session_id:
                    return [], None, gr.update(visible=False)
                hist = load_session(session_id)
                return hist, None, gr.update(visible=False)

            def new_session():
                session_id = create_new_session()
                # Update choices
                choices = get_available_sessions()
                return [], "", gr.update(choices=choices, value=session_id), None, gr.update(visible=False)

            def clear_session_memory():
                sid = get_session_id()
                memory_file = os.path.join(ROOT_DIR, "data", "session_memory", f"{sid}.json")
                if os.path.exists(memory_file):
                    try:
                        with open(memory_file, "r") as f:
                            data = json.load(f)
                        data["conversation_history"] = []
                        with open(memory_file, "w") as f:
                            json.dump(data, f, indent=2)
                    except Exception:
                        pass
                
                # Update choices
                choices = get_available_sessions()
                return [], "", gr.update(choices=choices, value=sid), None, gr.update(visible=False)

            def handle_vote(data: gr.LikeData):
                if data.liked:
                    pass
                else:
                    pass

            chatbot.like(handle_vote, None, None)

            msg.submit(respond, [msg, chatbot], [msg, chatbot, pdf_viewer, pdf_col, cost_display, session_dropdown])
            clear.click(lambda: (None, gr.update(visible=False)), inputs=None, outputs=[pdf_viewer, pdf_col])
            new_session_btn.click(new_session, outputs=[chatbot, msg, session_dropdown, pdf_viewer, pdf_col])
            session_dropdown.change(load_selected_session, inputs=[session_dropdown], outputs=[chatbot, pdf_viewer, pdf_col])
            clear_session_btn.click(clear_session_memory, outputs=[chatbot, msg, session_dropdown, pdf_viewer, pdf_col])
            build_btn.click(fn=build_index_ui, outputs=chatbot)
            
            def init_ui():
                sid = get_session_id()
                choices = get_available_sessions()
                # If there are no sessions, or the current one isn't in choices, create it in memory logic
                # So we just update dropdown with whatever is available
                hist = load_session(sid)
                return sid, hist, gr.update(choices=choices, value=sid)

            demo.load(init_ui, outputs=[gr.State(), chatbot, session_dropdown])

        # TAB 2: Multi-Agent Human-in-the-Loop Workflow
        with gr.Tab("Multi-Agent Analysis (Human in Loop)"):
            with gr.Row(elem_classes="top-title-row"):
                with gr.Column(scale=8):
                    gr.Markdown("### AI Team (Researcher + Writer)")
                    gr.Markdown("Directly query the AI team. The Researcher will scour the vector database, and the Writer will prepare a draft for your approval. You can provide feedback and ask it to rewrite before finalizing.")
            
            with gr.Row():
                with gr.Column(scale=2):
                    agent_query_input = gr.Textbox(label="Your Request", lines=2, placeholder="e.g. Find all documents mentioning 'revenue' and summarize the key findings.")
                    agent_run_btn = gr.Button("Assign Task to AI Team", variant="primary")
                    agent_status_txt = gr.Textbox(label="System Status", interactive=False, value="Awaiting Task...")
                    
                    with gr.Group():
                        gr.Markdown("**Review & Edit**")
                        agent_feedback_input = gr.Textbox(label="Feedback for Writer (If rejecting/editing)", lines=2, placeholder="e.g. Make it more professional and use bullet points.")
                        
                        with gr.Row():
                            agent_preview_toggle_btn = gr.Button("Toggle Markdown Preview")
                            agent_revise_btn = gr.Button("Revise Draft (Send Feedback)")
                            agent_approve_btn = gr.Button("Approve Draft (Finalize)", variant="primary")
                            agent_clear_btn = gr.Button("Clear Form", variant="stop")
                
                with gr.Column(scale=3):
                    agent_draft_box = gr.Textbox(label="Writer Agent Draft Output", lines=25, interactive=False)
                    agent_preview_box = gr.Markdown(visible=False)

            def toggle_preview(draft_text, is_previewing):
                if is_previewing:
                    return gr.update(visible=True), gr.update(visible=False), False
                else:
                    return gr.update(visible=False), gr.update(visible=True, value=draft_text), True

            preview_state = gr.State(False)
            agent_preview_toggle_btn.click(
                toggle_preview,
                inputs=[agent_draft_box, preview_state],
                outputs=[agent_draft_box, agent_preview_box, preview_state],
                show_progress="hidden",
                api_name=False
            )

            # Wire up agent buttons
            agent_run_btn.click(
                agent_generate_draft,
                inputs=[agent_query_input],
                outputs=[agent_status_txt, agent_draft_box]
            ).then(
                fn=lambda text, is_prev: gr.update(value=text) if is_prev else gr.update(),
                inputs=[agent_draft_box, preview_state],
                outputs=[agent_preview_box]
            )
            
            agent_revise_btn.click(
                agent_process_feedback,
                inputs=[agent_feedback_input, agent_draft_box],
                outputs=[agent_status_txt, agent_draft_box, agent_feedback_input]
            ).then(
                fn=lambda text, is_prev: gr.update(value=text) if is_prev else gr.update(),
                inputs=[agent_draft_box, preview_state],
                outputs=[agent_preview_box]
            )
            
            agent_approve_btn.click(
                agent_approve_draft,
                inputs=[agent_draft_box],
                outputs=[
                    agent_query_input,
                    agent_status_txt,
                    agent_draft_box,
                    agent_preview_box,
                    agent_feedback_input,
                    preview_state
                ]
            )

            agent_clear_btn.click(
                lambda: (
                    gr.update(value=""),
                    gr.update(value="Awaiting Task..."),
                    gr.update(value=""),
                    gr.update(value="", visible=False),
                    gr.update(value=""),
                    False
                ),
                inputs=None,
                outputs=[
                    agent_query_input,
                    agent_status_txt,
                    agent_draft_box,
                    agent_preview_box,
                    agent_feedback_input,
                    preview_state
                ]
            )

if __name__ == "__main__":
    demo.launch()