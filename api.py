import sys
import os
import uuid
import shutil
import socket

# Force UTF-8 output on Windows (cp1252 by default — breaks non-ASCII chars)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import gradio as gr

from backend.config import config
from backend.document_processor import process_document
from backend.embedding_store import embedding_store
from backend.rag_engine import ask_question, _load_model
from backend.extractor import extract_shipment_data

# Pre-load the model at startup so it's ready before the UI opens.
print("Loading model - this may take a moment on first run (downloads ~2-3 GB)...")
_load_model()
print("Model ready.")

# ---------------------------------------------------------------------------
#  Gradio UI Setup
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
.gradio-container { max-width: 1000px !important; margin: auto; padding-top: 20px;}
.main-title { text-align: center; font-weight: 800; font-size: 2.5em; margin-bottom: 0.2em; color: #1e293b; letter-spacing: -0.02em;}
.sub-title { text-align: center; color: #64748b; font-weight: 500; font-size: 1.1em; margin-bottom: 2.5em; }
.fixed-height-file .wrap { min-height: 120px !important; max-height: 120px !important; }
"""

def upload_and_process(file):
    if not file:
        return "Error: Please upload a document first."

    os.makedirs(config.upload_dir, exist_ok=True)
    doc_id = str(uuid.uuid4())[:8]

    # Handle Gradio file object
    if isinstance(file, str):
        src = file
    else:
        src = file.name if hasattr(file, 'name') else str(file)
        
    filename = os.path.basename(src)
    save_path = os.path.join(config.upload_dir, f"{doc_id}_{filename}")
    shutil.copy2(src, save_path)

    try:
        chunks = process_document(save_path)
    except Exception as e:
        return f"Error - Parse failed: {str(e)}"

    try:
        num_chunks = embedding_store.add_document(doc_id, chunks)
    except Exception as e:
        return f"Error - Indexing failed: {str(e)}"

    # Store state cleanly using function attributes
    upload_and_process._doc_id = doc_id
    upload_and_process._file_path = save_path

    return f"Success: Indexed {filename}! ({num_chunks} chunks)"

# Initialize UI state
upload_and_process._doc_id = ""
upload_and_process._file_path = ""


def ask_bot(question):
    doc_id = upload_and_process._doc_id
    if not doc_id:
        return "Error: Please upload and index a document first."

    if not question or not question.strip():
        return "Error: Please enter a question."

    try:
        result = ask_question(doc_id=doc_id, question=question)
    except Exception as e:
        return f"Error: {str(e)}"

    answer = result["answer"]
    confidence = result["confidence"]
    guardrail = result["guardrail"]
    conf_score = confidence["confidence_score"]

    if conf_score >= 0.7:
        lvl = "(High Confidence)"
    elif conf_score >= 0.4:
        lvl = "(Medium Confidence)"
    else:
        lvl = "(Low Confidence)"

    if not guardrail["passed"]:
        return f"Guardrail Blocked:\n{answer}\n\nConfidence: {conf_score:.1%}"

    return f"Answer:\n{answer}\n\nConfidence: {conf_score:.1%} {lvl}"


def extract_data():
    file_path = upload_and_process._file_path
    if not file_path:
        return {"error": "Please upload and index a document first."}

    try:
        result = extract_shipment_data(file_path=file_path)
        if "_error" in result and result["_error"]:
            return {"error": result["_error"]}
        return result
    except Exception as e:
        return {"error": str(e)}


CUSTOM_JS = """
<script>
setInterval(function() {
    var elements = document.querySelectorAll('*');
    for (var i=0; i<elements.length; i++) {
        var el = elements[i];
        if (el.className && typeof el.className === 'string' && (el.className.includes('progress') || el.className.includes('eta'))) {
            if (el.innerText && el.innerText.match(/\\/[\\s]*[0-9.]+[s]/)) {
                el.innerText = el.innerText.replace(/\\/[\\s]*[0-9.]+[s]/, 's');
            }
        }
    }
}, 50);
</script>
"""

with gr.Blocks(title="Ultra Doc") as ui:
    gr.HTML("<div class='main-title'>Ultra Doc-Intelligence</div>")
    gr.HTML(f"<div class='sub-title'>Gemma 4 E2B • {config.device.upper()} • Offline RAG & Extraction</div>")
    
    with gr.Row():
        # First Column: Document Upload, Indexing, and Q&A Chat
        with gr.Column(scale=1, min_width=250):
            file_input = gr.File(label="Upload Document", type="filepath", file_count="single", elem_classes=["fixed-height-file"])
            index_btn = gr.Button("Index Document", variant="primary")
            status_text = gr.Textbox(label="Status", interactive=False, lines=2, max_lines=2)
            
            gr.Markdown("<hr>")
            
            query_input = gr.Textbox(label="Ask a Question", placeholder="e.g., What is the carrier rate?", lines=2, max_lines=2)
            ask_btn = gr.Button("Get Answer", variant="primary")
            answer_text = gr.Textbox(label="Chatbot Response", interactive=False, lines=8, max_lines=8)
            
        # Second Column: Structured Data Extraction
        with gr.Column(scale=1, min_width=250):
            extract_btn = gr.Button("Extract Shipment JSON", variant="secondary")
            extract_out = gr.JSON(label="Extracted Data")

    # Wiring events
    index_btn.click(fn=upload_and_process, inputs=[file_input], outputs=[status_text])
    
    ask_btn.click(fn=ask_bot, inputs=[query_input], outputs=[answer_text])
    query_input.submit(fn=ask_bot, inputs=[query_input], outputs=[answer_text])
    
    extract_btn.click(fn=extract_data, inputs=[], outputs=[extract_out])

if __name__ == "__main__":
    # Get local network IP for easier mobile/other PC access
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        wifi_ip = s.getsockname()[0]
        s.close()
    except:
        wifi_ip = "127.0.0.1"
        
    print("\n" + "="*60)
    print("  ULTRA DOC-INTELLIGENCE UI")
    print(f"  Local Access:   http://127.0.0.1:7860")
    print(f"  Wi-Fi Access:   http://{wifi_ip}:7860")
    print("="*60 + "\n")

    # Launching server
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        head=CUSTOM_JS,
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
        css=CUSTOM_CSS
    )