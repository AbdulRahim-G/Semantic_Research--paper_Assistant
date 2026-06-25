# frontend/app.py
import streamlit as st
import requests
import os
import json
import logging
from typing import List, Dict, Any
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Backend service address configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_PREFIX = f"{BACKEND_URL}/api/v1"

# Page configurations
st.set_page_config(
    page_title="Semantic Research Paper Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS stylesheet
def load_css(file_path: str):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("assets/custom.css")

# --- Session State Initializer ---
if "session_id" not in st.session_state:
    try:
        # Create a new chat session on the backend
        res = requests.post(f"{API_PREFIX}/chat/session", timeout=5)
        if res.status_code == 200:
            st.session_state.session_id = res.json()["id"]
        else:
            st.session_state.session_id = "temp-fallback-session-id"
    except Exception as e:
        logger.warning(f"Could not connect to backend to create session: {e}")
        st.session_state.session_id = "temp-fallback-session-id"

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- API Helper Functions ---
def get_papers() -> List[Dict[str, Any]]:
    try:
        res = requests.get(f"{API_PREFIX}/papers/", timeout=10)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        logger.error(f"Error fetching papers: {e}")
    return []

def upload_paper_api(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    try:
        files = {"file": (filename, file_bytes, "application/pdf")}
        res = requests.post(f"{API_PREFIX}/papers/upload", files=files, timeout=30)
        return res.json() if res.status_code == 200 else {"error": res.text}
    except Exception as e:
        return {"error": str(e)}

def delete_paper_api(paper_id: str) -> bool:
    try:
        res = requests.delete(f"{API_PREFIX}/papers/{paper_id}", timeout=10)
        return res.status_code == 200
    except Exception as e:
        logger.error(f"Error deleting paper: {e}")
        return False

def query_rag_api(question: str, paper_ids: List[str] = None) -> Dict[str, Any]:
    payload = {
        "session_id": st.session_state.session_id,
        "question": question,
        "paper_ids": paper_ids if paper_ids else None
    }
    try:
        res = requests.post(f"{API_PREFIX}/chat/query", json=payload, timeout=60)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        logger.error(f"Query error: {e}")
    return {"content": "Failed to receive response from backend API.", "citations": []}

def get_summary_api(paper_id: str) -> Dict[str, Any]:
    try:
        res = requests.post(f"{API_PREFIX}/papers/{paper_id}/summarize", timeout=60)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        logger.error(f"Summary query failed: {e}")
    return {}

def get_citations_api(paper_id: str) -> Dict[str, str]:
    try:
        res = requests.get(f"{API_PREFIX}/papers/{paper_id}/citation", timeout=10)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        logger.error(f"Citations query failed: {e}")
    return {"apa": "", "mla": "", "ieee": ""}

def get_graph_api(paper_ids: List[str] = None) -> Dict[str, Any]:
    try:
        params = {}
        if paper_ids:
            params["paper_ids"] = paper_ids
        res = requests.get(f"{API_PREFIX}/graph/", params=params, timeout=15)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        logger.error(f"Graph retrieval failed: {e}")
    return {"nodes": [], "edges": []}

def compare_papers_api(paper_ids: List[str]) -> Dict[str, Any]:
    try:
        res = requests.post(f"{API_PREFIX}/graph/compare", json={"paper_ids": paper_ids}, timeout=60)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        logger.error(f"Comparison query failed: {e}")
    return {}

# --- Streamlit Layout ---
st.markdown("<h1 style='text-align: center; color: #6366f1;'>🎓 Semantic Research Paper Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94a3b8; margin-bottom: 30px;'>An Intelligent RAG-Based System for Research Understanding, Graph Exploration & Comparison</p>", unsafe_allow_html=True)

# Main Workspace Tabs
tab_chat, tab_upload, tab_summary, tab_compare = st.tabs([
    "💬 Interactive Q&A", 
    "📁 Upload & Ingest", 
    "📋 Summaries", 
    "📊 Comparative Matrix"
])

# Load current papers list
papers = get_papers()

# Sidebar: Display current active library and global session ID
with st.sidebar:
    st.image("https://img.icons8.com/color/144/null/university.png", width=80)
    st.markdown("### 📚 Active Library")
    if not papers:
        st.info("No papers uploaded yet.")
    else:
        st.write(f"Total Papers: {len(papers)}")
        for p in papers:
            status_emoji = "🟢" if p["status"] == "completed" else ("🟡" if p["status"] == "processing" else "🔴")
            st.markdown(f"{status_emoji} **{p['title'][:40]}**")
        
        st.write("")
        if st.button("🔄 Refresh Library", key="refresh_sidebar_btn", use_container_width=True):
            st.rerun()
            
    st.divider()
    st.caption(f"Session: `{st.session_state.session_id}`")

# --- Tab 1: Interactive Chat Q&A ---
with tab_chat:
    st.header("Ask Questions About Your Papers")
    st.caption("Answers are strictly grounded in the retrieved segments of selected research papers.")
    
    # Filter selection
    selected_paper_titles = st.multiselect(
        "Focus search on specific papers (leave empty to search entire library):",
        options=[p["title"] for p in papers if p["status"] == "completed"]
    )
    
    target_ids = []
    if selected_paper_titles:
        target_ids = [p["id"] for p in papers if p["title"] in selected_paper_titles]

    # Chat message container
    chat_container = st.container(height=500)
    
    # Load and display history from backend/session state
    with chat_container:
        for msg in st.session_state.messages:
            role_class = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-assistant"
            st.markdown(f"<div class='{role_class}'>{msg['content']}</div>", unsafe_allow_html=True)
            if msg.get("citations"):
                with st.expander("🔍 Show Page Citations"):
                    for cite in msg["citations"]:
                        st.markdown(
                            f"<div class='citation-box'>"
                            f"<b>[Source #{cite['source_index']}] Page {cite['page_number']}</b> - "
                            f"<i>{cite['authors']} ({cite['year']})</i>. {cite['title']}<br/>"
                            f"<code>Snippet: {cite['text_snippet'][:180]}...</code>"
                            f"</div>", 
                            unsafe_allow_html=True
                        )

    # Chat Input
    if prompt := st.chat_input("Enter your question (e.g., 'What are the main results in the Transformer paper?')"):
        # Display user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            st.markdown(f"<div class='chat-bubble-user'>{prompt}</div>", unsafe_allow_html=True)
            
        with st.spinner("Retrieving context and generating answer..."):
            result = query_rag_api(prompt, paper_ids=target_ids)
            
        # Parse response
        answer = result.get("content") or result.get("answer") or "Could not extract answer from response."
        citations = result.get("citations", [])
        
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "citations": citations
        })
        
        with chat_container:
            st.markdown(f"<div class='chat-bubble-assistant'>{answer}</div>", unsafe_allow_html=True)
            if citations:
                with st.expander("🔍 Show Page Citations"):
                    for cite in citations:
                        st.markdown(
                            f"<div class='citation-box'>"
                            f"<b>[Source #{cite['source_index']}] Page {cite['page_number']}</b> - "
                            f"<i>{cite['authors']} ({cite['year']})</i>. {cite['title']}<br/>"
                            f"<code>Snippet: {cite['text_snippet'][:180]}...</code>"
                            f"</div>", 
                            unsafe_allow_html=True
                        )

# --- Tab 2: Upload & Management ---
with tab_upload:
    st.header("Upload Scientific Papers")
    st.write("Drag and drop single or multiple PDF papers. They will be processed and indexed asynchronously.")

    uploaded_files = st.file_uploader("Choose PDF Files", type=["pdf"], accept_multiple_files=True)
    
    if st.button("🚀 Process & Ingest Papers"):
        if uploaded_files:
            success_count = 0
            has_error = False
            for file in uploaded_files:
                # Reset file reader pointer to beginning in case it was already read
                file.seek(0)
                with st.spinner(f"Uploading {file.name}..."):
                    res = upload_paper_api(file.read(), file.name)
                    if "error" in res:
                        st.error(f"Failed to upload {file.name}: {res['error']}")
                        has_error = True
                    else:
                        st.success(f"Successfully uploaded {file.name}! Processing scheduled.")
                        success_count += 1
            if success_count > 0:
                import time
                time.sleep(3.0 if has_error else 1.5)
                st.rerun()
        else:
            st.warning("Please select at least one file to upload.")

    st.divider()
    col_sub, col_ref = st.columns([6, 3])
    with col_sub:
        st.subheader("Manage Current Library")
    with col_ref:
        if st.button("🔄 Refresh Status", key="refresh_status_btn"):
            st.rerun()
    
    if not papers:
        st.info("No papers indexed yet.")
    else:
        # Build list interface
        for p in papers:
            col_name, col_status, col_action = st.columns([5, 2, 2])
            with col_name:
                st.markdown(f"📄 **{p['title']}**")
                st.caption(f"Authors: {', '.join(p['authors']) if p['authors'] else 'Unknown'} | Year: {p['publication_year'] or 'N/A'}")
            with col_status:
                if p["status"] == "completed":
                    st.success("🟢 Ready")
                elif p["status"] == "processing":
                    st.info("🟡 Ingesting...")
                else:
                    st.error("🔴 Failed")
            with col_action:
                if st.button("🗑️ Delete", key=f"del_{p['id']}"):
                    if delete_paper_api(p["id"]):
                        st.success("Deleted successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to delete.")

# --- Tab 3: Summaries & Citation Generator ---
with tab_summary:
    st.header("Structured summarizations & Bibliographies")
    st.caption("Generate multi-section analytical summaries and retrieve formatted citations (APA, MLA, IEEE).")

    ready_papers = [p for p in papers if p["status"] == "completed"]
    if not ready_papers:
        st.warning("Please upload and successfully process a paper first.")
    else:
        selected_paper = st.selectbox(
            "Select paper to summarize:",
            options=ready_papers,
            format_func=lambda x: f"{x['title']} ({x['publication_year'] or 'N/A'})"
        )
        
        if selected_paper:
            if st.button("📝 Generate Detailed Summary", key=f"sum_btn_{selected_paper['id']}"):
                with st.spinner("Extracting contents and summarizing sections..."):
                    summary = get_summary_api(selected_paper["id"])
                    if summary:
                        st.subheader("Abstract Summary")
                        st.write(summary.get("abstract", "N/A"))
                        
                        st.subheader("Detailed Summary")
                        st.write(summary.get("detailed_summary", "N/A"))
                        
                        st.subheader("Key Contributions")
                        for cont in summary.get("key_contributions", []):
                            st.markdown(f"- {cont}")
                            
                        st.subheader("Methodology Summary")
                        st.write(summary.get("methodology", "N/A"))
                        
                        st.subheader("Limitations")
                        for lim in summary.get("limitations", []):
                            st.markdown(f"- {lim}")
                            
                        st.subheader("Future Work")
                        for fw in summary.get("future_work", []):
                            st.markdown(f"- {fw}")
                    else:
                        st.error("Could not generate summary.")

# --- Tab 4: Multi-Paper Comparison Matrix ---
with tab_compare:
    st.header("Multi-Paper Comparison Panel")
    st.caption("Select two or more papers to generate a side-by-side comparative analysis of datasets, models, methodologies, and outcomes.")

    ready_papers = [p for p in papers if p["status"] == "completed"]
    if len(ready_papers) < 2:
        st.warning("Please upload and index at least 2 papers to access comparative analytics.")
    else:
        selected_titles = st.multiselect(
            "Select papers to compare:",
            options=[p["title"] for p in ready_papers]
        )
        
        if len(selected_titles) >= 2:
            compare_ids = [p["id"] for p in ready_papers if p["title"] in selected_titles]
            
            if st.button("📊 Run Comparison Matrix"):
                with st.spinner("Analyzing papers side-by-side..."):
                    matrix = compare_papers_api(compare_ids)
                    if matrix:
                        st.subheader("Comparison Matrix Results")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info("💡 **Methodology & Architecture**")
                            st.write(matrix.get("methodologies", "N/A"))
                            
                            st.success("⚙️ **Models & Parameters**")
                            st.write(matrix.get("models", "N/A"))
                            
                        with col2:
                            st.warning("📊 **Datasets & Splits**")
                            st.write(matrix.get("datasets", "N/A"))
                            
                            st.error("📉 **Limitations & Biases**")
                            st.write(matrix.get("limitations", "N/A"))
                            
                        st.divider()
                        st.markdown("### 🏆 Benchmarks & Quantitative Outcomes")
                        st.success(matrix.get("results", "N/A"))
                    else:
                        st.error("Failed to generate comparative matrix.")
        else:
            st.info("Select at least 2 papers above to begin.")

# --- Tab 5: Research Knowledge Graph Removed ---
