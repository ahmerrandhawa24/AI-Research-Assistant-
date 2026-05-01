#%%
# ============================================================
# app.py — AI Research Assistant
# Phase 9 — Advanced Features (Light Theme + Fixed FAISS)
# ============================================================

import streamlit as st
import sys
import os
import re
import fitz
import faiss
import pickle
import numpy as np
import tempfile
from datetime import datetime
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import importlib
import config
importlib.reload(config)

from config import (
    GROQ_API_KEY,
    LLM_MODEL,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
    PDF_FOLDER,
    VECTOR_DB_PATH,
    EXCLUDE_LAST_N_PAGES
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def load_groq_client():
    return Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def load_base_database():
    faiss_path    = os.path.join(VECTOR_DB_PATH, "index.faiss")
    metadata_path = os.path.join(VECTOR_DB_PATH, "metadata.pkl")
    if not os.path.exists(faiss_path):
        return None, []
    index = faiss.read_index(faiss_path)
    with open(metadata_path, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

embedder               = load_embedder()
client                 = load_groq_client()
base_index, base_chunks = load_base_database()

# ============================================================
# SESSION STATE
# ============================================================

if "chat_history"         not in st.session_state:
    st.session_state.chat_history         = []
if "total_questions"      not in st.session_state:
    st.session_state.total_questions      = 0
if "uploaded_chunks"      not in st.session_state:
    st.session_state.uploaded_chunks      = []
if "uploaded_index"       not in st.session_state:
    st.session_state.uploaded_index       = None
if "uploaded_file_name"   not in st.session_state:
    st.session_state.uploaded_file_name   = None
if "active_mode"          not in st.session_state:
    st.session_state.active_mode          = "knowledge_base"
if "doc_analysis_result"  not in st.session_state:
    st.session_state.doc_analysis_result  = None
if "analysis_type"        not in st.session_state:
    st.session_state.analysis_type        = None
if "upload_chat"          not in st.session_state:
    st.session_state.upload_chat          = []

# ============================================================
# CORE FUNCTIONS
# ============================================================

reference_keywords = [
    "et al.", "doi:", "http", "journal of",
    "proceedings of", "isbn", "publisher",
    "retrieved from", "vol.", "pp.", "ed.",
]

def clean_text(text):
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:()\-\'\"]+', ' ', text)
    return text.strip()

def extract_pdf_text_with_pages(pdf_bytes):
    doc        = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_data = []
    total      = len(doc)
    for page_num, page in enumerate(doc):
        if page_num >= total - EXCLUDE_LAST_N_PAGES:
            continue
        text    = page.get_text()
        cleaned = clean_text(text)
        if cleaned.strip():
            pages_data.append({
                "page_number" : page_num + 1,
                "text"        : cleaned
            })
    doc.close()
    return pages_data

def extract_full_text(pdf_bytes):
    doc  = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return clean_text(text)

def chunk_pages(pages_data, file_name):
    all_chunks = []
    chunk_id   = 0
    for page in pages_data:
        words = page["text"].split()
        start = 0
        while start < len(words):
            end         = start + CHUNK_SIZE
            chunk_words = words[start:end]
            chunk_text  = " ".join(chunk_words)
            all_chunks.append({
                "chunk_id"   : chunk_id,
                "chunk_text" : chunk_text,
                "file_name"  : file_name,
                "page_number": page["page_number"],
                "word_count" : len(chunk_words)
            })
            chunk_id += 1
            start    += CHUNK_SIZE - CHUNK_OVERLAP
    return all_chunks

def build_faiss_index(chunks):
    # Guard — no chunks
    if not chunks:
        return None

    texts = [c["chunk_text"] for c in chunks if c["chunk_text"].strip()]

    # Guard — empty texts
    if not texts:
        return None

    embeddings = embedder.encode(
        texts,
        show_progress_bar=False
    ).astype("float32")

    # Guard — bad shape
    if len(embeddings.shape) < 2:
        return None

    dim = embeddings.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(embeddings)
    return idx

def filter_chunks(candidates):
    filtered = []
    for chunk in candidates:
        text_lower   = chunk['chunk_text'].lower()
        kw_count     = sum(1 for kw in reference_keywords if kw in text_lower)
        cite_pattern = re.findall(r'\(\d{4}\)', chunk['chunk_text'])
        if not (kw_count >= 2 or len(cite_pattern) >= 4):
            filtered.append(chunk)
    return filtered

def search_index(query, idx, chunk_list, top_k=TOP_K_RESULTS):
    query_emb          = embedder.encode([query]).astype("float32")
    distances, indices = idx.search(query_emb, min(top_k * 3, len(chunk_list)))
    candidates = []
    for i, index_val in enumerate(indices[0]):
        chunk = chunk_list[index_val]
        candidates.append({
            "rank"        : i + 1,
            "score"       : round(float(distances[0][i]), 4),
            "chunk_id"    : chunk["chunk_id"],
            "file_name"   : chunk["file_name"],
            "page_number" : chunk["page_number"],
            "chunk_text"  : chunk["chunk_text"]
        })
    clean = filter_chunks(candidates)
    return clean[:top_k] if clean else candidates[:top_k]

def get_answer(question, history, idx, chunk_list):
    clean_chunks = search_index(question, idx, chunk_list)
    context      = ""
    for i, chunk in enumerate(clean_chunks):
        context += f"""
[Chunk {i+1}]
File    : {chunk['file_name']}
Page    : {chunk['page_number']}
Content : {chunk['chunk_text']}
---
"""
    history_text = ""
    if history:
        history_text = "Previous conversation:\n"
        for turn in history:
            history_text += f"User     : {turn['question']}\n"
            history_text += f"Assistant: {turn['answer'][:300]}\n"
            history_text += "---\n"

    prompt = f"""You are an expert AI Research Assistant.
You can answer questions on ANY topic based on the provided document context.

{history_text}

Context from documents:
{context}

STRICT RULES:
- Use ONLY information from the Content sections above
- Cite as: (File: filename, Page: X)
- If not in context say: "This topic is not covered in the provided documents"
- Do NOT make up facts or statistics

Question: {question}

Answer:"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024
    )
    return response.choices[0].message.content, clean_chunks

def analyze_document(full_text, analysis_type, file_name):
    prompts = {
        "abstract"   : f"""Extract or generate the abstract from this document.
If an abstract exists return it. If not write a concise abstract.
Keep it under 200 words.
Document: {full_text[:4000]}
Abstract:""",

        "conclusion" : f"""Extract the conclusion from this document.
If a conclusion section exists return it. If not summarize final findings.
Keep it under 300 words.
Document: {full_text[-3000:]}
Conclusion:""",

        "summary"    : f"""Write a comprehensive summary of this document.
Cover main topics, key findings, and important points.
Keep it under 400 words.
Document: {full_text[:6000]}
Summary:""",

        "methods"    : f"""Extract the methodology section from this document.
If methods section exists return it. If not describe how study was conducted.
Keep it under 400 words.
Document: {full_text[:6000]}
Methods:""",

        "key_findings": f"""Extract and list the key findings from this document.
Present as clear bullet points.
Focus on most important results and conclusions.
Document: {full_text[:6000]}
Key Findings:"""
    }

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompts[analysis_type]}],
        temperature=0.3,
        max_tokens=1024
    )
    return response.choices[0].message.content

# ============================================================
# CUSTOM CSS — Light Theme
# ============================================================

st.markdown("""
<style>
    .stApp { background-color: #f0f4f8; }

    /* Title box */
    .title-box {
        background: linear-gradient(135deg, #ffffff, #e8f0fe);
        border: 1px solid #c5d5f5;
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .title-box h1 { color: #1a237e; font-size: 26px; margin:0; font-weight:700; }
    .title-box p  { color: #5c6bc0; margin: 6px 0 0 0; font-size: 14px; }

    /* Chat bubbles */
    .user-bubble {
        background: #e8f0fe;
        border: 1px solid #c5d5f5;
        border-radius: 12px 12px 4px 12px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #1a237e;
        font-size: 15px;
    }
    .ai-bubble {
        background: #ffffff;
        border: 1px solid #c5d5f5;
        border-left: 4px solid #3f51b5;
        border-radius: 12px 12px 12px 4px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #212121;
        font-size: 15px;
        line-height: 1.8;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }

    /* Source card */
    .source-card {
        background: #f5f7ff;
        border: 1px solid #c5d5f5;
        border-left: 3px solid #3f51b5;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        font-size: 13px;
        color: #3f51b5;
    }

    /* Analysis box */
    .analysis-box {
        background: #ffffff;
        border: 1px solid #c5d5f5;
        border-left: 4px solid #3f51b5;
        border-radius: 12px;
        padding: 20px 24px;
        color: #212121;
        font-size: 15px;
        line-height: 1.8;
        margin: 12px 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    }

    /* Upload box */
    .upload-box {
        background: #ffffff;
        border: 2px dashed #c5d5f5;
        border-radius: 12px;
        padding: 40px 24px;
        text-align: center;
        margin: 12px 0;
    }

    /* Mode badge */
    .mode-badge {
        display: inline-block;
        background: #3f51b5;
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    /* Stat cards */
    .stat-card {
        background: #ffffff;
        border: 1px solid #c5d5f5;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .stat-number { font-size: 26px; font-weight: 700; color: #3f51b5; }
    .stat-label  { font-size: 11px; color: #757575; margin-top: 4px; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e0e0e0;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div {
        color: #212121 !important;
    }
    section[data-testid="stSidebar"] h3 {
        color: #1a237e !important;
        font-size: 18px !important;
        font-weight: 700 !important;
    }

    /* Sidebar header box */
    .sidebar-header {
        background: linear-gradient(135deg, #e8f0fe, #ffffff);
        border: 1px solid #c5d5f5;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }
    .sidebar-header h3 { color: #1a237e !important; margin: 0; font-size: 18px; }
    .sidebar-header p  { color: #5c6bc0 !important; font-size: 12px; margin: 4px 0 0 0; }

    /* Sidebar source card */
    .sidebar-source {
        background: #f5f7ff;
        border: 1px solid #c5d5f5;
        border-left: 3px solid #3f51b5;
        border-radius: 8px;
        padding: 8px 12px;
        margin: 4px 0;
        font-size: 12px;
        color: #3f51b5 !important;
    }

    /* Sidebar model info */
    .sidebar-info {
        background: #f5f7ff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 12px;
    }
    .sidebar-info p { color: #5c6bc0 !important; font-size: 11px; margin: 0; }

    /* Input */
    div[data-testid="stTextInput"] input {
        background-color: #ffffff !important;
        border: 1.5px solid #c5d5f5 !important;
        color: #212121 !important;
        border-radius: 8px !important;
        font-size: 15px !important;
    }
    div[data-testid="stTextInput"] input:focus {
        border-color: #3f51b5 !important;
        box-shadow: 0 0 0 2px rgba(63,81,181,0.15) !important;
    }
    div[data-testid="stTextInput"] input::placeholder {
        color: #9e9e9e !important;
    }

    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #3f51b5, #5c6bc0) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        box-shadow: 0 2px 6px rgba(63,81,181,0.3) !important;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #303f9f, #3f51b5) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #e8f0fe;
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #3f51b5 !important;
        font-weight: 600;
        border-radius: 6px;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #1a237e !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #f5f7ff !important;
        color: #3f51b5 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        border: 1px solid #c5d5f5 !important;
    }

    /* Radio */
    .stRadio label       { color: #212121 !important; font-size: 14px; }
    .stRadio div         { color: #212121 !important; }

    /* Divider */
    hr { border-color: #e0e0e0 !important; }

    /* Warning / Error */
    .stWarning { background: #fff8e1 !important; color: #f57f17 !important; }
    .stError   { background: #ffebee !important; color: #c62828 !important; }
    .stSuccess { background: #e8f5e9 !important; color: #2e7d32 !important; }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #ffffff !important;
        border: 2px dashed #c5d5f5 !important;
        border-radius: 12px !important;
        padding: 12px !important;
    }
    [data-testid="stFileUploader"] label {
        color: #3f51b5 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div class='sidebar-header'>
        <h3>🧠 Research Assistant</h3>
        <p>Powered by LLaMA 3 + FAISS</p>
    </div>
    """, unsafe_allow_html=True)

    # Mode selector
    st.markdown(
        "<p style='color:#1a237e;font-weight:700;"
        "font-size:14px;margin-bottom:6px'>📂 Mode</p>",
        unsafe_allow_html=True
    )
    mode = st.radio(
        "Select mode",
        ["🗄️ Knowledge Base", "📤 Upload New PDF"],
        label_visibility="collapsed"
    )

    if mode == "🗄️ Knowledge Base":
        st.session_state.active_mode = "knowledge_base"
    else:
        st.session_state.active_mode = "upload"

    st.markdown("---")

    # Stats
    col1, col2 = st.columns(2)
    base_pdf_count = len(set(
        c['file_name'] for c in base_chunks
    )) if base_chunks else 0

    with col1:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-number'>{base_pdf_count}</div>
            <div class='stat-label'>Base PDFs</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-number'>{st.session_state.total_questions}</div>
            <div class='stat-label'>Questions</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-number'>{len(base_chunks) if base_chunks else 0}</div>
            <div class='stat-label'>Chunks</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class='stat-card'>
            <div class='stat-number'>{len(st.session_state.chat_history)}</div>
            <div class='stat-label'>Turns</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # PDF list
    if base_chunks:
        st.markdown(
            "<p style='color:#1a237e;font-weight:700;"
            "font-size:13px;margin:0 0 8px 0'>"
            "📁 Knowledge Base Documents</p>",
            unsafe_allow_html=True
        )
        for pdf in list(set(c['file_name'] for c in base_chunks)):
            st.markdown(f"""
            <div class='sidebar-source'>
                📄 {pdf[:35]}{'...' if len(pdf) > 35 else ''}
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history    = []
        st.session_state.upload_chat     = []
        st.session_state.total_questions = 0
        st.rerun()

    st.markdown(f"""
    <div class='sidebar-info'>
        <p>🤖 {LLM_MODEL}<br>
        🔍 {EMBEDDING_MODEL}<br>
        📦 Top-K: {TOP_K_RESULTS}</p>
    </div>""", unsafe_allow_html=True)

# ============================================================
# MAIN AREA
# ============================================================

st.markdown("""
<div class='title-box'>
    <h1>🧠 AI Research Assistant</h1>
    <p>Upload any PDF for instant analysis — or ask questions from your knowledge base</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MODE 1 — KNOWLEDGE BASE
# ============================================================

if st.session_state.active_mode == "knowledge_base":

    st.markdown("""
    <div class='mode-badge'>🗄️ Knowledge Base Mode</div>
    """, unsafe_allow_html=True)

    if base_index is None:
        st.error("❌ No knowledge base found. Run code.py first to build the database.")
        st.stop()

    # Chat display
    if not st.session_state.chat_history:
        st.markdown("""
        <div style='text-align:center;padding:60px 0'>
            <div style='font-size:48px'>🧠</div>
            <p style='font-size:18px;color:#5c6bc0;margin-top:16px;font-weight:600'>
                Ask anything about your research papers
            </p>
            <p style='font-size:13px;color:#9e9e9e'>
                Try: "What is CBT?" or "What are depression relapse rates?"
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for turn in st.session_state.chat_history:
            st.markdown(f"""
            <div class='user-bubble'>
                👤 <strong>You</strong><br>{turn['question']}
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class='ai-bubble'>
                🤖 <strong>Assistant</strong><br><br>{turn['answer']}
            </div>""", unsafe_allow_html=True)

            if turn.get('sources'):
                with st.expander(
                    f"📚 View Sources ({len(turn['sources'])} chunks used)"
                ):
                    seen = set()
                    for chunk in turn['sources']:
                        key = f"{chunk['file_name']}_p{chunk['page_number']}"
                        if key not in seen:
                            seen.add(key)
                            st.markdown(f"""
                            <div class='source-card'>
                                📄 <strong>{chunk['file_name']}</strong>
                                &nbsp;|&nbsp; Page {chunk['page_number']}
                            </div>""", unsafe_allow_html=True)
                            with st.expander(
                                f"Preview — Page {chunk['page_number']}"
                            ):
                                st.write(chunk['chunk_text'][:400] + "...")

            st.markdown("<br>", unsafe_allow_html=True)

    # Input
    st.markdown("---")
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        question = st.text_input(
            "q",
            placeholder="Ask a question about your research papers...",
            label_visibility="collapsed",
            key="kb_input"
        )
    with col_btn:
        ask_clicked = st.button("Ask 🚀", key="kb_ask")

    if ask_clicked and question.strip():
        with st.spinner("🔍 Searching knowledge base and generating answer..."):
            recent  = st.session_state.chat_history[-3:]
            answer, sources = get_answer(
                question, recent,
                base_index, base_chunks
            )
            st.session_state.chat_history.append({
                "question" : question,
                "answer"   : answer,
                "sources"  : sources,
                "time"     : datetime.now().strftime("%H:%M:%S")
            })
            st.session_state.total_questions += 1
        st.rerun()
    elif ask_clicked:
        st.warning("⚠️ Please type a question first")

# ============================================================
# MODE 2 — UPLOAD NEW PDF
# ============================================================

elif st.session_state.active_mode == "upload":

    st.markdown("""
    <div class='mode-badge'>📤 Upload Mode — Any PDF</div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload any PDF file",
        type=["pdf"],
        help="Upload any PDF — research paper, book, report, article"
    )

    if uploaded_file is not None:

        if st.session_state.uploaded_file_name != uploaded_file.name:

            with st.spinner(f"⏳ Processing {uploaded_file.name}..."):
                pdf_bytes       = uploaded_file.read()
                pages_data      = extract_pdf_text_with_pages(pdf_bytes)
                uploaded_chunks = chunk_pages(pages_data, uploaded_file.name)
                uploaded_index  = build_faiss_index(uploaded_chunks)
                full_text       = extract_full_text(pdf_bytes)

                # Guard — scanned or empty PDF
                if uploaded_index is None:
                    st.error("""
                        ❌ Could not process this PDF.
                        This usually means the PDF is scanned or image-based
                        with no extractable text. Please try a text-based PDF.
                    """)
                    st.session_state.uploaded_file_name = None
                    st.stop()

                st.session_state.uploaded_chunks    = uploaded_chunks
                st.session_state.uploaded_index     = uploaded_index
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.uploaded_full_text = full_text
                st.session_state.doc_analysis_result = None
                st.session_state.upload_chat        = []

            st.success(
                f"✅ {uploaded_file.name} processed — "
                f"{len(uploaded_chunks)} chunks ready"
            )

        # File info bar
        st.markdown(f"""
        <div style='background:#e8f0fe;border:1px solid #c5d5f5;
        border-radius:10px;padding:14px 18px;margin:12px 0'>
            <p style='color:#1a237e;font-weight:700;margin:0;font-size:15px'>
                📄 {st.session_state.uploaded_file_name}
            </p>
            <p style='color:#5c6bc0;font-size:13px;margin:4px 0 0 0'>
                {len(st.session_state.uploaded_chunks)} chunks indexed and ready
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Tabs
        tab1, tab2 = st.tabs(["📊 Document Analysis", "💬 Ask Questions"])

        # ── TAB 1 — Document Analysis ──
        with tab1:
            st.markdown(
                "<p style='color:#1a237e;font-weight:700;"
                "font-size:16px;margin-bottom:16px'>"
                "What would you like to extract?</p>",
                unsafe_allow_html=True
            )

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1: abs_btn = st.button("📋 Abstract",     use_container_width=True)
            with col2: con_btn = st.button("🏁 Conclusion",   use_container_width=True)
            with col3: sum_btn = st.button("📝 Summary",      use_container_width=True)
            with col4: met_btn = st.button("🔬 Methods",      use_container_width=True)
            with col5: kf_btn  = st.button("💡 Key Findings", use_container_width=True)

            analysis_type = None
            if abs_btn: analysis_type = "abstract"
            if con_btn: analysis_type = "conclusion"
            if sum_btn: analysis_type = "summary"
            if met_btn: analysis_type = "methods"
            if kf_btn:  analysis_type = "key_findings"

            if analysis_type:
                with st.spinner(f"⏳ Extracting {analysis_type}..."):
                    result = analyze_document(
                        st.session_state.uploaded_full_text,
                        analysis_type,
                        st.session_state.uploaded_file_name
                    )
                    st.session_state.doc_analysis_result = result
                    st.session_state.analysis_type       = analysis_type

            if st.session_state.doc_analysis_result:
                type_labels = {
                    "abstract"    : "📋 Abstract",
                    "conclusion"  : "🏁 Conclusion",
                    "summary"     : "📝 Summary",
                    "methods"     : "🔬 Methods",
                    "key_findings": "💡 Key Findings"
                }
                label = type_labels.get(st.session_state.analysis_type, "📄 Result")
                st.markdown(
                    f"<p style='color:#1a237e;font-weight:700;"
                    f"font-size:16px;margin:16px 0 8px 0'>{label}</p>",
                    unsafe_allow_html=True
                )
                st.markdown(f"""
                <div class='analysis-box'>
                    {st.session_state.doc_analysis_result}
                </div>
                """, unsafe_allow_html=True)

        # ── TAB 2 — Ask Questions ──
        with tab2:
            st.markdown(
                "<p style='color:#1a237e;font-weight:700;"
                "font-size:16px;margin-bottom:12px'>"
                "Ask anything about this document</p>",
                unsafe_allow_html=True
            )

            if not st.session_state.upload_chat:
                st.markdown(f"""
                <div style='text-align:center;padding:40px 0'>
                    <div style='font-size:36px'>📄</div>
                    <p style='color:#5c6bc0;margin-top:12px;font-weight:600'>
                        Ask any question about<br>
                        <strong style='color:#1a237e'>
                        {st.session_state.uploaded_file_name}
                        </strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                for turn in st.session_state.upload_chat:
                    st.markdown(f"""
                    <div class='user-bubble'>
                        👤 <strong>You</strong><br>{turn['question']}
                    </div>""", unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class='ai-bubble'>
                        🤖 <strong>Assistant</strong><br><br>{turn['answer']}
                    </div>""", unsafe_allow_html=True)

                    if turn.get('sources'):
                        with st.expander(
                            f"📚 Sources ({len(turn['sources'])} chunks)"
                        ):
                            seen = set()
                            for chunk in turn['sources']:
                                key = f"{chunk['file_name']}_p{chunk['page_number']}"
                                if key not in seen:
                                    seen.add(key)
                                    st.markdown(f"""
                                    <div class='source-card'>
                                        📄 <strong>{chunk['file_name']}</strong>
                                        &nbsp;|&nbsp; Page {chunk['page_number']}
                                    </div>""", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("---")
            col_q, col_b = st.columns([5, 1])
            with col_q:
                upload_q = st.text_input(
                    "uq",
                    placeholder=f"Ask about {st.session_state.uploaded_file_name}...",
                    label_visibility="collapsed",
                    key="upload_input"
                )
            with col_b:
                upload_ask = st.button("Ask 🚀", key="upload_ask")

            if upload_ask and upload_q.strip():
                with st.spinner("🔍 Searching document..."):
                    recent = st.session_state.upload_chat[-3:]
                    answer, sources = get_answer(
                        upload_q,
                        recent,
                        st.session_state.uploaded_index,
                        st.session_state.uploaded_chunks
                    )
                    st.session_state.upload_chat.append({
                        "question" : upload_q,
                        "answer"   : answer,
                        "sources"  : sources,
                        "time"     : datetime.now().strftime("%H:%M:%S")
                    })
                    st.session_state.total_questions += 1
                st.rerun()
            elif upload_ask:
                st.warning("⚠️ Please type a question first")

    else:
        st.markdown("""
        <div class='upload-box'>
            <div style='font-size:48px'>📤</div>
            <p style='color:#5c6bc0;font-size:16px;
            margin:12px 0;font-weight:600'>
                Upload any PDF file to get started
            </p>
            <p style='color:#9e9e9e;font-size:13px'>
                Research papers · Books · Reports · Articles · Any PDF
            </p>
        </div>
        """, unsafe_allow_html=True)
# %%
