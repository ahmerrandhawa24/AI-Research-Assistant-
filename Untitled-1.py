
#%%
import fitz                          
import pdfplumber                    
import faiss                         
import chromadb                      
import streamlit                     
from sentence_transformers import SentenceTransformer   
from groq import Groq                
from dotenv import load_dotenv       
from config import *                 

print("✅ All imports successful")

# %%
if GROQ_API_KEY:
    print(f"✅ Groq API key loaded: {GROQ_API_KEY[:8]}...")
else:
    print("❌ Groq API key NOT found - check your .env file")
# %%
client = Groq(api_key=GROQ_API_KEY)

response = client.chat.completions.create(
    model=LLM_MODEL,
    messages=[
        {"role": "user", "content": "Say hello in one sentence."}
    ]
)

print(f"✅ Groq LLM working: {response.choices[0].message.content}")
# %%
print("\n⏳ Loading embedding model (first time may take 1-2 min)...")

embedder = SentenceTransformer(EMBEDDING_MODEL)
test_embedding = embedder.encode("This is a test sentence.")

print(f"✅ Embedding model working — vector size: {len(test_embedding)}")
# %%
# step 2 ............. 
#%%
import fitz          # PyMuPDF
import os
from config import PDF_FOLDER


# %%
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    raw_text = ""

    for page_num, page in enumerate(doc):
        text = page.get_text()
        raw_text += f"\n--- Page {page_num + 1} ---\n{text}"

    doc.close()
    return raw_text


def load_all_pdfs(folder_path):
    all_documents = []

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    if not pdf_files:
        print("❌ No PDFs found in folder")
        return []

    print(f"📂 Found {len(pdf_files)} PDFs\n")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        text = extract_text_from_pdf(pdf_path)

        all_documents.append({
            "file_name": pdf_file,
            "text": text,
            "total_chars": len(text)
        })

        print(f"  ✅ {pdf_file} — {len(text)} chars")

    print(f"\n📊 Total documents loaded: {len(all_documents)}")
    total = sum(d['total_chars'] for d in all_documents)
    print(f"📊 Total characters across all PDFs: {total}")

    return all_documents


# --- Run it ---
all_documents = load_all_pdfs(PDF_FOLDER)

# %%
import re

def clean_text(text):
    # Remove page markers we added
    text = re.sub(r'--- Page \d+ ---', '', text)

    # Remove extra whitespace and blank lines
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r' +', ' ', text)

    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-\'\"]+', ' ', text)

    # Strip leading/trailing spaces
    text = text.strip()

    return text

for doc in all_documents:
    original_len = len(doc["text"])
    doc["cleaned_text"] = clean_text(doc["text"])
    cleaned_len = len(doc["cleaned_text"])

    print(f"✅ {doc['file_name']}")
    print(f"   Before: {original_len} chars → After: {cleaned_len} chars")
    print(f"   Removed: {original_len - cleaned_len} chars of noise\n")

print("✅ All documents cleaned")
# %%
from config import CHUNK_SIZE, CHUNK_OVERLAP

def split_into_chunks(text, file_name):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + CHUNK_SIZE
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "file_name": file_name,
            "chunk_text": chunk_text,
            "chunk_index": len(chunks),
            "word_count": len(chunk_words)
        })

        # Move forward but keep overlap
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# --- Apply chunking to all documents ---
all_chunks = []

for doc in all_documents:
    chunks = split_into_chunks(doc["cleaned_text"], doc["file_name"])
    all_chunks.extend(chunks)

    print(f"✅ {doc['file_name']}")
    print(f"   → {len(chunks)} chunks created\n")

print(f"📊 Total chunks across all PDFs: {len(all_chunks)}")
print(f"\n--- Sample Chunk Preview ---")
print(f"File     : {all_chunks[0]['file_name']}")
print(f"Chunk #  : {all_chunks[0]['chunk_index']}")
print(f"Words    : {all_chunks[0]['word_count']}")
print(f"Text     : {all_chunks[0]['chunk_text'][:200]}...")
# %%
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages_data = []

    for page_num, page in enumerate(doc):
        text = page.get_text()
        cleaned = clean_text(text)

        if cleaned.strip():  # skip empty pages
            pages_data.append({
                "file_name": os.path.basename(pdf_path),
                "page_number": page_num + 1,
                "text": cleaned
            })

    doc.close()
    print(f"✅ {os.path.basename(pdf_path)} — {len(pages_data)} pages extracted")
    return pages_data


def load_all_pdfs(folder_path):
    all_pages = []
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    print(f"📂 Found {len(pdf_files)} PDFs\n")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        pages = extract_text_from_pdf(pdf_path)
        all_pages.extend(pages)

    print(f"\n📊 Total pages loaded: {len(all_pages)}")
    return all_pages


# --- Run it ---
all_pages = load_all_pdfs(PDF_FOLDER)
# %%
def split_into_chunks(pages_data):
    all_chunks = []
    chunk_id = 0

    for page in pages_data:
        words = page["text"].split()
        start = 0

        while start < len(words):
            end = start + CHUNK_SIZE
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            all_chunks.append({
                "chunk_id":     chunk_id,
                "chunk_text":   chunk_text,
                "file_name":    page["file_name"],
                "page_number":  page["page_number"],
                "word_count":   len(chunk_words)
            })

            chunk_id += 1
            start += CHUNK_SIZE - CHUNK_OVERLAP

    return all_chunks


# --- Run it ---
all_chunks = split_into_chunks(all_pages)

print(f"📊 Total chunks created: {len(all_chunks)}")
print(f"\n--- Sample Chunk (with full metadata) ---")
sample = all_chunks[10]
print(f"  chunk_id   : {sample['chunk_id']}")
print(f"  file_name  : {sample['file_name']}")
print(f"  page_number: {sample['page_number']}")
print(f"  word_count : {sample['word_count']}")
print(f"  chunk_text : {sample['chunk_text'][:200]}...")

# ============================================================
# PHASE 3 - Imports
# ============================================================

#%%
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, VECTOR_DB_PATH

# %%
print("⏳ Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
print(f"✅ Embedding model loaded: {EMBEDDING_MODEL}")

# --- Quick test ---
test = embedder.encode("test sentence")
print(f"✅ Vector size: {len(test)} dimensions")
# %%
print(f"⏳ Converting {len(all_chunks)} chunks to embeddings...")
print("   This may take 1-2 minutes...\n")

# Extract just the text from each chunk
chunk_texts = [chunk["chunk_text"] for chunk in all_chunks]

# Convert all texts to embeddings in one go
embeddings = embedder.encode(
    chunk_texts,
    show_progress_bar=True,
    batch_size=32
)

print(f"\n✅ Embeddings created")
print(f"   Total embeddings : {len(embeddings)}")
print(f"   Embedding shape  : {embeddings.shape}")
print(f"   Each vector size : {embeddings.shape[1]} dimensions")
# %%
embeddings_float = np.array(embeddings).astype("float32")

# Get vector size (384)
vector_size = embeddings_float.shape[1]

# Create FAISS index
index = faiss.IndexFlatL2(vector_size)

# Add all embeddings to index
index.add(embeddings_float)

print(f"✅ FAISS index created")
print(f"   Vectors stored : {index.ntotal}")
print(f"   Vector size    : {vector_size} dimensions")
# %%
import os

# --- File paths ---
faiss_index_path    = os.path.join(VECTOR_DB_PATH, "index.faiss")
metadata_path       = os.path.join(VECTOR_DB_PATH, "metadata.pkl")

# --- Save FAISS index ---
faiss.write_index(index, faiss_index_path)
print(f"✅ FAISS index saved  : {faiss_index_path}")

# --- Save metadata (all chunk info) ---
with open(metadata_path, "wb") as f:
    pickle.dump(all_chunks, f)
print(f"✅ Metadata saved     : {metadata_path}")

# --- Verify files exist on disk ---
faiss_size    = os.path.getsize(faiss_index_path) / 1024
metadata_size = os.path.getsize(metadata_path) / 1024

print(f"\n📁 Files saved in: {VECTOR_DB_PATH}/")
print(f"   index.faiss  : {faiss_size:.1f} KB")
print(f"   metadata.pkl : {metadata_size:.1f} KB")
print(f"\n✅ Phase 3 complete — database ready on disk")

# %%
loaded_index = faiss.read_index(faiss_index_path)
print(f"✅ FAISS index loaded from disk")
print(f"   Vectors in index : {loaded_index.ntotal}")

# --- Load metadata ---
with open(metadata_path, "rb") as f:
    loaded_chunks = pickle.load(f)
print(f"✅ Metadata loaded from disk")
print(f"   Total chunks     : {len(loaded_chunks)}")

# --- Quick sanity check ---
print(f"\n--- Sample loaded chunk ---")
sample = loaded_chunks[5]
print(f"  chunk_id   : {sample['chunk_id']}")
print(f"  file_name  : {sample['file_name']}")
print(f"  page_number: {sample['page_number']}")
print(f"  chunk_text : {sample['chunk_text'][:150]}...")

print(f"\n🎉 Phase 3 complete — vector database working perfectly!")
# %%
import faiss
import numpy as np
from config import TOP_K_RESULTS

# %%
def search(query, top_k=TOP_K_RESULTS):

    # Step 1 — Convert question to embedding
    query_embedding = embedder.encode([query]).astype("float32")

    # Step 2 — Search FAISS for most similar chunks
    distances, indices = loaded_index.search(query_embedding, top_k)

    # Step 3 — Retrieve matching chunks with metadata
    results = []
    for i, idx in enumerate(indices[0]):
        chunk = loaded_chunks[idx]
        results.append({
            "rank":         i + 1,
            "score":        round(float(distances[0][i]), 4),
            "chunk_id":     chunk["chunk_id"],
            "file_name":    chunk["file_name"],
            "page_number":  chunk["page_number"],
            "chunk_text":   chunk["chunk_text"]
        })

    return results
# %%
query = "What are the symptoms of depression?"

print(f"🔍 Query: {query}\n")
print(f"{'='*60}\n")

results = search(query)

for r in results:
    print(f"  Rank          : #{r['rank']}")
    print(f"  Score         : {r['score']} (lower = more relevant)")
    print(f"  File          : {r['file_name']}")
    print(f"  Page          : {r['page_number']}")
    print(f"  Text preview  : {r['chunk_text'][:200]}...")
    print(f"\n{'-'*60}\n")
# %%
# ============================================================
# PHASE 4 - Test with different question (multi-PDF test)
# ============================================================

query2 = "What is cognitive behavioral therapy?"

print(f"🔍 Query: {query2}\n")
print(f"{'='*60}\n")

results2 = search(query2)

for r in results2:
    print(f"  Rank     : #{r['rank']}")
    print(f"  Score    : {r['score']}")
    print(f"  File     : {r['file_name']}")
    print(f"  Page     : {r['page_number']}")
    print(f"  Preview  : {r['chunk_text'][:150]}...")
    print(f"\n{'-'*60}\n")
    
# ----------------------------
# phase 5 
# ----------------

#%%
import importlib
import config
importlib.reload(config)

from config import GROQ_API_KEY, LLM_MODEL
from groq import Groq

print(f"✅ Model loaded: {LLM_MODEL}")

# %%
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL
# %%
client = Groq(api_key=GROQ_API_KEY)
print(f"✅ Groq client ready")
print(f"   Model : {LLM_MODEL}")
# %%
def build_prompt(question, retrieved_chunks):

    # Build context from retrieved chunks
    context = ""
    for chunk in retrieved_chunks:
        context += f"""
Source   : {chunk['file_name']}
Page     : {chunk['page_number']}
Content  : {chunk['chunk_text']}
---
"""

    # Final prompt
    prompt = f"""You are an expert Mental Health Research Assistant.
You help researchers and clinicians understand depression, therapy, and related topics.

Below is context retrieved from research papers:

{context}

Instructions:
- Answer ONLY using the context provided above
- If the answer is not in the context say: "This topic is not covered in my knowledge base"
- Be clear, professional and detailed
- Always mention which paper and page your answer comes from
- Do NOT make up any information
- Ignore reference lists or bibliography sections in context

Question: {question}

Answer:"""

    return prompt


# --- Test it ---
test_question = "What are the symptoms of depression?"
test_chunks = search(test_question)
test_prompt = build_prompt(test_question, test_chunks)

print("✅ Prompt built successfully")
print(f"   Total prompt length : {len(test_prompt)} chars")
print(f"\n--- Prompt Preview ---\n")
print(test_prompt[:500] + "...")
# %%
def get_answer(question):

    # Step 1 — retrieve relevant chunks
    retrieved_chunks = search(question)

    # Step 2 — build prompt
    prompt = build_prompt(question, retrieved_chunks)

    # Step 3 — send to Groq LLM
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,  # low = more factual, less creative
        max_tokens=1024
    )

    # Step 4 — extract answer
    answer = response.choices[0].message.content

    return {
        "question"  : question,
        "answer"    : answer,
        "sources"   : retrieved_chunks
    }


# --- Test it ---
result = get_answer("What are the symptoms of depression?")

print(f"❓ Question : {result['question']}")
print(f"\n{'='*60}\n")
print(f"🤖 Answer:\n")
print(result['answer'])

# %%
# ============================================================
# PHASE 5 - Test with domain specific question
# ============================================================

result2 = get_answer("How effective is cognitive behavioral therapy for depression?")

print(f"❓ Question : {result2['question']}")
print(f"\n{'='*60}\n")
print(f"🤖 Answer:\n")
print(result2['answer'])
# %%
# ============================================================
# PHASE 5 - Display answer with clean citations
# ============================================================

def display_result(result):
    print(f"❓ Question : {result['question']}")
    print(f"\n{'='*60}\n")
    print(f"🤖 Answer:\n")
    print(result['answer'])
    print(f"\n{'='*60}")
    print(f"📚 Sources Used:\n")

    seen = set()
    for chunk in result['sources']:
        source_key = f"{chunk['file_name']}_p{chunk['page_number']}"
        if source_key not in seen:
            seen.add(source_key)
            print(f"  📄 {chunk['file_name']}")
            print(f"     Page : {chunk['page_number']}")
            print()


# --- Test it ---
result3 = get_answer("What is the relapse rate in depression?")
display_result(result3)


# ============================================================
# PHASE 6 - Imports
# ============================================================

#%%
import json
from datetime import datetime
# %%
TOP_K_RESULTS = 8
EXCLUDE_LAST_N_PAGES = 5
# %%
# ============================================================
# PHASE 2 - UPDATED extract text (skips reference pages)
# ============================================================

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages_data = []
    total_pages = len(doc)

    for page_num, page in enumerate(doc):

        # Skip last N pages (bibliography/references)
        if page_num >= total_pages - EXCLUDE_LAST_N_PAGES:
            continue

        text = page.get_text()
        cleaned = clean_text(text)

        if cleaned.strip():
            pages_data.append({
                "file_name"  : os.path.basename(pdf_path),
                "page_number": page_num + 1,
                "text"       : cleaned
            })

    doc.close()
    print(f"✅ {os.path.basename(pdf_path)}")
    print(f"   Total pages    : {total_pages}")
    print(f"   Pages skipped  : 5 (references)")
    print(f"   Pages kept     : {len(pages_data)}")
    return pages_data
# %%
# ============================================================
# RE-RUN — Reload config first
# ============================================================

import importlib
import config
importlib.reload(config)

from config import (
    PDF_FOLDER,
    VECTOR_DB_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EXCLUDE_LAST_N_PAGES
)

print(f"✅ Config reloaded")
print(f"   TOP_K_RESULTS       : 8")
print(f"   EXCLUDE_LAST_N_PAGES: {EXCLUDE_LAST_N_PAGES}")
# %%
# ============================================================
# RE-RUN — Re-chunk all pages
# ============================================================

all_chunks = split_into_chunks(all_pages)
print(f"📊 Total chunks : {len(all_chunks)}")
# %%
# ============================================================
# RE-RUN — Rebuild embeddings
# ============================================================

print("⏳ Rebuilding embeddings...")
chunk_texts     = [chunk["chunk_text"] for chunk in all_chunks]
embeddings      = embedder.encode(
    chunk_texts,
    show_progress_bar=True,
    batch_size=32
)
print(f"✅ Embeddings rebuilt : {embeddings.shape}")
# %%
# ============================================================
# RE-RUN — Rebuild FAISS index
# ============================================================

embeddings_float = np.array(embeddings).astype("float32")
vector_size      = embeddings_float.shape[1]
index            = faiss.IndexFlatL2(vector_size)
index.add(embeddings_float)
print(f"✅ FAISS index rebuilt : {index.ntotal} vectors")
# %%
# ============================================================
# RE-RUN — Save to disk
# ============================================================

faiss.write_index(index, faiss_index_path)
with open(metadata_path, "wb") as f:
    pickle.dump(all_chunks, f)

print(f"✅ Saved to disk")
print(f"   index.faiss  : {os.path.getsize(faiss_index_path)/1024:.1f} KB")
print(f"   metadata.pkl : {os.path.getsize(metadata_path)/1024:.1f} KB")
# %%
# ============================================================
# RE-RUN — Reload from disk
# ============================================================

loaded_index  = faiss.read_index(faiss_index_path)
with open(metadata_path, "rb") as f:
    loaded_chunks = pickle.load(f)

print(f"✅ Reloaded from disk")
print(f"   Vectors  : {loaded_index.ntotal}")
print(f"   Chunks   : {len(loaded_chunks)}")

# ============================================================
# PHASE 6 - Clean chunk filter
# ============================================================
#%%
def filter_chunks(chunks):
    filtered  = []
    reference_keywords = [
        "et al.", "doi:", "http", "journal of",
        "proceedings of", "isbn", "publisher",
        "retrieved from", "vol.", "pp.", "ed.",
        "translational psychiatry", "british journal",
        "american journal", "cochrane",
    ]

    for chunk in chunks:
        text_lower       = chunk['chunk_text'].lower()
        keyword_count    = sum(1 for kw in reference_keywords if kw in text_lower)
        citation_pattern = re.findall(r'\(\d{4}\)', chunk['chunk_text'])
        is_reference     = keyword_count >= 2 or len(citation_pattern) >= 4

        if is_reference:
            print(f"  🚫 Filtered — page {chunk['page_number']} of {chunk['file_name']}")
        else:
            filtered.append(chunk)

    print(f"  ✅ Kept {len(filtered)} clean chunks out of {len(chunks)}")
    return filtered

# ============================================================
# PHASE 6 - Updated get_answer with filtering
# ============================================================
#%%
def get_answer(question):

    # Step 1 — retrieve
    retrieved_chunks = search(question)

    # Step 2 — filter reference chunks
    print("🔍 Filtering chunks...\n")
    clean_chunks = filter_chunks(retrieved_chunks)

    # Step 3 — fallback if all filtered
    if not clean_chunks:
        print("  ⚠️ All chunks filtered — using original")
        clean_chunks = retrieved_chunks

    # Step 4 — build prompt
    prompt = build_prompt(question, clean_chunks)

    # Step 5 — send to LLM
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024
    )

    answer = response.choices[0].message.content

    return {
        "question" : question,
        "answer"   : answer,
        "sources"  : clean_chunks
    }
# %%
# ============================================================
# PHASE 6 - Updated get_answer with filtering
# ============================================================

def get_answer(question):

    # Step 1 — retrieve
    retrieved_chunks = search(question)

    # Step 2 — filter reference chunks
    print("🔍 Filtering chunks...\n")
    clean_chunks = filter_chunks(retrieved_chunks)

    # Step 3 — fallback if all filtered
    if not clean_chunks:
        print("  ⚠️ All chunks filtered — using original")
        clean_chunks = retrieved_chunks

    # Step 4 — build prompt
    prompt = build_prompt(question, clean_chunks)

    # Step 5 — send to LLM
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024
    )

    answer = response.choices[0].message.content

    return {
        "question" : question,
        "answer"   : answer,
        "sources"  : clean_chunks
    }
# %%
# ============================================================
# PHASE 6 - Evaluate faithfulness (hallucination check)
# ============================================================

def evaluate_faithfulness(question, answer, retrieved_chunks):
    context = "\n---\n".join([
        c['chunk_text'][:300]
        for c in retrieved_chunks
    ])

    prompt = f"""You are checking if an AI answer is faithful to its source.

Question: {question}

Context given to AI:
{context}

AI Answer:
{answer}

Respond in this exact format:
Hallucination detected : YES/NO
Made up facts          : YES/NO
Citations correct      : YES/NO
Faithfulness score     : X/10
Verdict                : one sentence"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=200
    )
    return response.choices[0].message.content
# %%
# ============================================================
# PHASE 6 - Run full evaluation on 3 questions
# ============================================================

test_questions = [
    "How effective is cognitive behavioral therapy for depression?",
    "What is the relapse rate in depression?",
    "What therapies are used for treating depression?"
]

for question in test_questions:
    print(f"\n{'='*60}")
    print(f"❓ {question}")
    print(f"{'='*60}\n")

    # Get answer
    result = get_answer(question)

    # Display answer
    print(f"🤖 Answer:\n{result['answer']}\n")

    # Evaluate retrieval
    print(f"📊 Retrieval Evaluation:\n")
    retrieval_score = evaluate_retrieval(
        question,
        result['sources']
    )
    print(retrieval_score)

    # Evaluate faithfulness
    print(f"\n🔍 Faithfulness Check:\n")
    faith_score = evaluate_faithfulness(
        result['question'],
        result['answer'],
        result['sources']
    )
    print(faith_score)
    print(f"\n{'='*60}\n")
# %%
# ============================================================
# PHASE 4 - Updated search with smarter retrieval
# ============================================================

def search(query, top_k=TOP_K_RESULTS):

    # Step 1 — embed query
    query_embedding = embedder.encode([query]).astype("float32")

    # Step 2 — search more chunks than needed (2x buffer)
    distances, indices = loaded_index.search(query_embedding, top_k * 3)

    # Step 3 — retrieve all candidates
    candidates = []
    for i, idx in enumerate(indices[0]):
        chunk = loaded_chunks[idx]
        candidates.append({
            "rank"        : i + 1,
            "score"       : round(float(distances[0][i]), 4),
            "chunk_id"    : chunk["chunk_id"],
            "file_name"   : chunk["file_name"],
            "page_number" : chunk["page_number"],
            "chunk_text"  : chunk["chunk_text"]
        })

    # Step 4 — filter reference chunks immediately
    reference_keywords = [
        "et al.", "doi:", "http", "journal of",
        "proceedings of", "isbn", "publisher",
        "retrieved from", "vol.", "pp.", "ed.",
        "translational psychiatry", "british journal",
        "american journal", "cochrane",
    ]

    clean_candidates = []
    for chunk in candidates:
        text_lower       = chunk['chunk_text'].lower()
        keyword_count    = sum(1 for kw in reference_keywords if kw in text_lower)
        citation_pattern = re.findall(r'\(\d{4}\)', chunk['chunk_text'])
        is_reference     = keyword_count >= 2 or len(citation_pattern) >= 4

        if not is_reference:
            clean_candidates.append(chunk)

    # Step 5 — return top_k clean results
    # fallback to candidates if all filtered
    final = clean_candidates[:top_k] if clean_candidates else candidates[:top_k]

    print(f"  📦 Retrieved {len(candidates)} chunks")
    print(f"  🚫 Filtered  {len(candidates) - len(clean_candidates)} reference chunks")
    print(f"  ✅ Returning {len(final)} clean chunks\n")

    return final
# %%
# ============================================================
# PHASE 5 - Updated get_answer (filtering now in search)
# ============================================================

def get_answer(question):

    # Step 1 — retrieve clean chunks
    print("🔍 Searching...\n")
    clean_chunks = search(question)

    # Step 2 — build prompt
    prompt = build_prompt(question, clean_chunks)

    # Step 3 — send to LLM
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024
    )

    answer = response.choices[0].message.content

    return {
        "question" : question,
        "answer"   : answer,
        "sources"  : clean_chunks
    }
# %%
# ============================================================
# PHASE 6 - Full evaluation test after fix
# ============================================================

test_questions = [
    "How effective is cognitive behavioral therapy for depression?",
    "What is the relapse rate in depression?",
    "What therapies are used for treating depression?"
]

for question in test_questions:
    print(f"\n{'='*60}")
    print(f"❓ {question}")
    print(f"{'='*60}\n")

    result = get_answer(question)

    print(f"🤖 Answer:\n{result['answer']}\n")

    print(f"📊 Retrieval Evaluation:\n")
    print(evaluate_retrieval(question, result['sources']))

    print(f"\n🔍 Faithfulness Check:\n")
    print(evaluate_faithfulness(
        result['question'],
        result['answer'],
        result['sources']
    ))
# %%
# ============================================================
# PHASE 6 - Faithfulness check only
# ============================================================

question = "How effective is cognitive behavioral therapy for depression?"

result = get_answer(question)

print(f"\n🔍 Faithfulness Check:\n")
faith = evaluate_faithfulness(
    result['question'],
    result['answer'],
    result['sources']
)
print(faith)
# %%
# ============================================================
# PHASE 5 - Updated build_prompt
# Stops LLM from extracting citations from chunk text
# ============================================================

def build_prompt(question, retrieved_chunks):

    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += f"""
[Chunk {i+1}]
File    : {chunk['file_name']}
Page    : {chunk['page_number']}
Content : {chunk['chunk_text']}
---
"""

    prompt = f"""You are an expert Mental Health Research Assistant.
You help researchers and clinicians understand depression, therapy and related topics.

Below is context retrieved from research papers:

{context}

STRICT RULES — follow exactly:
- Use ONLY information explicitly written in the Content sections above
- Do NOT use any author names, years, or study names mentioned inside the content
- Cite sources ONLY using the File and Page shown above each chunk
- Format citations as: (File: filename, Page: X)
- If answer is not in context say: "This topic is not covered in my knowledge base"
- Do NOT make up statistics, percentages, or effect sizes
- Do NOT reference any external studies

Question: {question}

Answer (cite only File and Page, never author names or years):"""

    return prompt
# %%
# ============================================================
# PHASE 6 - Re-test faithfulness after prompt fix
# ============================================================

question = "How effective is cognitive behavioral therapy for depression?"

result = get_answer(question)

print(f"🤖 Answer:\n")
print(result['answer'])

print(f"\n🔍 Faithfulness Check:\n")
faith = evaluate_faithfulness(
    result['question'],
    result['answer'],
    result['sources']
)
print(faith)
# %%
# ============================================================
# PHASE 6 - Fixed faithfulness checker
# Uses full chunk text as verification source
# ============================================================

def evaluate_faithfulness(question, answer, retrieved_chunks):

    # Pass full chunk text not just 300 chars
    context = "\n---\n".join([
        f"[File: {c['file_name']}, Page: {c['page_number']}]\n{c['chunk_text']}"
        for c in retrieved_chunks
    ])

    prompt = f"""You are checking if an AI answer is faithful to its source.

Question: {question}

Full context given to AI:
{context}

AI Answer:
{answer}

Check ONLY these things:
1. Are all facts in the answer present somewhere in the context?
2. Are file names cited correctly?
3. Are page numbers cited present in the context above?

Respond in this exact format:
Hallucination detected : YES/NO
Made up facts          : YES/NO
Citations correct      : YES/NO
Faithfulness score     : X/10
Verdict                : one sentence"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=200
    )
    return response.choices[0].message.content


# --- Re-test ---
print(f"\n🔍 Faithfulness Check (fixed):\n")
faith = evaluate_faithfulness(
    result['question'],
    result['answer'],
    result['sources']
)
print(faith)

# ============================================================
# PHASE 7 - CELL 1 - Imports
# ============================================================
#%%
from datetime import datetime
print("✅ Phase 7 imports done")
# %%
# ============================================================
# PHASE 7 - CELL 2 - Chat memory store
# Stores full conversation history for the session
# ============================================================

# Initialize empty chat history
chat_history = []

def add_to_history(question, answer, sources):
    chat_history.append({
        "turn"      : len(chat_history) + 1,
        "time"      : datetime.now().strftime("%H:%M:%S"),
        "question"  : question,
        "answer"    : answer,
        "sources"   : sources
    })

def get_recent_history(n=3):
    # Return last n turns for context window
    return chat_history[-n:] if len(chat_history) >= n else chat_history

def clear_history():
    chat_history.clear()
    print("✅ Chat history cleared")

def show_history():
    if not chat_history:
        print("📭 No chat history yet")
        return

    print(f"📚 Chat History — {len(chat_history)} turns\n")
    for turn in chat_history:
        print(f"  Turn     : {turn['turn']}")
        print(f"  Time     : {turn['time']}")
        print(f"  Question : {turn['question']}")
        print(f"  Answer   : {turn['answer'][:150]}...")
        print(f"  {'─'*50}")

print("✅ Chat memory system ready")
# %%
# ============================================================
# PHASE 7 - CELL 3 - Build prompt with memory
# Passes recent chat history into every prompt
# ============================================================

def build_prompt_with_memory(question, retrieved_chunks, history):

    # Build context from retrieved chunks
    context = ""
    for i, chunk in enumerate(retrieved_chunks):
        context += f"""
[Chunk {i+1}]
File    : {chunk['file_name']}
Page    : {chunk['page_number']}
Content : {chunk['chunk_text']}
---
"""

    # Build conversation history string
    history_text = ""
    if history:
        history_text = "Previous conversation:\n"
        for turn in history:
            history_text += f"User     : {turn['question']}\n"
            history_text += f"Assistant: {turn['answer'][:300]}\n"
            history_text += "---\n"

    prompt = f"""You are an expert Mental Health Research Assistant.
You help researchers and clinicians understand depression, therapy and related topics.

{history_text}

Below is context retrieved from research papers:
{context}

STRICT RULES:
- Use ONLY information explicitly written in the Content sections above
- Do NOT use any author names, years, or study names mentioned inside the content
- Cite sources ONLY using File and Page shown above each chunk
- Format citations as: (File: filename, Page: X)
- If the question refers to something from previous conversation, use that context
- If answer is not in context say: "This topic is not covered in my knowledge base"
- Do NOT make up statistics, percentages, or effect sizes
- Do NOT reference any external studies

Question: {question}

Answer (cite only File and Page, never author names or years):"""

    return prompt


print("✅ Memory-aware prompt builder ready")
# %%
# ============================================================
# PHASE 7 - CELL 4 - Updated get_answer with memory
# ============================================================

def get_answer(question):

    # Step 1 — retrieve clean chunks
    print("🔍 Searching...\n")
    clean_chunks = search(question)

    # Step 2 — get recent history
    history = get_recent_history(n=3)

    # Step 3 — build prompt with memory
    prompt = build_prompt_with_memory(question, clean_chunks, history)

    # Step 4 — send to LLM
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1024
    )

    answer = response.choices[0].message.content

    # Step 5 — save to history
    add_to_history(question, answer, clean_chunks)

    return {
        "question" : question,
        "answer"   : answer,
        "sources"  : clean_chunks
    }


print("✅ Memory-aware get_answer ready")
# %%
# ============================================================
# PHASE 7 - CELL 5 - Display result cleanly
# ============================================================

def display_result(result):
    print(f"\n❓ Question : {result['question']}")
    print(f"\n{'='*60}\n")
    print(f"🤖 Answer:\n")
    print(result['answer'])
    print(f"\n{'='*60}")
    print(f"📚 Sources Used:\n")

    seen = set()
    for chunk in result['sources']:
        source_key = f"{chunk['file_name']}_p{chunk['page_number']}"
        if source_key not in seen:
            seen.add(source_key)
            print(f"  📄 {chunk['file_name']}")
            print(f"     Page : {chunk['page_number']}")
    print()


print("✅ Display function ready")
# %%
# ============================================================
# PHASE 7 - CELL 6 - Test memory with follow-up questions
# This is the key test — does system remember context?
# ============================================================

# Clear history first for clean test
clear_history()

# --- Turn 1 — First question ---
print("🔵 TURN 1")
print("─"*60)
r1 = get_answer("What is cognitive behavioral therapy?")
display_result(r1)

# --- Turn 2 — Follow-up using "it" ---
print("\n🔵 TURN 2 — Follow up")
print("─"*60)
r2 = get_answer("How effective is it for severe cases?")
display_result(r2)

# --- Turn 3 — Another follow-up ---
print("\n🔵 TURN 3 — Another follow up")
print("─"*60)
r3 = get_answer("What about relapse after treatment?")
display_result(r3)

# --- Turn 4 — General follow-up ---
print("\n🔵 TURN 4 — General follow up")
print("─"*60)
r4 = get_answer("Can you summarize what we discussed?")
display_result(r4)
# %%
# ============================================================
# PHASE 7 - CELL 7 - Show full chat history
# ============================================================

show_history()
# %%
