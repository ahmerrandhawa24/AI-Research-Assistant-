
# %%
import os
from dotenv import load_dotenv

load_dotenv()

# LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"

# Embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunking
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# Retrieval
TOP_K_RESULTS = 8

# PDF Processing
EXCLUDE_LAST_N_PAGES = 5        # ← add this line

# Folders
PDF_FOLDER = "data/pdfs"
VECTOR_DB_PATH = "data/vectordb"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

# ✅ ADD THIS PART
print("✅ Config loaded successfully")

if GROQ_API_KEY:
    print("🔑 API Key Loaded:", GROQ_API_KEY[:5] + "...")
else:
    print("❌ API Key NOT found")
# %%
