from supabase import create_client, Client
import os

##################
from PyPDF2 import PdfReader

reader = PdfReader("/Users/gurdeepsingh/Desktop/learning/baobaopedia_v1/data/Pregnancy_Book_comp.pdf")
text = ""
i=0
for page in reader.pages:
    i+=1
    if i >= 6 and i <= 225:
        text += page.extract_text()
    if i > 225:
        break

import re

def clean(text: str) -> str:
    # 1. De-hyphenate line breaks: "matern-\nity" → "maternity"
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    # 2. Replace single newlines (inside paragraphs) with spaces
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    # 3. Normalize multiple blank lines to one paragraph break
    text = re.sub(r'\n{2,}', '\n\n', text)
    # 4. Collapse multiple spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # or 1000 if paragraphs are long
    chunk_overlap=250,   # 10–20% overlap helps retain context
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = text_splitter.split_text(clean(text))

from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
chunk_embeddings = embedder.encode(chunks, batch_size=16, show_progress_bar=True)

# Save embeddings + text into a local Chroma DB, then query it
import numpy as np
import chromadb

# 1) Normalize embeddings (good for cosine similarity)
E = np.asarray(chunk_embeddings, dtype=np.float32)
norms = np.linalg.norm(E, axis=1, keepdims=True)
E = E / np.clip(norms, 1e-12, None)
##################

# SUPABASE_URL = "https://pdelavjmlhvkzsgoirkx.supabase.co"
# SUPABASE_KEY = "sb_publishable_ujqtrfAA7R-irGdU4xZGKA_qOBYQZtu"  # better: use env var

def load_secrets(path="secrets.txt"):
    secrets = {}
    with open(path, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                secrets[key] = value.strip().strip('"').strip("'")
    return secrets

secrets = load_secrets()
SUPABASE_URL = secrets.get("SUPABASE_URL")
SUPABASE_KEY = secrets.get("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

ids = [f"p6to11-chunk-{i}" for i in range(len(chunks))]
metas = [{"source": "Pregnancy_Book_comp.pdf", "page_range": "6-11", "chunk_index": i}
         for i in range(len(chunks))]

rows = []
for i, (chunk, emb, meta) in enumerate(zip(chunks, E, metas)):
    # Remove null bytes (and optionally other control chars)
    safe_chunk = chunk.replace("\x00", "")  # minimal fix

    # Or more aggressive cleanup:
    # safe_chunk = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", chunk)

    rows.append({
        "chunk_id": ids[i],
        "source": meta["source"],
        "page_range": meta["page_range"],
        "chunk_index": meta["chunk_index"],
        "content": safe_chunk,
        "embedding": emb.tolist(),
    })


# simple insert (for big datasets you'd batch)
resp = supabase.table("pregnancy_chunks").insert(rows).execute()