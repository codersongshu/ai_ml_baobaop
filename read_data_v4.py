import os
import numpy as np
from supabase import create_client, Client
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# ---------- 1. Setup: Supabase + Embeddings + Llama ----------

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

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

HF_TOKEN = secrets.get("HF_TOKEN")

llm_client = InferenceClient(
    model="meta-llama/Llama-3.1-8B-Instruct",
    token=HF_TOKEN,
)

# ---------- 2. Retrieval from Supabase (pgvector) ----------

def retrieve_context_from_supabase(question: str, k: int = 5) -> tuple[str, list[dict]]:
    """
    Embed the question, query Supabase via pgvector, and return:
      - context string (joined top-k chunks)
      - list of rows (for debugging / display)
    """
    # 1) Embed the question
    q_vec = embedder.encode([question], normalize_embeddings=True).astype(np.float32)[0]

    # 2) Call the Postgres function via Supabase RPC
    # - Sends your **question embedding** (`q_vec`) to Supabase.
    # - Calls a **Postgres function** named `match_pregnancy_chunks`.
    # - Asks Postgres/pgvector to return the **top-k most similar chunks** from your stored embeddings.
    # - Returns the result to Python as `resp.data`.
    resp = supabase.rpc(
        "match_pregnancy_chunks",
        {
            "query_embedding": q_vec.tolist(),
            "match_count": k,
        },
    ).execute()

    rows = resp.data or []

    # 3) Build a context string from top-k chunks
    context_parts = []
    print("\nTop matches from Supabase:\n")
    for row in rows:
        meta = {
            "source": row.get("source"),
            "page_range": row.get("page_range"),
            "chunk_index": row.get("chunk_index"),
            "similarity": row.get("similarity"),
        }
        preview = row.get("content", "")
        print(f"- {meta} → {preview[:200]}...")
        print("======================\n")
        context_parts.append(preview)

    context = "\n\n".join(context_parts)
    return context, rows


# ---------- 3. Ask Llama with retrieved context ----------

def answer_question(question: str, k: int = 5) -> str:
    # 1) Retrieve context from Supabase
    context, _ = retrieve_context_from_supabase(question, k=k)

    if not context.strip():
        return "I couldn't find any relevant context in the knowledge base."

    # 2) Build prompt
    prompt = (
        "Answer the question based only on the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    # 3) Call Llama
    response = llm_client.chat_completion(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant who answers based only on the given context.",
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=300,
        temperature=0.3,
    )

    return response.choices[0].message["content"]


if __name__ == "__main__":
    # Simple interactive test
    question = "What are pain interventions?"
    answer = answer_question(question, k=5)
    print("\n=== FINAL ANSWER ===\n")
    print(answer)