from PyPDF2 import PdfReader

reader = PdfReader("/Users/gurdeepsingh/Desktop/learning/baobaopedia_v1/data/Pregnancy_Book_comp.pdf")
text = ""
i=0
for page in reader.pages:
    i+=1
    if i >= 6 and i <= 20:
        text += page.extract_text()
    if i > 11:
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

# print(clean(text))
# print(chunks)

# print(f"Number of chunks: {len(chunks)}")
# print(chunks[0])

from sentence_transformers import SentenceTransformer

# embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
chunk_embeddings = embedder.encode(chunks, batch_size=16, show_progress_bar=True)

# print(len(chunk_embeddings))
# print(chunk_embeddings[0])

# from langchain.vectorstores import FAISS
# db = FAISS.from_texts(chunks, embedder)

# query = "What should I do if I feel nervous during pregnancy?"
# results = db.similarity_search(query, k=3)

# print(results)

# --- add below your existing code ---

# Save embeddings + text into a local Chroma DB, then query it
import numpy as np
import chromadb

# 1) Normalize embeddings (good for cosine similarity)
E = np.asarray(chunk_embeddings, dtype=np.float32)
norms = np.linalg.norm(E, axis=1, keepdims=True)
E = E / np.clip(norms, 1e-12, None)

# 2) Create/open a persistent Chroma collection on disk
client = chromadb.PersistentClient(path="./chroma")  # folder created next to your script
# We’ll store *your* embeddings, so we do NOT pass an embedding_function here.
# Also set cosine distance for HNSW index:
coll = client.get_or_create_collection(
    name="pregnancy_book_p6_11",
    metadata={"hnsw:space": "cosine"}  # ensures cosine similarity
)

# 3) Prepare IDs + metadata (tweak as you like)
ids = [f"p6to11-chunk-{i}" for i in range(len(chunks))]
metas = [{"source": "Pregnancy_Book_comp.pdf", "page_range": "6-11", "chunk_index": i} for i in range(len(chunks))]

# 4) Upsert (store texts, embeddings, and metadata)
coll.upsert(
    ids=ids,
    documents=chunks,                 # your chunk texts
    embeddings=E.tolist(),            # normalized vectors
    metadatas=metas
)

print(f"Indexed {len(chunks)} chunks into Chroma at ./chroma")

# 5) Example: query the DB with a user question
question = "What should I expect in 18th week?"
q_vec = embedder.encode([question], normalize_embeddings=True).astype(np.float32)  # single (1, 384)

res = coll.query(
    query_embeddings=q_vec.tolist(),  # because we supplied our own embeddings
    n_results=2
)

# 6) Show top matches
txt = ' '
print("\nTop matches:")
for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
    # preview = (doc[:200] + "…") if len(doc) > 200 else doc
    preview = doc
    print(f"- {meta} → {preview}")
    print("======================\n")
    txt += str({preview})
    txt += ' '

from transformers import pipeline

question_answerer = pipeline("question-answering")
print(question_answerer(
    question="What should I expect in 18th week?",
    context=txt,
))

# # =============

# # After your `res = coll.query(..., n_results=25)` step:
# from sentence_transformers import CrossEncoder

# question = "How to know if it's boy or girl?"
# # Pull more candidates
# res = coll.query(query_embeddings=q_vec.tolist(), n_results=25)
# cands = res["documents"][0]
# metas = res["metadatas"][0]

# # 1) Lightweight keyword boost (optional but helps)
# boost_terms = ["boy", "girl"]
# boost = [sum(t.lower() in d.lower() for t in boost_terms) for d in cands]

# # 2) Cross-encoder rerank
# reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")
# scores = reranker.predict([(question, d) for d in cands])

# # Combine cross-encoder score + tiny keyword boost
# import numpy as np
# final_scores = np.array(scores) + 0.5 * np.array(boost)

# order = np.argsort(-final_scores)
# cands = [cands[i] for i in order]
# metas = [metas[i] for i in order]

# # 3) Extract a direct answer line if present
# import re
# answer = None
# for d in cands[:5]:
#     m = re.search(r"(How to know if it's boy or girl.*?)(?:\n|$)", d, flags=re.I)
#     if m:
#         answer = m.group(1)
#         break

# print("Answer:", answer or "Could not find an explicit line; showing top excerpt:\n" + cands[0][:400])

# # Optionally also print the source meta to cite where it came from
# print("Source:", metas[0])