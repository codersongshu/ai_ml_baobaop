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
question = "What exercise is best druing second trimester"
q_vec = embedder.encode([question], normalize_embeddings=True).astype(np.float32)  # single (1, 384)

res = coll.query(
    query_embeddings=q_vec.tolist(),  # because we supplied our own embeddings
    n_results=5
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

# from transformers import pipeline

# question_answerer = pipeline("question-answering")
# print(question_answerer(
#     question="What should I expect in 18th week?",
#     context=txt,
# ))

##########################
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# MODEL = "meta-llama/Llama-3.1-8B-Instruct"  # pick any instruct model you can run
# tok = AutoTokenizer.from_pretrained(MODEL)
# mdl = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto")
# gen = pipeline("text-generation", model=mdl, tokenizer=tok)

# def rag_answer(question, k=5):
#     # ctx_chunks = retrieve(question, k=k)
#     # context = "\n\n".join([f"[{i+1}] {c}" for i, c in enumerate(ctx_chunks)])
#     context = txt
#     prompt = f"""Use ONLY the context to answer. If missing, say you don't know.
# Return: a concise summary first, then bullet points, then a short conclusion.
# Also include a "Sources" line listing chunk numbers you used.

# Context:
# \"\"\"{context}\"\"\"

# Question: {question}
# Answer:"""
#     out = gen(prompt, max_new_tokens=450, temperature=0.2, top_p=0.9)[0]["generated_text"]
#     return out.split("Answer:",1)[-1].strip()

# print(rag_answer(question))


# from huggingface_hub import InferenceClient

# client = InferenceClient(model="meta-llama/Llama-3.1-8B-Instruct", token="")

# question = "What are different tax credits?"
# # context = """
# # The placenta is an organ that develops in the uterus during pregnancy. It provides oxygen and nutrients to the growing baby
# # and removes waste products from the baby's blood.
# # """
# context = txt

# prompt = f"Answer the question based only on the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

# response = client.text_generation(prompt, max_new_tokens=200, temperature=0.3)
# print(response)

def load_secrets(path="secrets.txt"):
    secrets = {}
    with open(path, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                secrets[key] = value.strip().strip('"').strip("'")
    return secrets

secrets = load_secrets()
HF_TOKEN = secrets.get("HF_TOKEN")

from huggingface_hub import InferenceClient

client = InferenceClient(model="meta-llama/Llama-3.1-8B-Instruct", token=HF_TOKEN)

# question = "What are different tax credits?"
# context = """
# Tax credits reduce the amount of tax owed. Common types include child tax credit, earned income tax credit, education credits,
# and energy efficiency credits for homeowners.
# """

context = txt

prompt = f"Answer the question based only on the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

response = client.chat_completion(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant who answers based only on the given context."},
        {"role": "user", "content": prompt}
    ],
    max_tokens=300,
    temperature=0.3
)

print(response.choices[0].message["content"])


##########################

# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# MODEL = "mistralai/Mistral-7B-Instruct-v0.3"  # or: "google/flan-t5-large" (smaller, CPU OK)
# tok = AutoTokenizer.from_pretrained(MODEL)
# mdl = AutoModelForCausalLM.from_pretrained(
#     MODEL, device_map="auto"  # remove device_map if CPU only
# )

# generator = pipeline("text-generation", model=mdl, tokenizer=tok)

# def answer(question, context):
#     prompt = f"""You are a helpful assistant. Answer the question *only* using the context.
# If the answer is not in the context, say you don't know.

# Context:
# \"\"\"{context}\"\"\"

# Question: {question}
# Answer (be detailed, organized, and cite short quotes or line numbers from the context if helpful):"""
#     out = generator(
#         prompt,
#         max_new_tokens=350,
#         temperature=0.2,
#         top_p=0.9,
#         do_sample=True
#     )[0]["generated_text"]
#     return out.split("Answer", 1)[-1].strip()

# # example
# context = """Chapter 2: The placenta delivers oxygen and nutrients to the fetus...
# # Complications can arise if ... (p. 34)"""
# context = txt
# print(answer(question, context))