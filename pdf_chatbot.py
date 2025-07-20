import os
import time
import shutil
import argparse
import pickle
import numpy as np
import faiss
import tiktoken
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
from PyPDF2 import PdfReader
from dotenv import load_dotenv
load_dotenv()

# ─── CONFIG ─────────────────────────────────────────────────────────────
EMBED_MODEL = "text-embedding-ada-002"
CHAT_MODEL  = "gpt-4o-mini"
TOKENIZER   = "cl100k_base"
MAX_TOK_PER_MIN = 40_000

client = OpenAI()


def open_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# ─── STEP 1: Chunk PDF into token windows ────────────────────────────────
def chunk_by_tokens(pdf_path: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    reader = PdfReader(pdf_path)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    enc = tiktoken.get_encoding(TOKENIZER)
    toks = enc.encode(text)

    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    for i in range(0, len(toks), step):
        slice_ = toks[i : i + chunk_size]
        chunks.append(enc.decode(slice_))
        if i + chunk_size >= len(toks):
            break
    return chunks

# ─── STEP 2: Prepare/override cache folder ───────────────────────────────
def prepare_cache_folder(pdf_path: str, base_folder: str = "Data") -> Path:
    pdf_name = Path(pdf_path).stem
    cache_dir = Path(base_folder) / f"{pdf_name}_cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir

# ─── STEP 3: Embed chunks with rate-limiting + tqdm ──────────────────────
def embed_chunks_with_rate_limit(
    chunks: list[str],
    model: str = EMBED_MODEL,
    max_tokens_per_minute: int = MAX_TOK_PER_MIN
) -> list[list[float]]:
    """
    Embed `chunks`, batching up to `max_tokens_per_minute` tokens per minute.
    """
    enc = tiktoken.get_encoding(TOKENIZER)
    all_embs: list[list[float]] = []
    window_start = time.time()
    batch: list[str] = []
    batch_tokens = 0

    def flush_batch():
        nonlocal window_start, batch, batch_tokens, all_embs
        if not batch:
            return
        elapsed = time.time() - window_start
        if elapsed < 60:
            time.sleep(60 - elapsed)
        resp = client.embeddings.create(input=batch, model=model)
        # resp.data is a list of objects, each with an .embedding attribute
        for item in resp.data:
            all_embs.append(item.embedding)
        window_start = time.time()
        batch.clear()
        batch_tokens = 0

    for chunk in chunks:
        tok_count = len(enc.encode(chunk))
        if tok_count > max_tokens_per_minute:
            raise ValueError(f"Chunk of {tok_count} tokens exceeds cap")
        if batch_tokens + tok_count > max_tokens_per_minute:
            flush_batch()
        batch.append(chunk)
        batch_tokens += tok_count

    # flush any remaining chunks
    flush_batch()
    return all_embs

# ─── STEP 4: Build FAISS index in memory ────────────────────────────────
def build_faiss_index(embeddings: list[list[float]]) -> faiss.IndexFlatL2:
    arr = np.array(embeddings, dtype=np.float32)
    index = faiss.IndexFlatL2(arr.shape[1])
    index.add(arr)
    return index

# ─── STEP 5: Retrieval + Chat helpers ───────────────────────────────────
def retrieve_chunks(
    query: str,
    index: faiss.IndexFlatL2,
    chunks: list[str],
    top_k: int = 5
) -> list[str]:
    # get the embedding using the .data list and .embedding field
    resp = client.embeddings.create(input=query, model=EMBED_MODEL)
    q_emb = resp.data[0].embedding  # was resp["data"][0]["embedding"]

    D, I = index.search(
        np.array(q_emb, dtype=np.float32).reshape(1, -1),
        top_k
    )
    return [chunks[i] for i in I[0]]

def answer_question(question: str, context: list[str]) -> str:
    prompt = "Use the following excerpts from the PDF to answer the question as comprehensively as possible:\n\n"
    for idx, excerpt in enumerate(context, 1):
        prompt += f"### Excerpt {idx}\n{excerpt}\n\n"
    prompt += f"### Question:\n{question}\n\n### Answer:"
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

# ─── STEP 6: Manage transcript files ─────────────────────────────────────
def get_next_transcript_path(pdf_name: str) -> Path:
    base = Path("Chatbot_Interactions") / pdf_name
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(base.glob("chat_*.txt"))
    used = {int(p.stem.split("_")[1]) for p in existing if "_" in p.stem}
    idx = 0
    while idx in used:
        idx += 1
    return base / f"chat_{idx}.txt"

# ─── MAIN ORCHESTRATION ─────────────────────────────────────────────────
def main():
    import argparse
    from pathlib import Path
    import pickle
    import numpy as np
    import shutil
    import tiktoken  # for token counting

    p = argparse.ArgumentParser(description="PDF → RAG Chatbot")
    p.add_argument("pdf_path", help="Path to your PDF file")
    args = p.parse_args()
    pdf_path = args.pdf_path
    pdf_name = Path(pdf_path).stem

    # determine cache folder
    cache_dir = Path("Data") / f"{pdf_name}_cache"

    # ── if cache exists, ask user what to do ────────────────────────────────
    if cache_dir.exists():
        while True:
            choice = input(
                f"A cache for '{pdf_name}' was found. What do you want to do?\n"
                "  [a] Query the existing model\n"
                "  [b] Create a new model (override cache)\n"
                "Select a or b: "
            ).strip().lower()
            if choice in ("a", "b"):
                break
            print("Please enter 'a' or 'b'.\n")

        if choice == "a":
            # load existing cache
            print("[→] Loading cached embeddings & chunks…")
            with open(cache_dir / "chunks.pkl", "rb") as f:
                chunks = pickle.load(f)
            embs = np.load(cache_dir / "embeddings.npy")
        else:
            # override old cache and retrain
            print("[→] Overriding existing cache and retraining…")
            shutil.rmtree(cache_dir)
            chunks = None
            embs = None
    else:
        chunks = None
        embs = None

    # ── CHUNK & EMBED if needed ───────────────────────────────────────────────
    if chunks is None or embs is None:
        # 1) Chunk & report total tokens
        print("[1/4] Chunking PDF…")
        full_text = open_pdf(pdf_path)
        enc = tiktoken.get_encoding("cl100k_base")
        total_tokens = len(enc.encode(full_text))
        chunks = chunk_by_tokens(pdf_path)
        print(f"→ Chunking complete: {len(chunks)} chunks; PDF has {total_tokens} tokens total.\n")

        # 2) Prepare cache dir & embed
        print("[2/4] Embedding chunks & caching…")
        cache_dir.mkdir(parents=True, exist_ok=True)
        embs = embed_chunks_with_rate_limit(chunks)
        np.save(cache_dir / "embeddings.npy", np.array(embs, dtype=np.float32))
        with open(cache_dir / "chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)
    else:
        print("[→] Skipping chunk + embed (using cache).")

    # ── BUILD INDEX ───────────────────────────────────────────────────────────
    print("[3/4] Building FAISS index…")
    index = build_faiss_index(embs)

    # ── INTERACTIVE CHAT WITH MEMORY ─────────────────────────────────────────
    # initialize conversation history
    history = [
        {"role": "system", "content": "You are a helpful assistant who has read the PDF."}
    ]
    print("[4/4] Ready! Ask questions (type 'exit' to quit).")
    transcript_file = get_next_transcript_path(pdf_name)
    transcript: list[tuple[str,str]] = []

    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break

        # retrieve relevant chunks
        ctx = retrieve_chunks(q, index, chunks, top_k=5)

        # add retrieved excerpts as a system message
        excerpts = "\n\n".join(f"Excerpt {i+1}:\n{txt}" for i, txt in enumerate(ctx))
        history.append({"role": "system", "content": excerpts})

        # add the user's question
        history.append({"role": "user", "content": q})

        # call the chat model with full history
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=history,
            temperature=0.2,
        )
        ans = resp.choices[0].message.content.strip()

        # output and record
        print("Bot:", ans, "\n")
        history.append({"role": "assistant", "content": ans})
        transcript.append((q, ans))

    # ── SAVE TRANSCRIPT ────────────────────────────────────────────────────────
    with open(transcript_file, "w") as f:
        for q, a in transcript:
            f.write(f"You: {q}\nBot: {a}\n\n")
    print(f"✅ Saved chat to {transcript_file}")


if __name__ == "__main__":
    main()