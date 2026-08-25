"""
Embed PDF chunks and store into Chroma with metadata.

Usage:
  .\.venv\Scripts\python.exe app\embeddings\embed_and_store_chroma.py --pdf data\raw\test.pdf

This script:
- extracts text from the provided PDF (page-wise)
- chunks text by double-newline
- embeds chunks using sentence-transformers ('all-MiniLM-L6-v2')
- upserts into a local Chroma collection with metadata (source, page, chunk_id)

Swap to OpenAI embeddings by passing --openai (requires OPENAI_API_KEY in env).
"""
from pathlib import Path
import argparse
import uuid

# choose extractor: pypdf is pure-python and robust for text-based PDFs
from pypdf import PdfReader

# embedding options
try:
    from sentence_transformers import SentenceTransformer
    have_st_model = True
except Exception:
    have_st_model = False

import os
import chromadb
from chromadb.config import Settings


def extract_chunks_from_pdf(pdf_path: Path, min_chars: int = 50):
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({'page': i + 1, 'text': text})

    chunks = []
    for p in pages:
        parts = [part.strip() for part in p['text'].split('\n\n') if part.strip()]
        for idx, part in enumerate(parts):
            if len(part) < min_chars:
                continue
            chunk_id = f"{pdf_path.stem}_p{p['page']}_c{idx}"
            chunks.append({'id': chunk_id, 'text': part, 'page': p['page'], 'source': str(pdf_path)})
    return chunks


def embed_texts_sentence_transformers(texts, model_name='all-MiniLM-L6-v2'):
    if not have_st_model:
        raise RuntimeError('sentence-transformers not available in the environment')
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', type=str, required=True, help='Path to PDF to ingest')
    parser.add_argument('--chroma-dir', type=str, default='./chroma_db', help='Chroma persist directory')
    parser.add_argument('--openai', action='store_true', help='Use OpenAI embeddings instead of sentence-transformers')
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f'PDF not found: {pdf_path}')

    print('Extracting chunks from PDF...')
    chunks = extract_chunks_from_pdf(pdf_path)
    print(f'Found {len(chunks)} chunks')

    texts = [c['text'] for c in chunks]
    ids = [c['id'] for c in chunks]
    metadatas = [{'source': c['source'], 'page': c['page'], 'chunk_id': c['id']} for c in chunks]

    if len(texts) == 0:
        raise SystemExit('No chunks to embed; check PDF extraction or lower min_chars')

    print('Creating embeddings...')
    if args.openai:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise SystemExit('OPENAI_API_KEY is not set in environment')
        model_name = 'text-embedding-3-small'
        embeddings = []
        # Support both new openai (>=1.0) and legacy (<1.0)
        try:
            # new OpenAI client
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.embeddings.create(model=model_name, input=texts)
            embeddings = [d.embedding for d in resp.data]
        except Exception:
            # fallback to legacy openai API
            import openai
            openai.api_key = api_key
            embeddings = []
            for t in texts:
                resp = openai.Embedding.create(input=t, model=model_name)
                embeddings.append(resp['data'][0]['embedding'])
    else:
        embeddings = embed_texts_sentence_transformers(texts)

    print('Connecting to Chroma (persistent preferred, no duckdb+parquet)...')
    client = None
    collection = None
    chroma_dir_path = str(Path(args.chroma_dir).resolve())
    # Try multiple client constructors that support persistence but avoid specifying duckdb+parquet explicitly
    try:
        # Newer chromadb may accept persist_directory as kwarg
        client = chromadb.Client(persist_directory=chroma_dir_path)
        try:
            collection = client.get_or_create_collection(name='rag_docs', metadata={'source': 'local_pdf'})
        except Exception:
            try:
                collection = client.get_collection('rag_docs')
            except Exception:
                collection = client.create_collection(name='rag_docs', metadata={'source': 'local_pdf'})
    except TypeError:
        try:
            # Try Settings with just persist_directory (avoid specifying db impl)
            client = chromadb.Client(Settings(persist_directory=chroma_dir_path))
            try:
                collection = client.get_or_create_collection(name='rag_docs', metadata={'source': 'local_pdf'})
            except Exception:
                try:
                    collection = client.get_collection('rag_docs')
                except Exception:
                    collection = client.create_collection(name='rag_docs', metadata={'source': 'local_pdf'})
        except Exception:
            print('Warning: could not create a persistent chromadb client; falling back to in-memory client.')
            client = chromadb.Client()
            try:
                collection = client.get_collection('rag_docs')
            except Exception:
                collection = client.create_collection(name='rag_docs', metadata={'source': 'local_pdf'})

    print('Upserting to Chroma...')
    collection.upsert(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
    )

    # Attempt to persist the client/collection if supported by this chromadb build.
    if hasattr(client, 'persist'):
        try:
            client.persist()
        except Exception as e:
            print('Warning: client.persist() failed:', e)
    elif hasattr(collection, 'persist'):
        try:
            collection.persist()
        except Exception as e:
            print('Warning: collection.persist() failed:', e)
    else:
        print('Note: chromadb persistence method not available in this installation. Data may not be persisted across runs.')

    print('Upsert complete. Collection size:', collection.count())


if __name__ == '__main__':
    main()
