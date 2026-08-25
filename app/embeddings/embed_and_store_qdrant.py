"""
Embed PDF chunks and upsert into Qdrant with metadata.

Usage:
  Start Qdrant via Docker Compose: docker-compose up -d
  .\.venv\Scripts\python.exe app\embeddings\embed_and_store_qdrant.py --pdf data\raw\test_1.pdf

This script supports both OpenAI embeddings (if --openai) and local sentence-transformers.
It stores metadata: source, page, chunk_id.
"""
from pathlib import Path
import argparse
import os
from pypdf import PdfReader

try:
    from sentence_transformers import SentenceTransformer
    have_st_model = True
except Exception:
    have_st_model = False

try:
    # qdrant-client 1.x API
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as q_models
    have_qdrant_client = True
except Exception:
    have_qdrant_client = False


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
    parser.add_argument('--pdf', type=str, required=True)
    parser.add_argument('--qdrant-host', type=str, default='localhost')
    parser.add_argument('--qdrant-port', type=int, default=6333)
    parser.add_argument('--collection', type=str, default='rag_docs')
    parser.add_argument('--openai', action='store_true')
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f'PDF not found: {pdf_path}')

    print('Extracting chunks...')
    chunks = extract_chunks_from_pdf(pdf_path)
    print(f'Found {len(chunks)} chunks')

    texts = [c['text'] for c in chunks]
    ids = [c['id'] for c in chunks]
    metadatas = [{'source': c['source'], 'page': c['page'], 'chunk_id': c['id']} for c in chunks]

    if args.openai:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise SystemExit('OPENAI_API_KEY not set')
        # Use new OpenAI client if available
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.embeddings.create(model='text-embedding-3-small', input=texts)
            embeddings = [d.embedding for d in resp.data]
        except Exception:
            import openai
            openai.api_key = api_key
            embeddings = []
            for t in texts:
                resp = openai.Embedding.create(input=t, model='text-embedding-3-small')
                embeddings.append(resp['data'][0]['embedding'])
    else:
        embeddings = embed_texts_sentence_transformers(texts)

    if not have_qdrant_client:
        raise SystemExit('qdrant-client not installed. Install with: pip install qdrant-client')

    print('Connecting to Qdrant...')
    # suppress compatibility check warning if API versions differ
    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port, prefer_grpc=False, check_compatibility=False)

    # Create or get collection using qdrant models
    from qdrant_client.http.models import VectorParams, Distance
    try:
        client.get_collection(collection_name=args.collection)
    except Exception:
        vec_params = VectorParams(size=len(embeddings[0]), distance=Distance.COSINE)
        client.create_collection(collection_name=args.collection, vectors_config=vec_params)

    # Upsert points — Qdrant requires point IDs to be integers or UUIDs. Convert our string ids to deterministic UUIDs.
    import uuid as _uuid
    points = []
    for idx, (eid, emb, meta, text) in enumerate(zip(ids, embeddings, metadatas, texts)):
        # deterministic UUID derived from the chunk id string so re-running won't create duplicates
        uid = _uuid.uuid5(_uuid.NAMESPACE_URL, eid)
        p = q_models.PointStruct(id=uid, vector=emb, payload={**meta, 'text': text})
        points.append(p)

    client.upsert(collection_name=args.collection, points=points)
    print('Upsert complete. Count should be:', len(points))


if __name__ == '__main__':
    main()
