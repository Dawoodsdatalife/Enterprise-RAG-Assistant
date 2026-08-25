"""
Query the Chroma collection and print top-k results with metadata and distances.

Usage:
  .\.venv\Scripts\python.exe app\embeddings\query_chroma.py --query "refund policy" --chroma-dir ./chroma_db
"""
from pathlib import Path
import argparse

try:
    from sentence_transformers import SentenceTransformer
    have_st_model = True
except Exception:
    have_st_model = False

import chromadb
from chromadb.config import Settings


def embed_query(model, text):
    return model.encode([text], convert_to_numpy=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, required=True)
    parser.add_argument('--chroma-dir', type=str, default='./chroma_db')
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()

    if not have_st_model:
        raise SystemExit('sentence-transformers is required for local query embedding; install sentence-transformers')

    model = SentenceTransformer('all-MiniLM-L6-v2')
    q_emb = embed_query(model, args.query)

    chroma_dir_path = str(Path(args.chroma_dir).resolve())
    try:
        # Prefer constructor that accepts persist_directory (avoid specifying duckdb+parquet)
        client = chromadb.Client(persist_directory=chroma_dir_path)
        try:
            collection = client.get_or_create_collection(name='rag_docs', metadata={'source': 'local_pdf'})
        except Exception:
            try:
                collection = client.get_collection('rag_docs')
            except Exception:
                collection = client.create_collection(name='rag_docs')
    except TypeError:
        try:
            # Try Settings with only persist_directory
            client = chromadb.Client(Settings(persist_directory=chroma_dir_path))
            try:
                collection = client.get_or_create_collection(name='rag_docs', metadata={'source': 'local_pdf'})
            except Exception:
                try:
                    collection = client.get_collection('rag_docs')
                except Exception:
                    collection = client.create_collection(name='rag_docs')
        except Exception:
            print('Warning: installed chromadb rejected persistent settings; falling back to default in-memory client')
            client= chromadb.Client()
            try:
                collection = client.get_collection('rag_docs')
            except Exception:
                collection = client.create_collection(name='rag_docs')

    # note: chroma query expects a list of query_embeddings (2D)
    results = collection.query(query_embeddings=q_emb.tolist(), n_results=args.k, include=['metadatas','documents','distances'])

    # results is a dict with keys 'ids', 'documents', 'metadatas', 'distances'
    for i, (doc, meta, dist) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0]), start=1):
        print(f'Rank {i} — distance {dist:.4f}')
        print('Source:', meta.get('source'))
        print('Page:', meta.get('page'))
        print('Chunk ID:', meta.get('chunk_id'))
        print('Snippet:', (doc[:400] + '...') if len(doc) > 400 else doc)
        print('---')


if __name__ == '__main__':
    main()
