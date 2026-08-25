"""
Query Qdrant: embed query and retrieve top-k vectors with metadata and distances.

Usage:
  .\.venv\Scripts\python.exe app\embeddings\query_qdrant.py --query "refund policy" --k 5
"""
from pathlib import Path
import argparse

try:
    from sentence_transformers import SentenceTransformer
    have_st_model = True
except Exception:
    have_st_model = False

try:
    from qdrant_client import QdrantClient
    have_qdrant_client = True
except Exception:
    have_qdrant_client = False


def embed_query_local(query, model_name='all-MiniLM-L6-v2'):
    if not have_st_model:
        raise SystemExit('sentence-transformers required for local embeddings')
    model = SentenceTransformer(model_name)
    emb = model.encode([query], convert_to_numpy=True)
    return emb[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, required=True)
    parser.add_argument('--qdrant-host', type=str, default='localhost')
    parser.add_argument('--qdrant-port', type=int, default=6333)
    parser.add_argument('--collection', type=str, default='rag_docs')
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()

    if not have_qdrant_client:
        raise SystemExit('qdrant-client not installed. pip install qdrant-client')

    q = args.query
    q_emb = embed_query_local(q)

    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    # Try qdrant-client search methods first; if unavailable, fall back to direct HTTP REST call
    res = None
    try:
        if hasattr(client, 'search'):
            res = client.search(collection_name=args.collection, query_vector=q_emb.tolist(), limit=args.k, with_payload=True, with_vector=False)
        elif hasattr(client, 'search_points'):
            res = client.search_points(collection_name=args.collection, query_vector=q_emb.tolist(), limit=args.k, with_payload=True, with_vector=False)
    except Exception:
        res = None

    if res is None:
        # Fallback to REST API
        import requests
        url = f"http://{args.qdrant_host}:{args.qdrant_port}/collections/{args.collection}/points/search"
        body = {"vector": q_emb.tolist(), "limit": args.k, "with_payload": True}
        try:
            r = requests.post(url, json=body, timeout=10)
            r.raise_for_status()
            data = r.json()
            # Qdrant REST returns {'result': [...]} or {'result': {'points': [...]}}
            if 'result' in data and isinstance(data['result'], list):
                res_list = data['result']
            elif 'result' in data and isinstance(data['result'], dict) and 'points' in data['result']:
                res_list = data['result']['points']
            else:
                # older format
                res_list = data.get('result', [])
            # Normalize to list of dicts with id, payload, score
            hits = []
            for item in res_list:
                # item might have 'id','payload','score'
                hits.append(item)
        except Exception as e:
            raise SystemExit(f'Qdrant REST search failed: {e}')
    else:
        # res may be an object or dict; try to normalize to list
        try:
            # if res is dict with 'result' key
            if isinstance(res, dict) and 'result' in res:
                hits = res['result']
            else:
                hits = list(res)
        except Exception:
            hits = list(res)

    # Print hits
    for i, hit in enumerate(hits, start=1):
        # hit may be an object with attributes or a dict
        if hasattr(hit, 'payload') or hasattr(hit, 'id'):
            payload = getattr(hit, 'payload', None) or {}
            hit_id = getattr(hit, 'id', None)
            score = getattr(hit, 'score', None)
        else:
            payload = hit.get('payload', {}) if isinstance(hit, dict) else {}
            hit_id = hit.get('id') if isinstance(hit, dict) else None
            score = hit.get('score') if isinstance(hit, dict) else None

        print(f'Rank {i} — id={hit_id} score={score}')
        print('Source:', payload.get('source'))
        print('Page:', payload.get('page'))
        print('Chunk ID:', payload.get('chunk_id'))
        text = payload.get('text', '')
        print('Snippet:', (text[:400] + '...') if len(text) > 400 else text)
        print('---')


if __name__ == '__main__':
    main()
