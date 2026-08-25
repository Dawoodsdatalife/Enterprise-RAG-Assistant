"""
Simple RAG answerer: retrieve top-k chunks from Chroma and ask OpenAI to answer using only those chunks.

Usage:
  set OPENAI_API_KEY=...
  .venv/Scripts/python.exe app/retrieval/rag_answer.py --query "What is the refund policy for enterprise customers?" --k 5

Behavior and guardrails:
- The script retrieves top-k chunks from the Chroma collection (or in-memory fallback).
- It builds a context payload listing each chunk with metadata (source, page, chunk_id).
- It instructs the LLM to answer using ONLY that context and to output a final JSON array of cited chunk_ids in a line starting with "CITATIONS:"
- It verifies the returned citations are a subset of the retrieved chunk ids. If the model cites anything outside retrieved ids, it flags the answer as unreliable.

Requirements:
- chromadb and sentence-transformers installed (for local embeddings and retrieval)
- OpenAI API key set in OPENAI_API_KEY environment variable

Note: This is a minimal guardrail. For stronger safety, integrate Nemo Guardrails or perform automated claim-checking steps.
"""
from pathlib import Path
import argparse
import os
import re
import json
from dotenv import load_dotenv
load_dotenv()

try:
    from sentence_transformers import SentenceTransformer
    have_st = True
except Exception:
    have_st = False

import chromadb
from chromadb.config import Settings

try:
    import openai
except Exception:
    openai = None

# Local generation via transformers (text2text)
try:
    from transformers import pipeline
    have_transformers = True
except Exception:
    have_transformers = False


def ask_local(messages, model_name='google/flan-t5-small'):
    if not have_transformers:
        raise SystemExit('transformers library is required for local generation. Install transformers and torch.')
    # concatenate system and user prompts into a single instruction string
    system = ''
    user = ''
    for m in messages:
        if m.get('role') == 'system':
            system = m.get('content', '')
        elif m.get('role') == 'user':
            user = m.get('content', '')
    prompt = system + "\n\n" + user
    generator = pipeline('text2text-generation', model=model_name, device=-1)
    out = generator(prompt, max_length=512, do_sample=False)
    if isinstance(out, list) and 'generated_text' in out[0]:
        return out[0]['generated_text']
    # some pipelines return 'generated_text' key
    return out[0].get('generated_text') if isinstance(out, list) else str(out)


def get_chroma_collection(chroma_dir: str):
    # Try persistent client config first, fall back to in-memory
    chroma_dir_path = str(Path(chroma_dir).resolve())
    try:
        client = chromadb.Client(persist_directory=chroma_dir_path)
        try:
            collection = client.get_collection('rag_docs')
        except Exception:
            collection = client.get_or_create_collection(name='rag_docs')
    except TypeError:
        try:
            client = chromadb.Client(Settings(persist_directory=chroma_dir_path))
            try:
                collection = client.get_collection('rag_docs')
            except Exception:
                collection = client.get_or_create_collection(name='rag_docs')
        except Exception:
            print('Warning: falling back to default chromadb client (in-memory).')
            client = chromadb.Client()
            try:
                collection = client.get_collection('rag_docs')
            except Exception:
                raise SystemExit('No chroma collection named "rag_docs" found in this client. Run the embedding script first.')
    return client, collection


def embed_query(text: str, model_name='all-MiniLM-L6-v2'):
    if not have_st:
        raise SystemExit('sentence-transformers is required to embed queries locally. Install sentence-transformers.')
    model = SentenceTransformer(model_name)
    emb = model.encode([text], convert_to_numpy=True)
    return emb


def build_prompt(question: str, retrieved):
    # retrieved: list of dicts with id, document, metadata
    context_blocks = []
    retrieved_ids = []
    for i, (doc, meta) in enumerate(retrieved, start=1):
        cid = meta.get('chunk_id') or meta.get('id') or f'chunk_{i}'
        retrieved_ids.append(cid)
        src = meta.get('source', 'unknown')
        page = meta.get('page', 'unknown')
        block = f"[CHUNK_ID:{cid}] [SOURCE:{src}] [PAGE:{page}]\n{doc.strip()}"
        context_blocks.append(block)

    context = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are an assistant that answers questions using ONLY the provided CONTEXT blocks.\n"
        "If the answer is not contained in the context, say 'I don't know' or 'Not enough information'.\n"
        "Provide concise, factual answers. At the end include two lines: first a line that starts with 'CITATIONS:' followed by a JSON array of the CHUNK_IDs you used as evidence, and second a line that starts with 'SUPPORTS:' followed by a JSON object mapping each CHUNK_ID to the exact supporting quote (a short excerpt) you used from that chunk.\n"
        "Do NOT invent sources or quotes. Do NOT reference anything outside the provided context blocks.\n"
    )

    user_prompt = (
        f"QUESTION:\n{question}\n\n"
        "CONTEXT:\n"
        f"{context}\n\n"
        "INSTRUCTIONS:\n"
        "Answer the question concisely using only the context. Provide exact citations by chunk id in the final 'CITATIONS' JSON array. Then include a 'SUPPORTS' JSON object mapping each cited chunk id to the exact short quote you relied on from that chunk.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return messages, retrieved_ids


def extract_citations_from_text(text: str):
    # Extract CITATIONS: [..] and SUPPORTS: {...}
    cit_m = re.search(r"CITATIONS:\s*(\[.*?\])", text, flags=re.DOTALL)
    sup_m = re.search(r"SUPPORTS:\s*(\{[\s\S]*?\})", text, flags=re.DOTALL)

    cited = None
    supports = None
    if cit_m:
        try:
            arr = json.loads(cit_m.group(1))
            if isinstance(arr, list):
                cited = arr
        except Exception:
            s = cit_m.group(1).strip("[] \n\r")
            parts = [p.strip().strip('"') for p in s.split(',') if p.strip()]
            cited = parts

    if sup_m:
        try:
            obj = json.loads(sup_m.group(1))
            if isinstance(obj, dict):
                supports = obj
        except Exception:
            # naive fallback: parse simple """ 'id':'quote' """
            raw = sup_m.group(1).strip('{} \n\r')
            pairs = [p.strip() for p in raw.split(',') if ':' in p]
            d = {}
            for p in pairs:
                k, v = p.split(':', 1)
                k = k.strip().strip('"\'')
                v = v.strip().strip('"\'')
                d[k] = v
            supports = d

    return cited, supports


def verify_citations(cited, supports, retrieved_map):
    # retrieved_map: dict chunk_id -> text
    if cited is None:
        return False, 'No citation block found in model response.'
    unknown = [c for c in cited if c not in retrieved_map]
    if unknown:
        return False, f'Model cited unknown chunk ids: {unknown}'
    # verify supports exist and are substrings of the corresponding chunk text
    if supports is None:
        return False, 'No SUPPORTS block found in model response.'
    mismatches = []
    for cid, quote in supports.items():
        chunk_text = retrieved_map.get(cid, '')
        if quote is None or quote.strip() == '':
            mismatches.append((cid, 'empty quote'))
            continue
        # normalize
        norm_chunk = ' '.join(chunk_text.split()).lower()
        norm_quote = ' '.join(quote.split()).lower()
        if norm_quote not in norm_chunk:
            mismatches.append((cid, quote))
    if mismatches:
        return False, f'Quote verification failed for: {mismatches}'
    return True, 'All citations and supporting quotes validated.'


def ask_openai(messages, model='gpt-3.5-turbo'):
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise SystemExit('OPENAI_API_KEY not set in environment')
    openai.api_key = api_key
    resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=0, max_tokens=800)
    text = resp['choices'][0]['message']['content']
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, required=True)
    parser.add_argument('--chroma-dir', type=str, default='./chroma_db')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--provider', choices=['local','openai'], default='local', help='Which generation provider to use. Default: local (transformers).')
    args = parser.parse_args()

    client, collection = get_chroma_collection(args.chroma_dir)

    # embed query using same model used for indexing (sentence-transformers default)
    q_emb = embed_query(args.query)
    # Query chroma
    results = collection.query(query_embeddings=q_emb.tolist(), n_results=args.k, include=['documents','metadatas','distances'])

    docs = results.get('documents', [[]])[0]
    metadatas = results.get('metadatas', [[]])[0]

    retrieved = list(zip(docs, metadatas))

    print(f'Retrieved {len(retrieved)} chunks:')
    for i, (doc, meta) in enumerate(retrieved, start=1):
        print(f"{i}. {meta.get('source')} p{meta.get('page')} id={meta.get('chunk_id')}")

    messages, retrieved_ids = build_prompt(args.query, retrieved)

    print('\nAsking LLM to generate an answer (Guardrail: must cite chunk IDs)')

    answer = None
    if args.provider == 'openai':
        try:
            answer = ask_openai(messages)
        except Exception as e:
            emsg = str(e)
            is_rate = False
            if 'RateLimit' in emsg or 'rate limit' in emsg.lower() or 'quota' in emsg.lower():
                is_rate = True
            if is_rate:
                print('Warning: OpenAI rate limit or quota exceeded. Falling back to local generation if available, else returning retrieved snippets.')
            else:
                print('Warning: OpenAI call failed:', e)
                print('Falling back to local generation if available, else returning retrieved snippets.')
            # try local generation as fallback
            try:
                answer = ask_local(messages)
            except Exception as e2:
                print('Local generation failed as fallback:', e2)
                # Build a safe fallback answer using retrieved chunks
                supports_fallback = {}
                fallback_citations = []
                for doc, meta in retrieved:
                    cid = meta.get('chunk_id') or meta.get('id') or None
                    if cid:
                        fallback_citations.append(cid)
                        supports_fallback[cid] = doc.strip()[:400]
                answer = (
                    'OpenAI unavailable and local generation failed. Providing retrieved passages as a safe fallback.\n\n'
                    f"CITATIONS: {json.dumps(fallback_citations)}\n"
                    f"SUPPORTS: {json.dumps(supports_fallback)}\n"
                )
    else:
        # local provider (default)
        try:
            answer = ask_local(messages)
        except Exception as e:
            print('Local generation failed:', e)
            print('Falling back to returning retrieved snippets as the answer.')
            supports_fallback = {}
            fallback_citations = []
            for doc, meta in retrieved:
                cid = meta.get('chunk_id') or meta.get('id') or None
                if cid:
                    fallback_citations.append(cid)
                    supports_fallback[cid] = doc.strip()[:400]
            answer = (
                'Local generation failed. Providing retrieved passages as a safe fallback.\n\n'
                f"CITATIONS: {json.dumps(fallback_citations)}\n"
                f"SUPPORTS: {json.dumps(supports_fallback)}\n"
            )

    print('\n=== MODEL ANSWER ===')
    print(answer)

    cited, supports = extract_citations_from_text(answer)
    # build retrieved_map: chunk_id -> document text
    retrieved_map = {}
    for doc, meta in retrieved:
        cid = meta.get('chunk_id') or meta.get('id')
        if cid:
            retrieved_map[cid] = doc

    ok, reason = verify_citations(cited, supports, retrieved_map)
    print('\nCitation & quote verification:', ok, '-', reason)

    if not ok:
        print('\nGuardrail triggered: citations or supporting quotes invalid. Returning failure.')
        return

    print('\nFinal answer accepted. Cited chunk ids:', cited)
    print('Supporting quotes:', supports)


if __name__ == '__main__':
    main()
