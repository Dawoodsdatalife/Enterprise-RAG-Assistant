# Enterprise RAG Assistant with Citation & Guardrails

This repository is a learning-first scaffold for building an enterprise Retrieval-Augmented Generation (RAG) assistant with source citations and guardrails.

Quickstart (local development)

1. Create and activate Python virtual environment

   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

2. Install dependencies

   .\.venv\Scripts\python.exe -m pip install -r requirements.txt

3. (Optional) Set HF & OpenAI tokens

   $env:HF_TOKEN = 'hf_...'
   $env:OPENAI_API_KEY = 'sk-...'

4. Start Qdrant (recommended persistent store)

   docker-compose up -d

5. Embed & upsert a PDF into Qdrant (local embeddings)

   .\.venv\Scripts\python.exe app\embeddings\embed_and_store_qdrant.py --pdf data\raw\test_1.pdf

   To use OpenAI embeddings instead:
   .\.venv\Scripts\python.exe app\embeddings\embed_and_store_qdrant.py --pdf data\raw\test_1.pdf --openai

6. Query Qdrant

   .\.venv\Scripts\python.exe app\embeddings\query_qdrant.py --query "enterprise refund policy" --k 5

7. Run RAG answerer

   By default the RAG answerer uses a local transformer model (no OpenAI key required). Run locally with:

   .\.venv\Scripts\python.exe app\retrieval\rag_answer.py --query "What is the refund policy for enterprise customers?" --k 5

   To explicitly use OpenAI (requires OPENAI_API_KEY):

   .\.venv\Scripts\python.exe app\retrieval\rag_answer.py --query "What is the refund policy for enterprise customers?" --k 5 --provider openai

Notes

- Qdrant listens on port 6333 by default; storage is persisted to ./qdrant_storage by docker-compose.
- For production, secure the Qdrant instance and avoid exposing it publicly without authentication.
- See docs/ for step-by-step explanations and further improvements including guardrails integration.
