# QuranChatBot

A privacy‑focused Qur'an question answering system built on retrieval‑augmented generation (RAG). The bot uses
OpenAI's paid API for both embeddings and reasoning, while all data and vector search operations remain on your
own infrastructure via Qdrant. The pipeline is designed to run on commodity hardware and can be integrated
into existing web applications via a FastAPI endpoint or tested interactively via Gradio.

This project performs three core tasks:

1. **Qur'an semantic search:** high‑quality vector retrieval (Qdrant) with a cross‑encoder reranker. The index is
   built from the supplied dataset using OpenAI's multilingual embedding model (`text-embedding-3-small`).
2. **Structured query handling:** detect and execute requests such as specific surah ranges or ayah numbers
   without requiring embeddings.
3. **Pronunciation and reasoning via OpenAI:** if a pronunciation question is detected the system consults a word
   lexicon and, when necessary, asks OpenAI to generate a pronunciation guide. For general questions it calls
   OpenAI to provide a short, grounded commentary on the selected ayah and tafsir snippet.

The only external API calls are to OpenAI. All Qur'an data is stored locally and vector search is performed via
Qdrant. You control your API keys via environment variables and can inspect or audit all prompts.

---

## Features

### 1. Quran Semantic Search

* Uses OpenAI embeddings (`text-embedding-3-small`) to index all ayahs and words (1536‑dimensional vectors).
* Stores vectors in Qdrant for efficient semantic retrieval on your own hardware.
* Applies a cross‑encoder reranker (MiniLM) to refine the top results on CPU.
* Automatically selects the single best ayah and provides alternatives.

### 2. Structured Query Handling

Handles queries like:

* First 5 ayahs of Surah Baqarah
* Show Surah Ikhlas
* Surah 18 ayah 10

### 3. Pronunciation Support

The system uses a two‑stage pronunciation approach:

1. Lexicon‑based matching using the word‑level Qur'an dataset (fuzzy matching on transliteration and gloss).
2. If no high‑confidence match is found, the bot invokes OpenAI to generate an approximate pronunciation guide.

You can ask questions such as:

* How do I pronounce **bismillah**?
* How do you say **Allah** in Arabic?

### 4. LLM Based Commentary

OpenAI provides:

* A short explanation grounded in the selected ayah and tafsir snippet.
* No hallucinated ayahs and no invented rulings.
* Clear commentary in modern English or Urdu (based on the selected language).

---

# Project Structure

```
QuranChatBot/
│
├── app/
│   ├── gradio_ui.py                 Main UI and orchestrator
│
├── core/
│   ├── __init__.py                 Centralised settings loaded from environment
├── embeddings/
│   ├── openai_embedder.py          Batched OpenAI embeddings client
├── llm/
│   ├── openai_client.py            OpenAI commentary and pronunciation logic
│
├── router/
│   ├── router.py                    Intent detection (pronunciation, structured, semantic)
│   ├── catalog.py
│
├── search/
│   ├── retrieval.py                 Qdrant search, reranker, embeddings
│   ├── textnorm.py
│
├── indexing/
│   ├── index_ayahs.py               Build ayah vectors
│   ├── index_words.py               Build word level dataset
│
├── metadata/
│   ├── ayahs_collection.json        Embedded Quran text, translations, tafsir
│   ├── words_collection.json        Word level lexicon
│
├── app/
│   ├── fastapi_app.py              FastAPI server exposing JSON API
│   └── gradio_ui.py                 Main UI and orchestrator
├── docker-compose.yml               Runs Qdrant (vector DB)
└── requirements.txt                 Python dependencies
```

---

# Installation Guide

## 1. Install Dependencies

### Install Python 3.10 or later

Create a virtual environment:

```
python -m venv .venv
source .venv/bin/activate     (Linux or Mac)
.venv\Scripts\activate        (Windows)
```

Install required packages:

```
pip install -r requirements.txt
```

---

## 2. Install Docker

Download and install Docker Desktop:
[https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)

Make sure Docker is running before starting the system.

---

## 3. Configure OpenAI

This project relies on OpenAI's paid API for both embeddings and reasoning. You must provide your own
OpenAI API key via environment variables before indexing or serving:

1. Obtain an API key from [OpenAI](https://platform.openai.com/). Ensure the key has access to the
   `text-embedding-3-small` and `gpt-4o-mini` models.
2. Create a `.env` file in the project root (or export variables in your shell) with at least the following:

   ````
   OPENAI_API_KEY=<your-secret-key>
   OPENAI_MODEL=gpt-4o-mini
   OPENAI_EMBED_MODEL=text-embedding-3-small
   OPENAI_TIMEOUT_S=60
   # Optional overrides
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   QDRANT_COLLECTION_AYAHS=quran_ayahs_openai_v1
   QDRANT_COLLECTION_WORDS=quran_words_openai_v1
   ````

See `.env.example` for a complete list of available variables. Do **not** commit your `.env` file to version control.

---

# Running Qdrant with Docker

Your `docker-compose.yml` defines a single service for Qdrant:

```
version: "3.9"

services:
  qdrant:
    image: qdrant/qdrant:v1.9.1
    container_name: qdrant
    restart: unless-stopped
    ports:
      - "6333:6333"   # REST API
      - "6334:6334"   # gRPC API
    volumes:
      - qdrant_storage:/qdrant/storage
    environment:
      QDRANT__SERVICE__GRPC_PORT: 6334
      QDRANT__STORAGE__OPTIMIZERS__DEFAULT_SEGMENT_NUMBER: 1
      QDRANT__CLUSTER__ENABLED: "false"

volumes:
  qdrant_storage:
```

Start it:

```
docker compose up -d
```

Check containers:

```
docker ps
```

You should see qdrant running.

---

# 4. Index the Qur'an data

Before searching, you must embed the ayahs and words using OpenAI embeddings and upsert them into Qdrant. Ensure your
`OPENAI_API_KEY` is configured (see above) and Qdrant is running.

Run the indexing commands from the project root:

```
python -m indexing.index_ayahs
python -m indexing.index_words
```

These scripts will create new collections with names defined by your environment (defaults are
`quran_ayahs_openai_v1` and `quran_words_openai_v1`) and with vector dimension 1536. You can verify that the
collections exist with:

```
curl http://localhost:6333/collections
```

---


# 5. Run the application

You can interact with the bot via the provided Gradio UI or integrate it into a web app via FastAPI.

### Gradio (interactive testing)

Start the Gradio server:

```
python app/gradio_ui.py
```

By default the app listens on `http://127.0.0.1:7860`. It presents a simple UI where you can type questions,
choose a translation language (English, Urdu, Farsi or Pashto) and optionally include dataset tafsir.

### FastAPI (for integration)

The FastAPI server exposes two endpoints:

* `GET /health` → returns basic connectivity and collection status.
* `POST /query` → accepts a JSON body with `query`, `lang`, `top_k` and `show_tafsir` fields and returns a
  structured answer.

To run the FastAPI server:

```
uvicorn app.fastapi_app:app --host 0.0.0.0 --port 8000
```

Both servers rely on the same underlying retrieval and LLM logic, so you can prototype in Gradio and then
switch to FastAPI without changing any code.

---

# Usage Examples

Try asking:

* How do I pray
* What is the purpose of life
* Why were humans created
* Explain Surah Ikhlas
* First 3 ayahs of Surah Yaseen
* How do I pronounce bismillah
* How do you say Allah in Arabic

---

# Technical Overview

### Retrieval Layer

* OpenAI embeddings (`text-embedding-3-small`)
* Qdrant vector search
* Cross‑encoder reranker (CPU)

### Routing Layer

Detects intent categories:

1. Pronunciation
2. Structured surah or ayah range
3. Semantic free text
4. Pure Arabic word lookup
5. Simple numerical references

### LLM Layer
OpenAI provides:

* Commentary grounded in the retrieved ayah and optional tafsir snippet.
* No invented verses and no unsupported religious rulings.
* Short, pastoral, human‑friendly explanations in English or Urdu.

The generation model defaults to:

```
OPENAI_MODEL = gpt-4o-mini
```

You can override this (e.g. to use `gpt-3.5-turbo`) by setting the `OPENAI_MODEL` environment variable.

---

# Deployment Notes

The entire system can run:

* Locally on a laptop for development and testing.
* On any Linux server with Docker for Qdrant and Python for the application code.
* Without a GPU – all models run on CPU unless you choose to enable GPU for the reranker.

If you deploy remotely:

* Expose Gradio externally by setting `server_name="0.0.0.0"` in `gradio_ui.py` or by running the FastAPI server with `--host 0.0.0.0`.
* Restrict inbound ports so that only the required services are reachable (e.g. Qdrant on port 6333 and FastAPI on port 8000). There is no longer an Ollama service.
* Consider placing a reverse proxy (Nginx or Traefik) in front of FastAPI for TLS termination and rate limiting.

Resource considerations:

* **Qdrant:** low memory footprint; allocate ~1 GB RAM and some persistent storage for the vector index.
* **OpenAI:** network latency dictates response time. Ensure your server can reach the OpenAI API endpoints.
* **CPU/GPU:** all components run on CPU by default. If you wish to accelerate reranking, you may adjust the
  cross‑encoder to use a GPU, but this is optional.

---

# Security and Privacy

Although most of the pipeline runs locally, **the system sends user queries and limited context to OpenAI**
when generating embeddings and commentary. Consider the following when deploying:

* **API key handling:** store your `OPENAI_API_KEY` in environment variables or a secrets manager. Do not
  commit it to source control. Rotate keys periodically.
* **PII in queries:** discourage users from including personally identifiable information in their questions.
  Whatever is sent to OpenAI could be retained for a limited period per their policy.
* **Logging and redaction:** if you enable request logging, avoid logging full user prompts, tafsir excerpts
  or generated replies. Redact sensitive portions or hash queries before storage.
* **Prompt injection:** since OpenAI’s model executes whatever prompt you send, sanitise inputs and ensure
  that only the intended question, ayah and tafsir snippet are included. Do not pass arbitrary user data
  without filtering.
* **Data licensing:** the supplied Qur'an text, translations and tafsir are licensed from Tanzeel and other
  authors. See the `DATA_SOURCES.md` file (placeholder) for attribution details. Verify your intended
  usage complies with the source licences.

---

# Risks & Human Review Needed

The Qur'an chatbot provides informational guidance only. It is **not a mufti** and cannot issue
religious rulings. Humans must review and approve the following aspects before deploying publicly:

* **Generated commentary:** ensure that responses are grounded in the provided ayah and tafsir snippet and do
  not include fabricated content or rulings. Provide a disclaimer such as "For religious rulings, consult a
  qualified scholar" alongside responses.
* **Pronunciation guides:** approximate Latin spellings may vary; cross‑check with a qualified Arabic
  teacher.
* **Dataset integrity:** confirm that the Quran, translations and tafsir used are accurate and
  appropriately licensed. Review any modifications to the data.
* **Privacy policy:** clearly communicate that user queries will be sent to OpenAI. Obtain user consent if
  necessary.

---

# Known Limitations

* **External dependency:** if the OpenAI service is unavailable or rate‑limited, the bot will not be able
  to embed or generate commentary. Build caching and retries into your deployment to mitigate outages.
* **Latency:** calling OpenAI introduces network latency. Long prompts or slow internet connections will
  increase response times.
* **Approximate pronunciation:** the lexicon covers only Qur'anic vocabulary. Pronunciation guides for
  uncommon terms rely on the LLM and may not perfectly reflect classical recitation.
* **Reranker on CPU:** the cross‑encoder reranker runs on CPU by default. This can be slow for large
  candidate sets. You can configure it to run on a GPU if available.

---

# Future Improvements

* **Structured outputs:** leverage OpenAI's structured response format (JSON schema) to guarantee stable
  response shapes and extract key themes automatically.
* **Expanded language support:** add more translation and tafsir languages beyond English, Urdu, Farsi and
  Pashto.
* **Audio pronunciation:** integrate a text‑to‑speech model to provide audio recitation of selected ayahs and
  pronunciation guides.
* **User feedback loop:** allow users to rate answers and improve the reranking and generation algorithms.
