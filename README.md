# Hybrid RAG — FastAPI + Streamlit + FAISS + Neo4j Aura

A production-oriented **Hybrid Retrieval-Augmented Generation (RAG)** application combining **semantic vector retrieval** with **Knowledge Graph retrieval** to answer questions grounded in uploaded PDF documents.

**Production:** https://hybridragbot.sbs  
**GitHub:** https://github.com/mohanaraajanai/hybrid-rag-project

---

## 1. Project Overview

This project demonstrates a complete Hybrid RAG lifecycle:

```text
PDF
 ↓
Validation
 ↓
Extraction
 ↓
Chunking
 ↓
Local Embeddings
 ↓
FAISS Vector Index
 ↓
Neo4j Knowledge Graph
 ↓
Entity Extraction
 ↓
Hybrid Retrieval
 ↓
Context Building
 ↓
LLM Answer Generation
 ↓
Answer + Sources
 ↓
Streamlit UI
```

The system combines two complementary retrieval paths:

- **Vector retrieval:** semantic similarity using Sentence Transformers + FAISS.
- **Graph retrieval:** entity/chunk/relationship retrieval using Neo4j Aura.
- **Hybrid retrieval:** merges both results and deduplicates by stable `chunk_id`.
- **Grounded generation:** builds context from retrieved evidence before producing the answer.
- **Guardrails:** rejects invalid input such as empty queries and prompt-injection-style requests.
- **Document scoping:** graph retrieval is restricted to document IDs represented by the active FAISS metadata, preventing unrelated historical graph data from contaminating answers.

---

## 2. Architecture

```text
                         ┌─────────────────────────┐
                         │         User            │
                         │ Upload PDF / Ask Query  │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │    Streamlit Frontend   │
                         │ frontend/streamlit_app  │
                         └────────────┬────────────┘
                                      │ HTTP
                                      ▼
                         ┌─────────────────────────┐
                         │      FastAPI Backend    │
                         │       backend/app       │
                         └────────────┬────────────┘
                                      │
                 ┌────────────────────┴─────────────────────┐
                 │                                          │
                 ▼                                          ▼
        ┌──────────────────┐                       ┌──────────────────┐
        │   PDF Ingestion  │                       │   Query / RAG    │
        └────────┬─────────┘                       └────────┬─────────┘
                 │                                          │
                 ▼                                          ▼
        ┌──────────────────┐                       ┌──────────────────┐
        │ Validate / Hash  │                       │   Guardrails     │
        │ Duplicate Check  │                       └────────┬─────────┘
        │ Extract Text     │                                │
        └────────┬─────────┘                                ▼
                 │                               ┌──────────────────────┐
                 ▼                               │ Hybrid Retriever     │
        ┌──────────────────┐                      └──────────┬───────────┘
        │ Create Chunks    │                                 │
        └───────┬──────────┘                    ┌────────────┴────────────┐
                │                               │                         │
        ┌───────┴────────┐                      ▼                         ▼
        │                │               ┌──────────────┐       ┌────────────────┐
        ▼                ▼               │ FAISS Vector │       │ Neo4j Graph    │
    Embeddings       Neo4j Storage       │ Retrieval    │       │ Retrieval       │
        │                │               └──────┬───────┘       └───────┬────────┘
        ▼                ▼                      │                       │
      FAISS       Documents / Chunks /          └──────────┬────────────┘
                   Entities / Relations                     ▼
                                                 ┌──────────────────────┐
                                                 │ Context Builder      │
                                                 └──────────┬───────────┘
                                                            ▼
                                                 ┌──────────────────────┐
                                                 │ LLM Answer Generator │
                                                 └──────────┬───────────┘
                                                            ▼
                                                 ┌──────────────────────┐
                                                 │ Answer + Sources     │
                                                 └──────────────────────┘
```

---

## 3. Features

### PDF upload and validation
The Streamlit interface accepts PDF uploads. FastAPI validates the file, reads its bytes, computes a SHA-256 hash, performs duplicate detection, saves the PDF, and extracts page-level text.

### Duplicate detection
The registry uses a deterministic file hash to identify files that were already processed. This prevents blindly creating another document for the same uploaded PDF.

### PDF extraction
Text is extracted page-by-page using `pypdf` and persisted as processed JSON.

### Chunking
Extracted text is divided into chunks while retaining:

```text
document_id
filename
page_number
page_chunk_index
chunk_id
text
character_count
```

The same chunk IDs are used by the vector and graph layers so they can be correlated.

### Local embeddings
The validated embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The production test generated **384-dimensional** vectors.

### FAISS
FAISS stores dense vectors and metadata for semantic similarity search.

Runtime artifacts:

```text
data/vector_store/documents.faiss
data/vector_store/documents_metadata.json
```

### Neo4j Aura Knowledge Graph
The graph uses an isolated Hybrid RAG model containing:

```text
HybridDocument
HybridChunk
HybridEntity
```

Important relationships include:

```text
(:HybridDocument)-[:HAS_HYBRID_CHUNK]->(:HybridChunk)
(:HybridChunk)-[:MENTIONS_HYBRID_ENTITY]->(:HybridEntity)
```

### Entity and relationship extraction
Each chunk is processed by the LLM-based extraction component. Extracted entities and relationships are persisted in Neo4j.

### Hybrid retrieval
FAISS and Neo4j are queried together. Results are merged, source types are retained, and duplicate chunks are removed using `chunk_id`.

### Document-scoped graph retrieval
A production issue was found where Neo4j graph retrieval could return chunks from older documents. The final `graph_retriever.py` reads active document IDs from FAISS metadata and scopes graph retrieval to the same corpus.

### Grounded answer generation
The RAG layer builds context from retrieved chunks and sends that evidence to the LLM for answer generation.

### Source citations
The UI exposes a **View Sources** section showing page/source information and whether evidence came from vector or graph retrieval.

### Guardrails
Validated blocked cases include:

```text
empty query
prompt-injection-style request
```

### Evaluation
The completed evaluation run contained:

```text
9 total cases
9 passed
7 normal answer cases
2 blocked cases
Average keyword match: 1.0
```

---

## 4. Technology Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Backend API | FastAPI |
| ASGI server | Uvicorn |
| Frontend | Streamlit |
| PDF extraction | pypdf |
| Embeddings | Sentence Transformers |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector index | FAISS |
| Knowledge Graph | Neo4j Aura |
| Neo4j access | Neo4j Python driver |
| LLM | OpenAI |
| Containerization | Docker |
| Orchestration | Docker Compose |
| Reverse proxy | Traefik |
| SSL | Let's Encrypt |
| Testing | pytest and project-specific test scripts |

---

## 5. Repository Structure

```text
hybrid-rag-project/
│
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── query.py
│   │   │   └── upload.py
│   │   │
│   │   ├── core/
│   │   │   └── __init__.py
│   │   │
│   │   ├── document_registry.py
│   │   │
│   │   ├── embeddings/
│   │   │   ├── __init__.py
│   │   │   └── local_embeddings.py
│   │   │
│   │   ├── generation/
│   │   │   └── __init__.py
│   │   │
│   │   ├── ingestion/
│   │   │   ├── __init__.py
│   │   │   ├── chunker.py
│   │   │   ├── document_processing_service.py
│   │   │   ├── metadata.py
│   │   │   ├── pdf_extractor.py
│   │   │   ├── pdf_validator.py
│   │   │   └── pipeline.py
│   │   │
│   │   ├── knowledge_graph/
│   │   │   ├── __init__.py
│   │   │   ├── document_graph.py
│   │   │   ├── entity_extractor.py
│   │   │   ├── entity_graph.py
│   │   │   └── neo4j_connection.py
│   │   │
│   │   ├── rag/
│   │   │   ├── answer_generator.py
│   │   │   ├── context_builder.py
│   │   │   ├── guardrails.py
│   │   │   └── rag_pipeline.py
│   │   │
│   │   ├── retrieval/
│   │   │   ├── __init__.py
│   │   │   ├── graph_retriever.py
│   │   │   ├── hybrid_retriever.py
│   │   │   └── vector_retriever.py
│   │   │
│   │   ├── vector_store/
│   │   │   ├── __init__.py
│   │   │   └── faiss_store.py
│   │   │
│   │   └── main.py
│   │
│   ├── build_vector_index.py
│   │
│   ├── test_answer_generator.py
│   ├── test_context_builder.py
│   ├── test_document_graph.py
│   ├── test_entity_extraction.py
│   ├── test_entity_graph.py
│   ├── test_full_entity_pipeline.py
│   ├── test_graph_context_integration.py
│   ├── test_graph_retrieval.py
│   ├── test_hybrid_retrieval.py
│   ├── test_neo4j_connection.py
│   ├── test_neo4j_graph.py
│   ├── test_phase11_pipeline.py
│   ├── test_phase12_evaluation.py
│   ├── test_rag_pipeline.py
│   ├── test_vector_result_structure.py
│   ├── test_vector_retrieval.py
│   │
│   └── tests/
│       ├── test_document_registry.py
│       └── test_duplicate_detection.py
│
├── frontend/
│   └── streamlit_app.py
│
├── data/
│   ├── uploads/
│   ├── processed/
│   ├── registry/
│   └── vector_store/
│
├── sample.pdf
├── test_upload.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .dockerignore
├── .env.example
├── .gitignore
└── README.md
```

---

## 6. File-by-File Guide

### `backend/app/main.py`
FastAPI application entry point. Registers application routes and is the target of the Uvicorn process.

### `backend/app/api/upload.py`
Defines `POST /api/upload/pdf`. It first invokes the PDF upload/extraction pipeline and then invokes `DocumentProcessingService` for downstream Hybrid RAG processing.

### `backend/app/api/query.py`
Defines the query API used by the Streamlit frontend.

### `backend/app/document_registry.py`
Handles deterministic file hashing, lookup of existing documents, and document registration.

### `backend/app/ingestion/pipeline.py`
Initial ingestion lifecycle:

```text
validate
→ read bytes
→ hash
→ duplicate check
→ save PDF
→ extract text
→ save processed JSON
→ register document
```

### `backend/app/ingestion/document_processing_service.py`
Main downstream orchestrator:

```text
processed JSON
→ chunks
→ embeddings
→ FAISS
→ Neo4j document/chunks
→ entity extraction
→ Neo4j entity graph
```

### `backend/app/ingestion/chunker.py`
Creates deterministic document chunks and their metadata.

### `backend/app/ingestion/pdf_extractor.py`
Extracts page-wise PDF text.

### `backend/app/ingestion/pdf_validator.py`
Validates uploaded PDFs.

### `backend/app/ingestion/metadata.py`
Defines document metadata structures.

### `backend/app/embeddings/local_embeddings.py`
Loads the Sentence Transformers model and provides document/query embedding functions.

### `backend/app/vector_store/faiss_store.py`
Persistent FAISS wrapper responsible for create, save, load, metadata handling, and similarity search.

### `backend/build_vector_index.py`
Standalone FAISS rebuild/validation utility.

### `backend/app/knowledge_graph/neo4j_connection.py`
Creates the Neo4j driver from environment variables and provides query execution.

### `backend/app/knowledge_graph/document_graph.py`
Creates graph constraints and stores `HybridDocument` and `HybridChunk` data.

### `backend/app/knowledge_graph/entity_extractor.py`
Performs LLM-based entity/relationship extraction.

### `backend/app/knowledge_graph/entity_graph.py`
Stores extracted entities and relationships in Neo4j.

### `backend/app/retrieval/vector_retriever.py`
Converts the user query to an embedding and performs FAISS similarity search.

### `backend/app/retrieval/graph_retriever.py`
Performs keyword/entity graph retrieval and, in the final production version, scopes results to the active FAISS document corpus.

### `backend/app/retrieval/hybrid_retriever.py`
Combines vector and graph results, normalizes metadata, tracks retrieval sources, and deduplicates using `chunk_id`.

### `backend/app/rag/guardrails.py`
Validates/rejects problematic queries.

### `backend/app/rag/context_builder.py`
Transforms retrieval results into grounded LLM context and computes context/retrieval statistics.

### `backend/app/rag/answer_generator.py`
Generates grounded natural-language answers using retrieved evidence.

### `backend/app/rag/rag_pipeline.py`
Top-level query orchestration.

### `frontend/streamlit_app.py`
User interface for PDF upload, processing, question entry, answer display, and source inspection.

### Test files
The repository includes targeted tests for retrieval, graph storage, entity extraction, context generation, RAG orchestration, duplicate detection, and phase/evaluation validation.

---

## 7. Ingestion Flow in Detail

```text
User uploads PDF
      ↓
Streamlit
      ↓
POST /api/upload/pdf
      ↓
PDF validation
      ↓
SHA-256 hash
      ↓
Duplicate lookup
      ↓
Save original PDF
      ↓
Extract page text
      ↓
Persist processed JSON
      ↓
Create chunks
      ↓
Generate local embeddings
      ↓
Create/update FAISS index
      ↓
Store HybridDocument
      ↓
Store HybridChunk
      ↓
Extract entities/relationships
      ↓
Store graph entities
      ↓
Document ready for retrieval
```

A production Docker test PDF produced:

```text
2 pages
3 chunks
3 embeddings
384 embedding dimension
3 stored Neo4j chunks
18 entity-storage operations
9 distinct graph entities
```

---

## 8. Query Flow in Detail

```text
User question
      ↓
Streamlit
      ↓
POST /api/query
      ↓
Guardrails
      ↓
Create vector query embedding
      ↓
FAISS semantic retrieval
      ↓
Read active document IDs
      ↓
Neo4j graph retrieval scoped to those IDs
      ↓
Merge vector + graph results
      ↓
Deduplicate by chunk_id
      ↓
Build grounded context
      ↓
LLM answer generation
      ↓
Answer + source metadata
      ↓
Streamlit
```

---

## 9. Data Model

### Document

Conceptually:

```text
HybridDocument
 ├── document_id
 ├── filename
 ├── file_size_bytes
 ├── page_count
 ├── total_characters
 ├── source_type
 ├── extraction_method
 └── processed_file
```

### Chunk

```text
HybridChunk
 ├── chunk_id
 ├── document_id
 ├── filename
 ├── page_number
 ├── page_chunk_index
 ├── text
 ├── character_count
 └── chunk_type
```

### Entity

```text
HybridEntity
 ├── entity_id
 ├── name
 └── entity_type
```

### Key relationships

```text
HybridDocument
      │
      └── HAS_HYBRID_CHUNK
                  │
                  ▼
             HybridChunk
                  │
                  └── MENTIONS_HYBRID_ENTITY
                              │
                              ▼
                         HybridEntity
```

---

## 10. Hybrid Retrieval Design

The important idea is that vector and graph retrieval have different jobs.

### FAISS

Useful for questions expressed in natural language or using semantic paraphrases.

```text
Question
  ↓
Embedding
  ↓
Similarity search
  ↓
Relevant text chunks
```

### Neo4j

Useful for entity-oriented and structurally connected context.

```text
Question terms
  ↓
Entity matching
  ↓
Graph traversal
  ↓
Related entities/chunks
```

### Hybrid

```text
             Query
               │
       ┌───────┴────────┐
       ▼                ▼
     FAISS            Neo4j
       │                │
       └───────┬────────┘
               ▼
       Merge + Deduplicate
               ▼
          Grounded Context
```

---

## 11. Document Scoping

One of the most important production fixes was aligning the graph corpus with the active vector corpus.

Before the fix:

```text
FAISS → current uploaded document
Neo4j → all historical documents
```

After the fix:

```text
FAISS metadata
     ↓
active document_id(s)
     ↓
Neo4j filter
     ↓
same document corpus
```

This prevented unrelated graph sources from appearing in source citations and answer context.

---

## 12. Deployment

### Local

Backend:

```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001
```

Frontend:

```bash
streamlit run frontend/streamlit_app.py
```

### Docker

```bash
docker compose up -d --build
```

### VPS production

Project:

```text
/opt/app/hybrid-rag-project
```

Backend:

```text
host 8002 → container 8001
```

Frontend:

```text
host 8502 → container 8501
```

The alternate host ports avoid conflict with the existing `rag-streamlit` service on port 8501.

---

## 13. Production Reverse Proxy

An existing Traefik installation is reused.

Traefik:

```text
/docker/traefik/docker-compose.yml
```

Runs in host mode and listens on:

```text
80
443
```

It provides:

- Docker service discovery,
- HTTP routing,
- HTTPS routing,
- HTTP → HTTPS redirect,
- Let's Encrypt HTTP challenge,
- persistent ACME certificate storage.

The Hybrid RAG frontend is routed through Traefik using Docker labels.

---

## 14. Production Domain and SSL

Production URL:

```text
https://hybridragbot.sbs
```

DNS:

```text
A  hybridragbot.sbs  →  89.116.20.234
```

The domain has no AAAA record.

The production SSL certificate was verified using OpenSSL:

```text
Subject: CN = hybridragbot.sbs
Issuer: Let's Encrypt, CN = YR2
```

The HTTPS endpoint returned:

```text
HTTP/2 200
```

---

## 15. Environment Configuration

Use `.env.example` as the template for local configuration.

Typical runtime settings include:

```text
OPENAI_API_KEY
NEO4J_URI
NEO4J_USERNAME
NEO4J_PASSWORD
NEO4J_DATABASE
```

Never commit secrets.

The real:

```text
.env
```

belongs only in the runtime environment and is excluded from source control.

---

## 16. Runtime Data

```text
data/
├── uploads/
├── processed/
├── registry/
└── vector_store/
```

### `uploads/`
Original user-uploaded PDFs.

### `processed/`
Page-wise extracted JSON documents.

### `registry/`
Document identity and duplicate-detection records.

### `vector_store/`
Persisted FAISS index and metadata.

Runtime data is intentionally not part of the Git source tree.

---

## 17. API Endpoints

### Health check

```text
GET /
```

Example:

```json
{
  "message": "Hybrid RAG API is running"
}
```

### PDF upload

```text
POST /api/upload/pdf
```

Multipart form field:

```text
file
```

### Query

```text
POST /api/query
```

Used by the Streamlit frontend to submit user questions to the RAG pipeline.

---

## 18. Evaluation and Validation

### Functional evaluation

```text
9 total cases
9 passed
7 normal answer cases
2 blocked cases
Average keyword match: 1.0
```

### Production smoke questions

```text
Which technology is used as the vector store?
→ FAISS

Which technology is used for the Knowledge Graph?
→ Neo4j Aura

Which framework provides the backend API?
→ FastAPI
```

The final source view showed current document pages for both vector and graph retrieval.

---

## 19. Operational Commands

### Start

```bash
docker compose up -d
```

### Stop

```bash
docker compose down
```

### Rebuild

```bash
docker compose up -d --build --force-recreate
```

### Status

```bash
docker compose ps
```

### Backend logs

```bash
docker compose logs backend --tail=50
```

### Frontend logs

```bash
docker compose logs frontend --tail=50
```

### Backend health

```bash
curl http://127.0.0.1:8002/
```

### Check FAISS

```bash
find data/vector_store -maxdepth 2 -type f -printf '%p
' | sort
```

### Rebuild FAISS manually

```bash
docker exec -it hybrid-rag-backend   sh -c 'cd /app/backend && python build_vector_index.py'
```

---

## 20. Troubleshooting

### `documents.faiss` not found

Check:

```bash
find data/vector_store -maxdepth 2 -type f
```

Rebuild:

```bash
docker exec -it hybrid-rag-backend   sh -c 'cd /app/backend && python build_vector_index.py'
```

### Neo4j authentication error

Verify:

```text
NEO4J_URI
NEO4J_USERNAME
NEO4J_PASSWORD
NEO4J_DATABASE
```

Then test from inside the backend container:

```bash
docker exec hybrid-rag-backend python -c "import os; from neo4j import GraphDatabase; driver=GraphDatabase.driver(os.environ['NEO4J_URI'], auth=(os.environ['NEO4J_USERNAME'],os.environ['NEO4J_PASSWORD'])); driver.verify_connectivity(); print('NEO4J CONNECTION SUCCESSFUL'); driver.close()"
```

### Port conflict

Current VPS mapping intentionally uses:

```text
8002 → backend
8502 → frontend
```

because another application already uses port 8501.

### Traefik / domain issue

Verify DNS:

```bash
dig +short A hybridragbot.sbs
```

Verify listeners:

```bash
ss -lntp | grep -E ':(80|443)\s'
```

Verify HTTPS:

```bash
curl -I https://hybridragbot.sbs
```

---

## 21. Security Considerations

Current deployment includes:

- secrets stored outside Git,
- Docker isolation,
- HTTPS,
- Let's Encrypt certificates,
- persistent certificate storage,
- graph document scoping,
- input guardrails.

Recommended future hardening:

- restrict direct public access to ports 8002/8502,
- expose the application publicly through Traefik only,
- add authentication for multi-user access,
- add rate limiting,
- add monitoring/logging,
- add backups,
- pin dependencies,
- add automated deployment/CI checks.

---

## 22. Learning Value

This project demonstrates the progression from a basic RAG pipeline to a broader Hybrid RAG architecture.

It covers:

```text
RAG fundamentals
     ↓
PDF ingestion
     ↓
Chunking
     ↓
Embeddings
     ↓
Vector search
     ↓
Knowledge Graph
     ↓
Entity extraction
     ↓
Graph retrieval
     ↓
Hybrid retrieval
     ↓
Grounding
     ↓
Guardrails
     ↓
Evaluation
     ↓
Docker
     ↓
VPS deployment
     ↓
Reverse proxy
     ↓
HTTPS
```

It therefore provides both an educational implementation and a deployed portfolio project.

---

## 23. Project Completion Snapshot

```text
Application                ✅
FastAPI backend             ✅
Streamlit frontend          ✅
PDF ingestion               ✅
Duplicate detection         ✅
Local embeddings            ✅
FAISS                       ✅
Neo4j Aura                  ✅
Entity extraction           ✅
Graph retrieval             ✅
Hybrid retrieval            ✅
Document scoping            ✅
Grounded generation         ✅
Source display              ✅
Guardrails                  ✅
Evaluation                  ✅
Docker                      ✅
Hostinger VPS               ✅
Domain                      ✅
Let's Encrypt SSL           ✅
```

Production:

```text
https://hybridragbot.sbs
```

---

## 24. Future Enhancements

The current system is complete for its intended Hybrid RAG learning/portfolio scope. Possible next iterations include:

- multi-user authentication,
- independent document/workspace isolation,
- more robust multi-document FAISS management,
- background ingestion jobs,
- ingestion progress reporting,
- richer graph relationships,
- reranking,
- caching,
- observability,
- CI/CD,
- automated backups,
- API authentication and rate limiting.

---

## 25. Author

**Mohan Arajan**

Repository:

https://github.com/mohanaraajanai/hybrid-rag-project

Production:

https://hybridragbot.sbs
