# HR Bot — Document Ingestion Pipeline Architecture

## Overview

An event-driven pipeline that processes HR documents uploaded to S3, generates multimodal embeddings via Gemini through an AI Gateway, and stores the vectors with metadata in Amazon OpenSearch Service. All AWS infrastructure is provisioned and managed using **boto3 SDK.**

---

## Scope (Phase 1)

> **In scope**: Document ingestion — S3 upload → parse → chunk → embed → store in OpenSearch.
> **Out of scope (Phase 2)**: Query/retrieval, RAG answer generation, chat UI.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                               │
│                                                                         │
│  ┌──────────────┐    S3 PUT/POST     ┌─────────────────────────────┐   │
│  │  S3 Bucket   │ ─────────────────► │     Lambda Function         │   │
│  │  (HR Docs)   │   Event Trigger    │     (Ingestion Handler)     │   │
│  └──────────────┘                    │                             │   │
│                                      │  /tmp workspace             │   │
│                                      │  ┌───────────────────────┐  │   │
│                                      │  │ 1. Download S3 → /tmp │  │   │
│                                      │  │ 2. Detect file type   │  │   │
│                                      │  │                       │  │   │
│                                      │  │  PDF         DOCX     │  │   │
│                                      │  │   │            │      │  │   │
│                                      │  │   ▼            ▼      │  │   │
│                                      │  │ PyMuPDF   python-docx │  │   │
│                                      │  │ page→img  text blocks │  │   │
│                                      │  │   │            │      │  │   │
│                                      │  │   └─────┬──────┘      │  │   │
│                                      │  │         ▼             │  │   │
│                                      │  │   Chunking Layer      │  │   │
│                                      │  │   + Metadata tags     │  │   │
│                                      │  │         │             │  │   │
│                                      │  └─────────┼─────────────┘  │   │
│                                      │            │                 │   │
│                                      │            ▼                 │   │
│                                      │  ┌───────────────────────┐  │   │
│                                      │  │   AI Gateway          │  │   │
│                                      │  │   Gemini Multimodal   │  │   │
│                                      │  │   Embedding Model     │  │   │
│                                      │  └──────────┬────────────┘  │   │
│                                      │             │                │   │
│                                      │             ▼                │   │
│                                      │  ┌───────────────────────┐  │   │
│                                      │  │  OpenSearch Service   │  │   │
│                                      │  │  k-NN Vector Index    │  │   │
│                                      │  └───────────────────────┘  │   │
│                                      │             │                │   │
│                                      │             ▼                │   │
│                                      │       Cleanup /tmp          │   │
│                                      └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Flow (Step by Step)

```
S3 Upload (PDF / DOCX)
        │
        ▼
S3 Event Notification (ObjectCreated)
        │
        ▼
Lambda Triggered
        │
        ├── boto3: download object → /tmp/{filename}
        ├── boto3: fetch S3 object tags (department, policy_type, effective_date)
        │
        ├─── [PDF] ──────────────────────────────────────────┐
        │    PyMuPDF: render each page → PNG image           │
        │    PyMuPDF: extract raw text per page              │
        │    Chunk = one page (image + text)                 │
        │                                                    │
        ├─── [DOCX] ─────────────────────────────────────────┤
        │    python-docx: extract paragraphs                 │
        │    Semantic chunking (~500 tokens, 50-token overlap)│
        │    Chunk = text block                              │
        │                                                    ▼
        │                                    Build payload per chunk
        │                                    { image_b64, text, metadata }
        │                                              │
        │                                              ▼
        │                                    POST → AI Gateway
        │                                    Gemini Multimodal Embedding
        │                                    ← vector[]
        │                                              │
        │                                              ▼
        │                                    opensearch-py: bulk index
        │                                    { vector, text, metadata }
        │
        └── Cleanup /tmp  →  Done
```

---

## Tech Stack

| Layer             | Technology                       | Reason                                              |
| ----------------- | -------------------------------- | --------------------------------------------------- |
| Runtime           | Python 3.12                      | Best library ecosystem for document processing      |
| AWS SDK           | boto3                            | S3 access, Lambda config, OpenSearch domain mgmt    |
| PDF processing    | PyMuPDF (`fitz`)               | Fast page-to-image rendering, handles complex PDFs  |
| DOCX processing   | `python-docx`                  | Standard library for Word document extraction       |
| HTTP client       | `httpx`                        | Async-capable, clean API for AI Gateway calls       |
| Embedding model   | Gemini Multimodal via AI Gateway | Native handling of images, tables, and text         |
| Vector store      | Amazon OpenSearch Service        | Managed k-NN, AWS-native, supports metadata filters |
| OpenSearch client | `opensearch-py`                | Official Python client for OpenSearch               |
| Observability     | CloudWatch Logs + Metrics        | Native Lambda integration                           |
| Error handling    | SQS Dead Letter Queue (DLQ)      | Captures failed Lambda invocations for reprocessing |

---

## OpenSearch Index Schema

### Index Settings

```json
{
  "settings": {
    "index": {
      "knn": true,
      "knn.algo_param.ef_search": 512
    }
  },
  "mappings": {
    "properties": {
      "document_id":    { "type": "keyword" },
      "chunk_id":       { "type": "keyword" },
      "source_s3_key":  { "type": "keyword" },
      "file_name":      { "type": "keyword" },
      "file_type":      { "type": "keyword" },
      "page_number":    { "type": "integer" },
      "chunk_index":    { "type": "integer" },
      "text_content":   { "type": "text" },
      "department":     { "type": "keyword" },
      "policy_type":    { "type": "keyword" },
      "effective_date": { "type": "date" },
      "ingested_at":    { "type": "date" },
      "embedding": {
        "type": "knn_vector",
        "dimension": 3072,
        "method": {
          "name": "hnsw",
          "engine": "nmslib",
          "space_type": "cosinesimil"
        }
      }
    }
  }
}
```

### Sample Document

```json
{
  "document_id":   "d3f1a2b4-...",
  "chunk_id":      "c7e9f0a1-...",
  "source_s3_key": "hr-docs/policies/leave_policy_2024.pdf",
  "file_name":     "leave_policy_2024.pdf",
  "file_type":     "pdf",
  "page_number":   3,
  "chunk_index":   2,
  "text_content":  "Employees are entitled to 20 days of annual leave...",
  "department":    "HR",
  "policy_type":   "leave",
  "effective_date":"2024-01-01",
  "ingested_at":   "2025-01-15T10:00:00Z",
  "embedding":     [0.021, -0.043, 0.017, "...3072 values total"]
}
```

---

## S3 Object Tag Convention

Metadata is sourced from S3 object tags set at upload time. These map directly to OpenSearch metadata fields.

| S3 Tag Key         | Example Value           | Required | OpenSearch Field   |
| ------------------ | ----------------------- | -------- | ------------------ |
| `department`     | `HR`, `Finance`     | No       | `department`     |
| `policy_type`    | `leave`, `handbook` | No       | `policy_type`    |
| `effective_date` | `2024-01-01`          | No       | `effective_date` |

Missing tags default to `"unknown"` (strings) or `null` (dates).

---

## Lambda Configuration (via boto3 SDK)

```
Runtime:         Python 3.12
Memory:          1024 MB
Timeout:         10 minutes (600 seconds)
/tmp storage:    512 MB (sufficient for <10MB source files)
Trigger:         S3 ObjectCreated (PUT, POST, COPY)
DLQ:             SQS queue for failed invocations
Environment vars:
  OPENSEARCH_ENDPOINT   Amazon OpenSearch domain endpoint
  AI_GATEWAY_URL        AI Gateway base URL
  AI_GATEWAY_API_KEY    API key (stored in Secrets Manager)
  OPENSEARCH_INDEX      Target index name
  S3_BUCKET_NAME        Source S3 bucket name
```

---

## Project Structure

```
hr_bot/
│
├── architecture.md                  ← this file
│
├── infra/
│   ├── setup_s3.py                 # Create S3 bucket + event notification (boto3)
│   ├── setup_lambda.py              # Create/update Lambda function (boto3)
│   ├── setup_opensearch.py          # Create OpenSearch domain + index (boto3)
│   ├── setup_sqs_dlq.py             # Create SQS DLQ for Lambda (boto3)
│   └── config.py                    # Shared config (region, names, etc.)
│
├── lambda/
│   ├── handler.py                   # Lambda entry point
│   ├── downloader.py                # S3 download + tag extraction
│   ├── parsers/
│   │   ├── pdf_parser.py            # PyMuPDF: page → image + text
│   │   └── docx_parser.py           # python-docx: text extraction + chunking
│   ├── chunker.py                   # Semantic chunking for text
│   ├── embedder.py                  # AI Gateway → Gemini multimodal embedding
│   ├── indexer.py                   # opensearch-py: bulk index chunks
│   └── cleanup.py                   # /tmp cleanup utility
│
├── requirements.txt                 # Python dependencies
└── tests/
    ├── test_pdf_parser.py
    ├── test_docx_parser.py
    ├── test_chunker.py
    ├── test_embedder.py
    └── test_indexer.py
```

---

## Task Breakdown

### Task 1 — OpenSearch index setup (`infra/setup_opensearch.py`)

**Objective**: Create the OpenSearch domain and k-NN index using boto3 before any data flows in.

**Steps**:

- Use `boto3` `opensearchservice` client to create/verify the OpenSearch domain
- Use `opensearch-py` client to create the index with k-NN mapping defined above
- Set `dimension: 3072` to match Gemini embedding output size
- Add keyword/date fields for all metadata

**Test**: Assert index exists; index a dummy document with a random vector and retrieve it by `chunk_id`.

---

### Task 2 — S3 bucket + Lambda trigger (`infra/setup_s3.py`, `infra/setup_lambda.py`)

**Objective**: Create the S3 bucket and configure it to trigger the Lambda on PUT/POST using boto3.

**Steps**:

- `boto3` S3 client: create bucket, enable versioning
- `boto3` Lambda client: create function (zip the `lambda/` directory)
- `boto3` S3 client: `put_bucket_notification_configuration` pointing to the Lambda ARN
- `boto3` Lambda client: add resource-based policy so S3 can invoke Lambda

**Test**: Upload a test file to S3, verify Lambda invocation via CloudWatch logs.

---

### Task 3 — S3 download + tag extraction (`lambda/downloader.py`)

**Objective**: Lambda downloads the triggered document to `/tmp` and reads its S3 object tags.

**Steps**:

- `boto3` S3 client: `download_file(bucket, key, '/tmp/{filename}')`
- `boto3` S3 client: `get_object_tagging(Bucket, Key)` → parse into metadata dict
- Log file size; raise error if file exceeds `/tmp` limit
- `try/finally` block guarantees cleanup even on crash

**Test**: Unit test with mocked boto3; integration test with a real tagged PDF.

---

### Task 4 — PDF processing (`lambda/parsers/pdf_parser.py`)

**Objective**: Render each PDF page as a PNG image and extract raw text per page.

**Steps**:

- `fitz.open('/tmp/{file}')` to load PDF
- Iterate pages: `page.get_pixmap(dpi=150)` → save as `/tmp/page_{n}.png`
- `page.get_text()` → raw text string per page
- Return list of `{ page_num, image_path, text }`

**Test**: Sample PDF with table and embedded image — assert correct page count, PNG files exist, text is non-empty.

---

### Task 5 — DOCX processing (`lambda/parsers/docx_parser.py`)

**Objective**: Extract text from DOCX files and produce semantic chunks.

**Steps**:

- `Document('/tmp/{file}')` from `python-docx`
- Collect paragraphs, concatenate until ~500 tokens
- Slide window with 50-token overlap for context continuity
- Return list of `{ chunk_index, text }`

**Test**: Sample DOCX — assert chunk sizes within bounds, overlap text matches end of previous chunk, no empty chunks.

---

### Task 6 — AI Gateway integration (`lambda/embedder.py`)

**Objective**: Call AI Gateway to generate Gemini multimodal embeddings per chunk.

**Steps**:

- For PDF chunks: base64-encode the page PNG, send `{ image_b64, text }` to AI Gateway
- For DOCX chunks: send `{ text }` only
- Parse response to extract float vector
- Retry on 429/5xx with exponential backoff (3 attempts, 2s base delay)

**Test**: Unit test with mocked `httpx`; integration test — send one real PDF page, assert non-empty float array with 3072 dimensions returned.

---

### Task 7 — OpenSearch indexing (`lambda/indexer.py`)

**Objective**: Bulk index all chunks (vector + metadata) into the OpenSearch k-NN index.

**Steps**:

- Build bulk payload from all chunks: `chunk_id`, `embedding`, `text_content`, metadata
- Use `opensearch-py` `bulk()` helper
- Handle per-document errors (partial failure must not abort the whole batch)
- Log count of successfully indexed vs failed chunks

**Test**: Index test chunks, query by `document_id`, assert all chunks present with correct metadata and non-null vectors.

---

### Task 8 — Handler wiring + observability (`lambda/handler.py`)

**Objective**: Wire all modules into a single robust Lambda handler with structured logging and CloudWatch metrics.

**Steps**:

- Orchestrate: download → parse → chunk → embed → index → cleanup
- Structured JSON logging: `document_id`, `chunk_count`, `embedding_latency_ms`, `indexing_latency_ms`
- Emit custom CloudWatch metrics: `DocumentsProcessed`, `ChunksIndexed`, `EmbeddingErrors`, `IndexingErrors`
- On unrecoverable error: log S3 key, let Lambda fail → SQS DLQ captures the event for reprocessing

**Test**: End-to-end — upload one PDF and one DOCX, verify both appear in OpenSearch with correct chunk counts and metadata. Simulate AI Gateway failure, verify DLQ receives the event.

---

## Dependencies (`requirements.txt`)

```
boto3==1.34.0
PyMuPDF==1.24.0
python-docx==1.1.0
httpx==0.27.0
opensearch-py==2.6.0
```

---

## CloudWatch Metrics Reference

| Metric Name            | Unit  | Description                                    |
| ---------------------- | ----- | ---------------------------------------------- |
| `DocumentsProcessed` | Count | Successfully processed documents               |
| `ChunksIndexed`      | Count | Total chunks stored in OpenSearch              |
| `EmbeddingErrors`    | Count | Chunks that failed embedding (AI Gateway)      |
| `IndexingErrors`     | Count | Chunks that failed to index in OpenSearch      |
| `EmbeddingLatency`   | Ms    | Time taken per AI Gateway embedding call       |
| `IndexingLatency`    | Ms    | Time taken for OpenSearch bulk index operation |

---

## Error Handling Strategy

| Failure Point        | Strategy                                                |
| -------------------- | ------------------------------------------------------- |
| S3 download fails    | Raise exception → Lambda fails → SQS DLQ              |
| PDF/DOCX parse fails | Log error + S3 key, raise → SQS DLQ                    |
| AI Gateway 429       | Exponential backoff, 3 retries, then fail chunk         |
| AI Gateway 5xx       | Exponential backoff, 3 retries, then skip chunk + log   |
| OpenSearch bulk fail | Log failed chunk IDs, continue with remaining chunks    |
| Lambda timeout       | Partial failure logged; SQS DLQ captures for full retry |

---

## Phase 2 Preview (Out of Scope Now)

When retrieval is built, it will reuse the same OpenSearch index:

```
User Query (text)
      │
      ▼
API Gateway → Lambda (Query Handler)
      │
      ├── AI Gateway: Gemini multimodal embed(query text)
      │   ← query_vector[]
      │
      ├── OpenSearch: k-NN search(query_vector, filters={department, policy_type})
      │   ← top-K chunks
      │
      ├── AI Gateway: Gemini (LLM) generate answer from chunks
      │   ← answer text
      │
      └── Return { answer, source_chunks } to chat UI
```
