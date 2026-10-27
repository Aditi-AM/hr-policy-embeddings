# HR Bot — Document Ingestion Pipeline

An AWS Lambda-based pipeline that automatically ingests HR policy documents (PDF, DOCX) from S3, generates vector embeddings via an AI Gateway, and indexes them into OpenSearch for semantic search.

## Architecture

```
S3 Bucket (ObjectCreated / ObjectRemoved)
        │
        ▼
  Lambda Function
        │
        ├─ 1. Download document from S3
        ├─ 2. Extract metadata from S3 object tags
        ├─ 3. Parse document into text chunks (PDF pages / DOCX semantic chunks)
        ├─ 4. Generate embeddings via AI Gateway
        ├─ 5. Index chunks + embeddings into OpenSearch (k-NN)
        └─ 6. Fallback: save embeddings to pkl → S3 if OpenSearch is unavailable
```

On `ObjectRemoved` events the pipeline deletes the corresponding chunks from both OpenSearch and the pkl store.

## Project Structure

```
hr_bot/
├── lambda_function.py        # Lambda entry point — routes S3 events to ingest/delete
├── config_utils.py           # Loads all config from AWS Secrets Manager
├── s3_utils.py               # S3 download, tag extraction, /tmp cleanup
├── file_handler.py           # PDF & DOCX parsing and chunking
├── ai_gateway_utils.py       # Embedding generation via AI Gateway REST API
├── opensearch_utils.py       # OpenSearch index management and bulk indexing
├── embeddings_log_utils.py   # Persistent pkl store (fallback when OpenSearch is down)
├── architecture.md           # Detailed architecture notes
├── requirements.txt          # Python dependencies
└── .gitignore
```

## Prerequisites

- **Python 3.11+**
- **AWS CLI** configured with credentials that have access to:
  - S3 (read/write on the target bucket)
  - Secrets Manager (read on config and token secrets)
  - OpenSearch (index/delete on the target domain)
- **AWS Secrets Manager** secrets created (see Configuration below)

## Configuration

All runtime configuration is stored in a single JSON secret in AWS Secrets Manager. No secrets are hardcoded in the code.

### Lambda Environment Variables

Set these in the Lambda console under **Configuration → Environment variables**:

| Key | Description | Default |
|---|---|---|
| `HR_BOT_CONFIG_SECRET_NAME` | Name of the config secret in Secrets Manager | `hr-bot-config-dev` |
| `ASM_REGION` | AWS region for Secrets Manager | `ap-south-1` |
| `AI_GATEWAY_TIMEOUT` | Embedding request timeout in seconds (optional) | `30` |

### Secrets Manager — Config Secret

Create a secret (e.g. `hr-bot-config-dev`) with this JSON structure:

```json
{
  "opensearch_endpoint": "https://search-xxx.ap-south-1.es.amazonaws.com",
  "opensearch_index": "hr-docs-embeddings",
  "embedding_dimension": 1536,
  "ai_gateway_url": "https://ai.titan.in/gateway",
  "embed_model": "gemini-multimodalembedding",
  "ai_gateway_token_secret_name": "titan-ess-entrapp-token-dev",
  "asm_region": "ap-south-1"
}
```

### Secrets Manager — AI Gateway Token Secret

Create a second secret (name must match `ai_gateway_token_secret_name` above) with:

```json
{
  "api_key_value": "<your-ai-gateway-api-key>"
}
```

## S3 Object Tags

Tag uploaded documents with these S3 object tags for metadata extraction:

| Tag Key | Example | Description |
|---|---|---|
| `department` | `HR` | Owning department |
| `policy_type` | `leave` | Policy category |
| `effective_date` | `2024-01-01` | Policy effective date (ISO format) |

Missing tags default to `"unknown"` / `null` — ingestion still proceeds.

## Local Development

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd hr_bot
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export HR_BOT_CONFIG_SECRET_NAME="hr-bot-config-dev"
export ASM_REGION="ap-south-1"
```

### 3. Run locally (optional)

Create a test script that invokes `lambda_handler` with a mock S3 event:

```python
from lambda_function import lambda_handler

event = {
    "Records": [{
        "eventName": "ObjectCreated:Put",
        "s3": {
            "bucket": {"name": "your-bucket"},
            "object": {"key": "your-file.pdf"}
        }
    }]
}

response = lambda_handler(event, context=None)
print(response)
```

## Deploying to AWS Lambda

### 1. Install dependencies into the project directory

```bash
pip install -r requirements.txt -t .
```

### 2. Package as a zip

```bash
zip -r lambda_deployment.zip . -x ".git/*" "__pycache__/*" "*.dist-info/*" ".venv/*" ".gitignore" "README.md" "architecture.md" "test_local.py" "*.pkl"
```

### 3. Upload to Lambda

```bash
aws lambda update-function-code \
  --function-name <your-function-name> \
  --zip-file fileb://lambda_deployment.zip
```

Or upload via the AWS Lambda console.

### 4. Configure the S3 trigger

In the Lambda console, add an S3 trigger:
- **Bucket**: your document bucket
- **Event types**: `s3:ObjectCreated:*` and `s3:ObjectRemoved:*`
- **Prefix/Suffix**: filter as needed (e.g. suffix `.pdf`)

### 5. IAM permissions

The Lambda execution role needs:

```json
{
  "Effect": "Allow",
  "Action": [
    "s3:GetObject",
    "s3:HeadObject",
    "s3:GetObjectTagging",
    "s3:PutObject"
  ],
  "Resource": "arn:aws:s3:::your-bucket/*"
},
{
  "Effect": "Allow",
  "Action": "secretsmanager:GetSecretValue",
  "Resource": [
    "arn:aws:secretsmanager:ap-south-1:*:secret:hr-bot-config-dev*",
    "arn:aws:secretsmanager:ap-south-1:*:secret:titan-ess-entrapp-token-dev*"
  ]
}
```

## Supported File Types

| Format | Parser | Chunking Strategy |
|---|---|---|
| PDF | `pypdf` | One chunk per page |
| DOCX / DOC | `python-docx` | Semantic chunks (~500 words, 50-word overlap) |

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Config secret not found in ASM` | Secret name mismatch | Verify `HR_BOT_CONFIG_SECRET_NAME` env var matches the actual secret name |
| `Access denied to secret` | Missing IAM permission | Add `secretsmanager:GetSecretValue` to the Lambda role |
| `AI Gateway returned 401/403` | Bad or expired API key | Update the token secret in Secrets Manager |
| `Cannot reach OpenSearch endpoint` | Network / VPC issue | Ensure Lambda is in the same VPC as OpenSearch, or endpoint is public |
| `PDF is encrypted` | Password-protected PDF | Remove encryption before uploading |
