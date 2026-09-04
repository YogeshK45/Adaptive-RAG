# Adaptive RAG - Agentic AI Chatbot

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.5.4-orange.svg)](https://python.langchain.com/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-gpt--oss--120b-red.svg)](https://groq.com/)
[![Embeddings](https://img.shields.io/badge/Embeddings-MiniLM--L6--v2%20(384d)-yellow.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Qdrant](https://img.shields.io/badge/Qdrant-VectorDB-purple.svg)](https://qdrant.tech/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Chat%20History-brightgreen.svg)](https://www.mongodb.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF.svg?logo=github-actions&logoColor=white)](https://github.com/features/actions)

## 📋 Overview

**Adaptive RAG** is an intelligent, production-ready Retrieval-Augmented Generation (RAG) system powered by an agentic AI architecture. It combines dynamic query routing, high-performance local vector embeddings, Qdrant vector retrieval, live web search, and ultra-fast LLM inference via Groq to provide accurate, context-aware answers to user queries.

The system adapts its retrieval strategy based on the query classification:
- **Index Route**: Retrieves and synthesizes answers from user-uploaded documents (PDF / TXT).
- **General Route**: Handles common knowledge, reasoning, and conversational queries directly via Groq LLM.
- **Search Route**: Performs live web searches using Tavily when real-time or external data is required.

The multi-turn conversation context is persisted in **MongoDB**, and workflows are orchestrated using **LangGraph**.

---

## 🎯 Key Features

### 🧠 Intelligent Query Routing
- **Adaptive Classification**: Automatically categorizes incoming questions to determine the optimal answering path.
- **Three Core Pipelines**:
  - **`index`**: Deep retrieval over uploaded documents stored in Qdrant.
  - **`general`**: Direct generation with Groq (`openai/gpt-oss-120b`).
  - **`search`**: Real-time web retrieval via Tavily Search API.

### 📚 Advanced RAG Pipeline
- **Document Ingestion**: Upload PDF and TXT files with custom semantic descriptions.
- **Recursive Chunking**: Smart sliding-window chunking (1000 characters, 150 character overlap).
- **Local & Free Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors) executed locally on CPU/GPU with zero API cost.
- **Vector Search**: Qdrant Vector Database with Cosine similarity indexing.
- **Relevance Grading**: Evaluates retrieved document chunks to ensure high context quality.
- **Query Rewriter**: Re-articulates queries when retrieved context is ambiguous or insufficient.

### 🤖 Agentic AI Architecture
- **LangGraph Workflow**: State graph managing nodes for query classification, document retrieval, relevance grading, query rewriting, web search, and response generation.
- **ReAct Tool-Calling Agent**: Reasoning + Acting pattern equipped with specialized tools.
- **Ultra-Fast LLM Inference**: Powered by Groq Cloud (`openai/gpt-oss-120b`, temperature=0).

### 💾 Persistent State & Memory
- **MongoDB Backend**: Stores chat history (HumanMessage, AIMessage, timestamps, session IDs) for multi-turn conversational memory.
- **Session Isolation**: Each user/conversation maintains its own isolated session context.

### 🎨 User Interface (Streamlit)
- **Interactive Multi-Page UI**: Clean login and multi-turn chat screens.
- **Document Sidebar**: In-app PDF/TXT document upload with progress indicators.
- **Seamless Navigation**: Integrated authentication, session token initialization, and chat experience.

### ⚡ API-First Architecture (FastAPI)
- **High-Performance REST Backend**: Fully asynchronous endpoints on `http://localhost:8000`.
- **Interactive Documentation**: Auto-generated Swagger UI at `/docs`.

### 🐳 Containerization & CI/CD
- **Production Docker Image**: Lightweight Debian-based container with pre-cached embedding weights.
- **Automated CI/CD Pipeline**: GitHub Actions workflow automatically builds, tests, and publishes images to Docker Hub on every push.

---

## 🏗️ Architecture & Workflow

```
                             ┌───────────────────────────┐
                             │  Streamlit Frontend (UI)  │
                             │   (http://localhost:8501) │
                             └─────────────┬─────────────┘
                                           │ HTTP Requests
                                           ▼
                             ┌───────────────────────────┐
                             │   FastAPI Backend (API)   │
                             │   (http://localhost:8000) │
                             └─────────────┬─────────────┘
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │        LangGraph State Machine          │
                      │ ┌─────────────────────────────────────┐ │
                      │ │       query_classifier Node         │ │
                      │ └──────────────────┬──────────────────┘ │
                      └────────────────────┼────────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    │                      │                      │
             [route == 'index']    [route == 'general']   [route == 'search']
                    │                      │                      │
                    ▼                      ▼                      ▼
         ┌─────────────────────┐ ┌───────────────────┐ ┌───────────────────┐
         │     doc_tool /      │ │   general_query   │ │    web_search     │
         │  ReAct Agent Node   │ │       Node        │ │  (Tavily Tool)    │
         └──────────┬──────────┘ └─────────┬─────────┘ └─────────┬─────────┘
                    │                      │                     │
                    ▼                      │                     │
         ┌─────────────────────┐           │                     │
         │     grade Node      │           │                     │
         │  (Relevant Context?)│           │                     │
         └──────────┬──────────┘           │                     │
                    │                      │                     │
         ┌──────────┴──────────┐           │                     │
   [score == 'yes']     [score == 'no']    │                     │
         │                     │           │                     │
         ▼                     ▼           │                     │
   ┌───────────┐         ┌───────────┐     │                     │
   │ generate  │         │  rewrite  │     │                     │
   │   Node    │         │   Node    │     │                     │
   └─────┬─────┘         └─────┬─────┘     │                     │
         │                     │           │                     │
         └─────────────────────┼───────────┴─────────────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │    Save Context to MongoDB    │
               │         (chat_history)        │
               └───────────────┬───────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │     Final Response to User    │
               └───────────────────────────────┘
```

---

## 📦 Project Structure

```
Adaptive-Rag/
├── .github/
│   └── workflows/
│       └── docker.yml                    # GitHub Actions CI/CD workflow
├── src/                                  # Backend application source code
│   ├── main.py                           # FastAPI application entry point & root route
│   ├── api/                              # REST API endpoints
│   │   └── routes.py                     # /rag/query, /rag/documents/upload, /api/init, /api/login
│   ├── config/                           # Application configuration & prompts
│   │   ├── settings.py                   # YAML & environment configuration loader
│   │   └── prompts.yaml                  # System, classification, grading & rewriting prompts
│   ├── core/                             # Core utilities & environment settings
│   │   ├── config.py                     # Pydantic Settings & environment variables
│   │   └── logger.py                     # Structured logging configuration
│   ├── db/                               # Database drivers
│   │   └── mongo_client.py               # Async MongoDB motor client
│   ├── llms/                             # LLM Provider integrations
│   │   ├── groq.py                       # Groq ChatGroq (openai/gpt-oss-120b)
│   │   └── openai.py                     # Compatibility wrapper re-exporting Groq LLM
│   ├── memory/                           # Conversation history & persistence
│   │   ├── chat_history_mongo.py         # MongoDB-backed multi-turn chat history
│   │   └── chathistory_in_memory.py      # In-memory fallback history
│   ├── models/                           # Data models & Pydantic schemas
│   │   ├── state.py                      # LangGraph GraphState definition
│   │   ├── query_request.py              # QueryRequest input schema
│   │   ├── grade.py                      # Relevance grade output model
│   │   ├── route_identifier.py           # RouteIdentifier schema (index/general/search)
│   │   └── verification_result.py        # Fact verification model
│   ├── rag/                              # Core RAG implementation
│   │   ├── graph_builder.py              # LangGraph workflow compilation
│   │   ├── retriever_setup.py            # HuggingFace Embeddings & Qdrant VectorStore
│   │   ├── document_upload.py            # PDF/TXT loading, recursive chunking & indexing
│   │   └── reAct_agent.py                # ReAct retriever agent setup
│   └── tools/                            # Tool definitions
│       ├── common_tools.py               # LLM description enhancer
│       └── graph_tools.py                # Query classification & fact-checking tools
│
├── streamlit_app/                        # Streamlit web application
│   ├── home.py                           # Authentication & session initialization page
│   ├── pages/
│   │   └── chat.py                       # Interactive chat interface & document uploader
│   └── utils/
│       └── api_client.py                 # Frontend HTTP client connecting to FastAPI (port 8000)
│
├── Dockerfile                            # Production Docker image specification
├── .dockerignore                         # Files excluded from Docker build context
├── requirements.txt                      # Project dependencies
├── .env                                  # Environment variables configuration
└── README.md                             # Project documentation
```

---

## 🔌 API Reference

**Base URL**: `http://localhost:8000`

### 1. Initialize Session
* **Endpoint**: `POST /api/init`
* **Description**: Initializes an API session and returns a session token for the client.
* **Response**:
  ```json
  {
    "api_token": "50715e65-5c31-4f1e-8a7e-53c4905f358a"
  }
  ```

### 2. User Authentication
* **Endpoint**: `POST /api/create_user` | `POST /api/login`
* **Body**:
  ```json
  {
    "username": "user123",
    "password": "password123"
  }
  ```
* **Response**:
  ```json
  {
    "jwt": "jwt_user123_86efccc5",
    "username": "user123"
  }
  ```

### 3. RAG Query
* **Endpoint**: `POST /rag/query`
* **Description**: Processes a query through the Adaptive RAG LangGraph workflow.
* **Body**:
  ```json
  {
    "query": "What are the remote work guidelines?",
    "session_id": "session_user123"
  }
  ```
* **Response**:
  ```json
  {
    "result": {
      "content": "Based on the uploaded policy, full-time employees are allowed up to 4 remote days per week...",
      "type": "ai"
    }
  }
  ```

### 4. Document Upload
* **Endpoint**: `POST /rag/documents/upload`
* **Description**: Ingests, chunks, embeds (384d), and indexes a document in Qdrant.
* **Headers**: `X-Description: TechCorp remote work and stipend policy`
* **Form-Data**: `file: <policy.pdf / policy.txt>`
* **Response**:
  ```json
  {
    "status": true
  }
  ```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python**: `3.9` or higher (tested with `Python 3.11` / `Python 3.13`)
- **Docker**: Docker Engine or Docker Desktop (optional, for containerized run)
- **MongoDB**: Local MongoDB instance (`mongodb://localhost:27017`) or MongoDB Atlas
- **Qdrant**: Qdrant Cloud cluster or local container (`http://localhost:6333`)
- **Groq API Key**: Free API key from [Groq Console](https://console.groq.com/)
- **Tavily API Key**: Search API key from [Tavily AI](https://tavily.com/)

### 2. Installation (Local Virtual Environment)

```bash
# 1. Clone the repository
git clone https://github.com/YogeshK45/Adaptive-RAG.git
cd Adaptive-RAG

# 2. Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the root directory:

```env
# Groq LLM Configuration
GROQ_API_KEY=gsk_your_groq_api_key_here

# Tavily Web Search Configuration
TAVILY_API_KEY=tvly-your_tavily_api_key_here

# Qdrant Vector Database Configuration
QDRANT_URL=https://your-cluster-id.us-west-2-0.aws.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_CODE_COLLECTION=codebase
QDRANT_DOCS_COLLECTION=guidelines

# MongoDB Configuration
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB_NAME=adaptive_rag

# Frontend-Backend Service URLs
RUST_BASE_URL=http://localhost:8000/api
PYTHON_BASE_URL=http://localhost:8000
```

### 4. Running Locally

#### Terminal 1 — Start the FastAPI Backend:
```bash
source venv/bin/activate
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```
*API interactive documentation will be available at: [http://localhost:8000/docs](http://localhost:8000/docs)*

#### Terminal 2 — Start the Streamlit Frontend:
```bash
source venv/bin/activate
streamlit run streamlit_app/home.py
```
*Web application will open at: [http://localhost:8501](http://localhost:8501)*

---

## 🐳 Docker Setup

The project is fully containerized with Docker for consistent, reproducible execution across local environments and cloud runners.

### Key Components:
- **`Dockerfile`**: Builds a lightweight container on top of `python:3.11-slim`, installs system build tools and SSL certificates, pre-caches the Hugging Face `sentence-transformers/all-MiniLM-L6-v2` model weights during build to eliminate startup latency, and exposes port `8000`.
- **`.dockerignore`**: Excludes local virtual environments (`venv/`), secrets (`.env`), cache files (`__pycache__/`), logs, and Git metadata to keep images secure and lightweight.

### 1. Build the Docker Image
```bash
docker build -t adaptive-rag:latest .
```

### 2. Run the Container Locally

#### Run FastAPI Backend (Port 8000):
```bash
docker run -d \
  --name adaptive-rag-app \
  -p 8000:8000 \
  --env-file .env \
  adaptive-rag:latest
```

*Access the running backend:*
- **Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **API Root**: [http://localhost:8000/](http://localhost:8000/)

#### Run Streamlit Frontend (Port 8501):
```bash
docker run -d \
  --name adaptive-rag-ui \
  -p 8501:8501 \
  --env-file .env \
  adaptive-rag:latest \
  streamlit run streamlit_app/home.py --server.port 8501 --server.address 0.0.0.0
```

*Access the web UI:*
- **Streamlit Interface**: [http://localhost:8501](http://localhost:8501)

#### Manage Containers:
```bash
# View running containers
docker ps

# Check container logs
docker logs -f adaptive-rag-app

# Stop and remove containers
docker stop adaptive-rag-app && docker rm adaptive-rag-app
```

---

## 🔄 CI/CD Pipeline

The repository includes an automated Continuous Integration and Continuous Delivery (CI/CD) workflow implemented using **GitHub Actions**.

### Pipeline Flow:
```
┌─────────────────┐       ┌──────────────────────┐       ┌──────────────────────┐       ┌──────────────────────┐
│   Git Push to   │ ───►  │    GitHub Actions    │ ───►  │     Docker Build     │ ───►  │      Docker Hub      │
│   main Branch   │       │   Runner (Ubuntu)    │       │ (Pre-cached Weights) │       │   Container Registry │
└─────────────────┘       └──────────────────────┘       └──────────────────────┘       └──────────────────────┘
```

### Workflow Features:
1. **Automated Triggers**: The pipeline automatically triggers on every `push` to the `main` branch and supports manual execution via `workflow_dispatch`.
2. **Build Isolation**: Uses `docker/setup-buildx-action` to build the Docker image in an isolated environment.
3. **Layer Caching**: Leverages GitHub Actions cache (`type=gha`) to cache dependencies and model layers, accelerating subsequent build runs.
4. **Automated Registry Push**: Securely logs into Docker Hub using encrypted repository secrets and publishes the updated image.

### Required GitHub Secrets:
To enable automated deployment to Docker Hub, configure the following secrets in your repository settings (**Settings** → **Secrets and variables** → **Actions**):

| Secret Name | Purpose |
|---|---|
| `DOCKERHUB_USERNAME` | Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub Personal Access Token (Read & Write permissions) |

The workflow definition is located at [`.github/workflows/docker.yml`](file:///Users/yogeshrajput/Desktop/Adaptive-Rag/.github/workflows/docker.yml).

---

## 🛠️ Technology Stack

| Layer | Technology | Details |
|---|---|---|
| **LLM Provider** | **Groq Cloud** | `openai/gpt-oss-120b` (deterministic temperature=0) |
| **Embeddings** | **Hugging Face** | `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, local & free) |
| **Workflow Graph** | **LangGraph** | Multi-node state graph with conditional branching |
| **Agent Framework** | **LangChain** | Tool-calling ReAct agent and prompt templates |
| **Vector Database** | **Qdrant** | Dense vector storage with Cosine similarity |
| **Chat Memory DB** | **MongoDB** | Persistent multi-turn chat history via `motor` |
| **Web Search** | **Tavily AI** | Real-time search API integration |
| **Backend API** | **FastAPI + Uvicorn** | Asynchronous REST endpoints |
| **Frontend UI** | **Streamlit** | Multi-page web chat application |
| **Containerization** | **Docker** | Multi-stage image with pre-cached model weights |
| **CI/CD Automation** | **GitHub Actions** | Automated build and push to Docker Hub registry |

---

## ❓ Frequently Asked Questions (FAQ)

**Q: Which embedding model is used and why?**  
A: The project uses `sentence-transformers/all-MiniLM-L6-v2`. It generates 384-dimensional dense vectors locally without any API cost. It requires 4x less memory in Qdrant compared to 1536-dimensional models and runs with ~5-10ms inference latency on standard CPUs.

**Q: Are there any OpenAI runtime dependencies left?**  
A: No. All LLM calls run through Groq (`ChatGroq`) and all embeddings run through local Hugging Face Sentence Transformers.

**Q: How does query routing work?**  
A: When a question arrives, the `query_classifier` node uses structured output to classify the intent into `index` (document RAG), `general` (direct LLM knowledge), or `search` (real-time web search with Tavily).

**Q: Where is conversation history saved?**  
A: In MongoDB inside the `chat_history` collection of the `adaptive_rag` database, keyed by `session_id`.

**Q: How are embedding model weights handled in Docker?**  
A: The weights are downloaded and cached during the Docker image build process (`RUN python -c ...`), ensuring the container starts up immediately without downloading weights on every boot.

---

## 📄 License

This project is licensed under the MIT License.
