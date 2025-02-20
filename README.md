# Guide (Azure and HuggingFace quick setup stack)

## Overview

This guide replace local Hugging Face embeddings with the Hugging Face Inference API. This eliminates the need to run the embedding model locally (e.g., with `transformers` and `torch`), reducing resource demands and simplifying deployment. Below, I’ll update the guide to use the Hugging Face Inference API for embeddings while keeping Azure OpenAI for the language model. I’ll adjust the installation, environment variables, and code accordingly.

---

### Updated Tech Stack Overview

- **Backend Framework**: FastAPI (unchanged).
- **RAG Framework**: LangChain (unchanged).
- **HITL Framework**: LangGraph (unchanged).
- **Language Model**: Azure OpenAI API (unchanged).
- **Embeddings**: Hugging Face Inference API (replaces local `transformers`).
- **Vector Database**: Pinecone (unchanged).
- **User/Feedback Storage**: SQLite (unchanged).
- **Web Features**: `requests` and `BeautifulSoup` (unchanged), SerpAPI (unchanged).
- **Frontend**: Streamlit (unchanged).

The key change is swapping local Hugging Face embeddings for the Inference API, which requires an API key and an HTTP client (`requests`).

---

### Step-by-Step Guide (Updated)

#### Step 1: Prerequisites

- **Python**: 3.10+ installed.
- **API Keys**:
  - Azure OpenAI: Endpoint, API key, and deployment name from [Azure Portal](https://portal.azure.com/).
  - Hugging Face Inference API: API token from [Hugging Face](https://huggingface.co/settings/tokens).
  - Pinecone: [Pinecone Console](https://www.pinecone.io/).
  - SerpAPI: [SerpAPI](https://serpapi.com/).
- **Environment**: Set up a virtual environment:

  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```

#### Step 2: Install Dependencies

Remove `transformers` and `torch`, as we no longer need them for local embeddings:

```bash
pip install fastapi uvicorn langchain langgraph azure-ai-openai pinecone-client sqlite3 requests beautifulsoup4 serpapi streamlit python-jwt
```

- **Notes**:
  - `requests` is already included for web features and will be reused for the Inference API.
  - No need for `torch` since embeddings are offloaded to the API.

#### Step 3: Project Structure (Unchanged)

```
chatbot_project/
├── backend/
│   ├── main.py
│   ├── database.py
│   ├── rag_agent.py
├── frontend/
│   ├── app.py
├── .env
└── requirements.txt  # Optional
```

#### Step 4: Update Environment Variables (`.env`)

Add the Hugging Face API key:

```bash
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint  # e.g., https://your-resource.azure.com/
AZURE_OPENAI_DEPLOYMENT=your_deployment_name  # e.g., gpt-35-turbo
HF_API_KEY=your_huggingface_api_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_pinecone_env  # e.g., "us-west1-gcp"
SERPAPI_API_KEY=your_serpapi_key
SECRET_KEY=your_secret_key_for_jwt
```

Ensure `python-dotenv` is installed (`pip install python-dotenv`).

#### Step 5: Backend Setup (Updated)

##### 5.1: SQLite Database (`backend/database.py`)

- **Unchanged**: No changes needed.

##### 5.2: RAG + HITL Agent (`backend/rag_agent.py`)

```python
from langchain_openai import AzureChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from pinecone import Pinecone
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Azure OpenAI
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version="2023-05-15"  # Check Azure docs for latest version
)

# Hugging Face Inference API Embeddings
HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
HF_API_KEY = os.getenv("HF_API_KEY")

def huggingface_inference_embeddings(texts):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": texts}
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"HF API error: {response.text}")
    embeddings = response.json()
    if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], list):
        return embeddings
    raise Exception("Unexpected HF API response format")

class HuggingFaceInferenceEmbeddings:
    def embed_documents(self, texts):
        return huggingface_inference_embeddings(texts)
    def embed_query(self, text):
        return huggingface_inference_embeddings([text])[0]

embeddings = HuggingFaceInferenceEmbeddings()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "chatbot-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=384, metric="cosine")  # 384 for MiniLM
vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# Sample data for RAG
documents = ["AI is advancing rapidly in 2025.", "Web search APIs are useful."]
vector_store.add_texts(documents)

# Define State for LangGraph
class AgentState(TypedDict):
    query: str
    retrieved_docs: list
    user_input: Optional[str]
    response: Optional[str]

# Nodes for LangGraph
def retrieve(state: AgentState) -> AgentState:
    docs = vector_store.similarity_search(state["query"], k=2)
    state["retrieved_docs"] = [doc.page_content for doc in docs]
    return state

def human_input(state: AgentState) -> AgentState:
    return state

def generate(state: AgentState) -> AgentState:
    prompt = PromptTemplate(
        input_variables=["query", "docs", "user_input"],
        template="Query: {query}\nDocs: {docs}\nHuman Input: {user_input}\nAnswer:"
    )
    response = llm.invoke(prompt.format(query=state["query"], 
                                        docs="\n".join(state["retrieved_docs"]), 
                                        user_input=state["user_input"] or "None"))
    state["response"] = response.content
    return state

# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("human_input", human_input)
workflow.add_node("generate", generate)
workflow.add_edge("retrieve", "human_input")
workflow.add_edge("human_input", "generate")
workflow.add_edge("generate", END)
workflow.set_entry_point("retrieve")
app = workflow.compile(interrupt_before=["human_input"])

def run_agent(query: str, user_input: str = None):
    state = {"query": query, "retrieved_docs": [], "user_input": user_input}
    result = app.invoke(state)
    return result
```

**Key Changes**:

- Replaced local `transformers` with a `HuggingFaceInferenceEmbeddings` class that calls the Hugging Face Inference API (`sentence-transformers/all-MiniLM-L6-v2`).
- Used `requests` to send text to the API and retrieve 384-dimensional embeddings.
- Removed `torch` and `AutoTokenizer`/`AutoModel` dependencies.
- Error handling added for API failures or unexpected responses.

##### 5.3: FastAPI Backend (`backend/main.py`)

- **Unchanged**: The endpoints remain the same, as they don’t directly interact with the embedding logic.

#### Step 6: Frontend Setup (`frontend/app.py`)

- **Unchanged**: The frontend interacts via APIs, so no modifications are needed.

#### Step 7: Run the Application

1. **Start Backend**:

   ```bash
   cd backend
   uvicorn main:app --reload
   ```

2. **Start Frontend**:

   ```bash
   cd frontend
   streamlit run app.py
   ```

3. Access at `http://localhost:8501`.

#### Step 8: Test the Features

- **Login**: `testuser`/`testpass`.
- **Query**: “What’s new in AI?”—uses HF Inference API for embeddings and Azure OpenAI for generation.
- **Feedback**: Rate responses as before.
- **Web Search/Get**: Unchanged, uses SerpAPI and `requests`.

#### Step 9: Deployment Notes

- **Local**: Works as-is.
- **Cloud**: Secure all API keys (Azure, HF, Pinecone, SerpAPI) in environment variables. No local model hosting means lower memory/CPU requirements.

---

### Explanation of Changes

- **Hugging Face Inference API**:
  - Uses `sentence-transformers/all-MiniLM-L6-v2` via the API endpoint (`https://api-inference.huggingface.co/models/...`).
  - `huggingface_inference_embeddings` sends text to the API and returns embeddings (384 dimensions).
  - Requires an API key from Hugging Face, passed in the `Authorization` header.
  - The custom `HuggingFaceInferenceEmbeddings` class adapts the API output to LangChain’s embedding interface.
- **Dependencies**: Removed `transformers` and `torch`, relying on `requests` (already present) for API calls.
- **Pinecone**: Dimension remains 384, consistent with MiniLM embeddings.

### Notes

- **Performance**: The Inference API offloads computation to Hugging Face’s servers, reducing local resource use but introducing latency from HTTP requests. Free tier has rate limits; consider a paid plan for production.
- **Error Handling**: The code checks for API errors and response format. If the API returns unexpected data, it raises an exception—log these in production.
- **Cost**: Azure OpenAI and HF Inference API incur usage costs. Monitor quotas via Azure and Hugging Face dashboards.

This setup maintains all features—authentication, RAG, HITL, feedback, and web capabilities—using Azure OpenAI for generation and Hugging Face Inference API for embeddings, fully functional with the updated code!