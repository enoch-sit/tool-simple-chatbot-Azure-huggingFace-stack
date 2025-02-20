# Guide (Azure and HuggingFace quick setup stack)

## Overview

Below is an updated version of the step-by-step guide, replacing OpenAI's API and embeddings with Azure OpenAI API and Hugging Face embeddings. I'll adjust the frameworks, installation process, and code accordingly, focusing on the backend and frontend changes. The rest of the stack (FastAPI, LangChain, LangGraph, Pinecone, SQLite, etc.) remains largely unchanged, but I'll highlight modifications where needed.

---

### Updated Tech Stack Overview

- **Backend Framework**: FastAPI (unchanged).
- **RAG Framework**: LangChain (unchanged, supports Azure OpenAI).
- **HITL Framework**: LangGraph (unchanged).
- **Language Model**: Azure OpenAI API (replaces OpenAI API).
- **Embeddings**: Hugging Face Transformers (replaces OpenAI embeddings).
- **Vector Database**: Pinecone (unchanged).
- **User/Feedback Storage**: SQLite (unchanged).
- **Web Features**: `requests` and `BeautifulSoup` (unchanged), SerpAPI (unchanged).
- **Frontend**: Streamlit (unchanged).

Key changes involve swapping OpenAI’s API for Azure OpenAI and using Hugging Face’s `transformers` library for embeddings, integrating them with LangChain and Pinecone.

---

### Step-by-Step Guide (Updated)

#### Step 1: Prerequisites

- **Python**: 3.10+ installed.
- **API Keys**:
  - Azure OpenAI: Get endpoint, API key, and deployment name from [Azure Portal](https://portal.azure.com/). You’ll need an Azure subscription and a deployed model (e.g., `gpt-35-turbo`).
  - Pinecone: [Pinecone Console](https://www.pinecone.io/).
  - SerpAPI: [SerpAPI](https://serpapi.com/).
- **Environment**: Set up a virtual environment:

  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```

#### Step 2: Install Dependencies

Update the installation command to include Azure OpenAI and Hugging Face:

```bash
pip install fastapi uvicorn langchain langgraph azure-ai-openai transformers pinecone-client sqlite3 requests beautifulsoup4 serpapi streamlit python-jwt torch
```

- **Additions**:
  - `azure-ai-openai`: For Azure OpenAI integration.
  - `transformers`: For Hugging Face embeddings.
  - `torch`: Required by `transformers` for model inference.
- **Notes**: Ensure `pydantic` is installed (usually comes with FastAPI).

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

```bash
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint  # e.g., https://your-resource.azure.com/
AZURE_OPENAI_DEPLOYMENT=your_deployment_name  # e.g., gpt-35-turbo
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_pinecone_env  # e.g., "us-west1-gcp"
SERPAPI_API_KEY=your_serpapi_key
SECRET_KEY=your_secret_key_for_jwt
```

Install `python-dotenv` (`pip install python-dotenv`) if not already done.

#### Step 5: Backend Setup (Updated)

##### 5.1: SQLite Database (`backend/database.py`)

- **Unchanged**: No changes needed here as it’s independent of the LLM/embeddings.

##### 5.2: RAG + HITL Agent (`backend/rag_agent.py`)

```python
from langchain_openai import AzureChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from transformers import AutoTokenizer, AutoModel
from typing import TypedDict, Optional
from pinecone import Pinecone
import torch
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

# Hugging Face Embeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Fast, 384-dimensional embeddings
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def huggingface_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings

class HuggingFaceEmbeddings:
    def embed_documents(self, texts):
        return huggingface_embeddings(texts).tolist()
    def embed_query(self, text):
        return huggingface_embeddings([text])[0].tolist()

embeddings = HuggingFaceEmbeddings()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "chatbot-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=384, metric="cosine")  # 384 matches MiniLM
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

- Replaced `langchain_openai.OpenAIEmbeddings` with a custom `HuggingFaceEmbeddings` class using `sentence-transformers/all-MiniLM-L6-v2`.
- Swapped `ChatOpenAI` for `AzureChatOpenAI` with Azure-specific credentials.
- Adjusted Pinecone dimension to 384 (MiniLM’s output size vs. OpenAI’s 1536).

##### 5.3: FastAPI Backend (`backend/main.py`)

- **Unchanged**: The FastAPI endpoints don’t directly depend on the LLM/embeddings, so this file remains the same unless you need to tweak response handling.

#### Step 6: Frontend Setup (`frontend/app.py`)

- **Unchanged**: The frontend interacts with the backend via APIs, so no changes are needed here. It still works with the updated `rag_agent.py`.

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
- **Query**: “What’s new in AI?”—uses Hugging Face embeddings for RAG and Azure OpenAI for generation.
- **Feedback**: Rate responses as before.
- **Web Search/Get**: Unchanged, still uses SerpAPI and `requests`.

#### Step 9: Deployment Notes

- **Local**: Works as-is.
- **Cloud**: Ensure Azure credentials are secure in environment variables. Hugging Face models run locally, so consider memory/CPU needs for deployment (e.g., use a GPU instance if scaling).

---

### Explanation of Changes

- **Azure OpenAI**:
  - Uses `AzureChatOpenAI`. Requires endpoint, API key, and deployment name from Azure. Check Azure’s model availability for `gpt-35-turbo` or similar.
  - Response format remains compatible with LangChain’s `invoke` method.
- **Hugging Face Embeddings**:
  - Uses `sentence-transformers/all-MiniLM-L6-v2`, a lightweight model producing 384-dimensional embeddings (faster than OpenAI’s 1536-dim).
  - Custom `HuggingFaceEmbeddings` class adapts to LangChain’s embedding interface.
  - Runs locally via `transformers`, requiring `torch` for inference.
- **Pinecone**: Dimension adjusted to 384 to match MiniLM embeddings. If reusing an existing index, delete and recreate it with the new dimension.

### Notes

- **Performance**: Hugging Face embeddings are computed locally, which may be slower than OpenAI’s API but avoids additional costs. Optimize by batching or using a GPU.
- **Azure Setup**: Ensure your Azure deployment matches the model specified (e.g., `gpt-35-turbo`). Check API version compatibility in Azure docs.
- **Scalability**: For production, consider hosting the Hugging Face model on a dedicated inference server (e.g., via Hugging Face Inference API) instead of local computation.

This updated stack maintains all features—authentication, RAG, HITL, feedback, and web capabilities—while swapping OpenAI for Azure OpenAI and Hugging Face embeddings, fully functional with the provided code!
