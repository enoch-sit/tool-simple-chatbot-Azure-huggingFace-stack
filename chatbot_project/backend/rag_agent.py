from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain.llms.base import LLM
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List
from pinecone import Pinecone
import requests
import os
from dotenv import load_dotenv
#import pdb; pdb.set_trace()

from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

load_dotenv()

# Custom Grok LLM for LangChain
class GrokLLM(LLM):
    api_key: Optional[str] = None
    api_url: str = "https://api.x.ai/v1/chat/completions"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key=api_key)  # Pass parameters to super().__init__
        self.api_key = api_key

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "grok",  # Adjust based on actual Grok model name
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        if response.status_code != 200:
            raise Exception(f"Grok API error: {response.text}")
        data = response.json()
        return data["choices"][0]["message"]["content"]

    @property
    def _llm_type(self) -> str:
        return "grok"

# Initialize Grok LLM
llm = GrokLLM(api_key=os.getenv("GROK_API_KEY"))

# Hugging Face Inference API Embeddings
#HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/all-MiniLM-L6-v2"
HF_API_KEY = os.getenv("HF_API_KEY")

def huggingface_inference_embeddings(texts):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": texts}
    # payload = {
    #     "inputs": {
    #         "source_sentence": "This is an apple",
    #         "sentences": texts
    #     }
    # }
    #breakpoint()
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"HF API error: {response.text}")
        #breakpoint()
    embeddings = response.json()
    #breakpoint()
    if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], list):
        return embeddings
    #breakpoint()
    raise Exception("Unexpected HF API response format")

class HuggingFaceInferenceEmbeddings:
    def embed_documents(self, texts):
        return huggingface_inference_embeddings(texts)
    def embed_query(self, text):
        return huggingface_inference_embeddings([text])[0]

embeddings = HuggingFaceInferenceEmbeddings()

# Function for mean pooling
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Local Hugging Face Embeddings
class LocalHuggingFaceEmbeddings:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    def embed_documents(self, texts):
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        return F.normalize(sentence_embeddings, p=2, dim=1).tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]

# Initialize embeddings with local inference
localEmbeddings = LocalHuggingFaceEmbeddings()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "rag"
if index_name not in pc.list_indexes().names():
    pc.create_index(name=index_name, dimension=384, metric="cosine")  # 384 for MiniLM
vector_store = PineconeVectorStore(index_name=index_name, embedding=localEmbeddings)

# Sample data for RAG
# Sample data for RAG
documents = [
    "That is a happy person",         # source_sentence
    "That is a happy dog",            # sentences[0]
    "That is a very happy person",    # sentences[1]
    "Today is a sunny day"            # sentences[2]
]
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
    response = llm(prompt.format(query=state["query"], 
                                 docs="\n".join(state["retrieved_docs"]), 
                                 user_input=state["user_input"] or "None"))
    state["response"] = response
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