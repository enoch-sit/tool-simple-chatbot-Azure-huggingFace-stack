from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
import jwt
from dotenv import load_dotenv
import os
from database import authenticate_user, save_feedback
from rag_agent import run_agent
import requests
from bs4 import BeautifulSoup

load_dotenv()
app = FastAPI()
SECRET_KEY = os.getenv("SECRET_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_API_KEY")

class User(BaseModel):
    username: str
    password: str

class Feedback(BaseModel):
    response: str
    rating: int

class Query(BaseModel):
    query: str
    user_input: Optional[str] = None

def get_current_user(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        username = payload.get("sub")
        if not username:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/login")
async def login(user: User):
    if authenticate_user(user.username, user.password):
        token = jwt.encode({"sub": user.username}, SECRET_KEY, algorithm="HS256")
        return {"access_token": token}
    raise HTTPException(status_code=401, detail="Invalid credentials")

@app.post("/query")
async def query(query: Query, token: str = Depends(get_current_user)):
    result = run_agent(query.query, query.user_input)
    if "response" not in result:  # Paused for human input
        return {"status": "paused", "retrieved_docs": result["retrieved_docs"]}
    return {"response": result["response"]}

@app.post("/feedback")
async def feedback(feedback: Feedback, token: str = Depends(get_current_user)):
    save_feedback(token, feedback.response, feedback.rating)
    return {"status": "feedback saved"}

@app.get("/web_search")
async def web_search(query: str, token: str = Depends(get_current_user)):
    url = f"https://serpapi.com/search?api_key={SERPAPI_KEY}&q={query}"
    response = requests.get(url)
    return response.json()

@app.get("/web_get")
async def web_get(url: str, token: str = Depends(get_current_user)):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    return {"content": soup.get_text()[:500]}  # Limited for brevity