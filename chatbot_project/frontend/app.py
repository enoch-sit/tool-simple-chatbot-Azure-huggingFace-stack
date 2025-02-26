import streamlit as st
import requests
from dotenv import load_dotenv
import os

load_dotenv()
API_URL = "http://localhost:8000"

def login(username, password):
    response = requests.post(f"{API_URL}/login", json={"username": username, "password": password})
    if response.status_code == 200:
        return response.json()["access_token"]
    return None

def query(token, query_text, user_input=None):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{API_URL}/query", json={"query": query_text, "user_input": user_input}, headers=headers)
    return response.json()

def feedback(token, response, rating):
    headers = {"Authorization": f"Bearer {token}"}
    requests.post(f"{API_URL}/feedback", json={"response": response, "rating": rating}, headers=headers)

def web_search(token, query):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_URL}/web_search", params={"query": query}, headers=headers)
    return response.json()

def web_get(token, url):
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_URL}/web_get", params={"url": url}, headers=headers)
    return response.json()

# Streamlit UI
st.title("Chatbot with HITL")

if "token" not in st.session_state:
    st.session_state.token = None

if not st.session_state.token:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        token = login(username, password)
        if token:
            st.session_state.token = token
            st.success("Logged in!")
        else:
            st.error("Login failed")
else:
    query_text = st.text_input("Enter your query")
    if st.button("Submit"):
        result = query(st.session_state.token, query_text)
        if result.get("status") == "paused":
            st.write("Retrieved Docs:", result["retrieved_docs"])
            user_input = st.text_input("Provide input to continue")
            if st.button("Continue"):
                result = query(st.session_state.token, query_text, user_input)
                st.write("Response:", result["response"])
                rating = st.slider("Rate this response (1-5)", 1, 5)
                if st.button("Submit Feedback"):
                    feedback(st.session_state.token, result["response"], rating)
                    st.success("Feedback submitted!")
        else:
            st.write("Response:", result["response"])
            rating = st.slider("Rate this response (1-5)", 1, 5)
            if st.button("Submit Feedback"):
                feedback(st.session_state.token, result["response"], rating)
                st.success("Feedback submitted!")

    # Web features
    search_query = st.text_input("Web Search Query")
    if st.button("Search Web"):
        results = web_search(st.session_state.token, search_query)
        st.write("Search Results:", results.get("organic_results", [])[:3])

    url = st.text_input("Enter URL to fetch")
    if st.button("Get Web Content"):
        content = web_get(st.session_state.token, url)
        st.write("Content:", content["content"])