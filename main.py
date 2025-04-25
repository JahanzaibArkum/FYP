import os
import pandas as pd
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from groq import Groq
import emoji
import torch

# ------------------- Environment Setup -------------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# ------------------- Device and Models -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# ------------------- Load Data & FAISS Index -------------------
hadith_data = pd.read_csv("combined_file.csv")
hadith_data = hadith_data[["Hadith English", "Hadith Arabic", "Reference"]]
hadith_list = hadith_data.to_dict(orient="records")
index = faiss.read_index("Full_hadith_index.faiss")

# ------------------- Streamlit Page Config -------------------
st.set_page_config(
    page_title=emoji.emojize("📜 Hadith Navigator Chatbot"),
    page_icon=emoji.emojize("📖"),
    layout="wide"
)

# ------------------- Session State -------------------
if "user_profile" not in st.session_state:
    st.session_state.user_profile = {
        "name": "",
        "threads": [{
            "chat_history": [],
            "preferred_topics": [],
            "last_query": "",
            "last_response": ""
        }],
        "current_thread": 0
    }

# ------------------- Retrieval Function -------------------
def retrieve_top_k_hadiths(query, k=10, similarity_threshold=0.4):
    query_embedding = embedding_model.encode([query]).astype("float32")

    index_size = index.ntotal
    embeddings = np.zeros((index_size, query_embedding.shape[1]), dtype="float32")
    index.reconstruct_n(0, index_size, embeddings)

    similarities = cosine_similarity(query_embedding, embeddings).flatten()

    relevant_hadiths = [
        {**hadith_list[idx], "similarity_score": sim_score}
        for idx, sim_score in enumerate(similarities)
        if sim_score >= similarity_threshold
    ]

    sorted_hadiths = sorted(relevant_hadiths, key=lambda x: x["similarity_score"], reverse=True)
    return sorted_hadiths[:k]

# ------------------- Follow-up Detection -------------------
def is_follow_up_query(current_query, last_query, threshold=0.6):
    current_embedding = embedding_model.encode([current_query]).astype("float32")
    last_embedding = embedding_model.encode([last_query]).astype("float32")
    similarity_score = cosine_similarity(current_embedding, last_embedding).flatten()[0]
    return similarity_score >= threshold

# ------------------- Chatbot Core (RAG Style) -------------------
def generate_response(query):
    user_profile = st.session_state.user_profile
    thread = user_profile["threads"][user_profile["current_thread"]]

    clarification_note = ""
    if thread["last_query"] and is_follow_up_query(query, thread["last_query"]):
        query = thread["last_query"]
        clarification_note = "You asked for clarification on the previous question."

    # Retrieve top-k relevant hadiths (retrieval step)
    context_hadiths = retrieve_top_k_hadiths(query)
    if not context_hadiths:
        return emoji.emojize("⚠️ No relevant Hadith found.")

    thread["chat_history"].append(query)

    # Update preferred topics dynamically
    if "patience" in query.lower() and "Patience" not in thread["preferred_topics"]:
        thread["preferred_topics"].append("Patience")
    if "faith" in query.lower() and "Faith" not in thread["preferred_topics"]:
        thread["preferred_topics"].append("Faith")

    previous_context = ""
    if thread["last_query"] and thread["last_response"]:
        previous_context = f"""
Previous User Query: {thread['last_query']}
Previous Bot Response: {thread['last_response']}
"""

    preferred_topics_text = f"User's preferred topics: {', '.join(thread['preferred_topics'])}\n"

    # Format hadith context for prompt (context injection)
    hadith_context = "\n\n".join([
        f"### Hadith {i+1}:\n{(h['Hadith English'][:1000] + '...') if len(h['Hadith English']) > 1000 else h['Hadith English']}\n**Reference**: {h['Reference']}"
        for i, h in enumerate(context_hadiths)
    ])

    prompt = f"""
You are an Islamic chatbot. Respond based only on the provided Hadiths below.

User: {user_profile['name']}

{previous_context}

Context (Hadiths):
{hadith_context}

{preferred_topics_text}
{clarification_note}

User Query: {query}

Response:
"""

    # Generate final response (generation step)
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192"
    )

    response = chat_completion.choices[0].message.content

    thread["last_query"] = query
    thread["last_response"] = response

    if len(thread["chat_history"]) > 15:
        user_profile["threads"].append({
            "chat_history": [],
            "preferred_topics": thread["preferred_topics"].copy(),
            "last_query": "",
            "last_response": ""
        })
        user_profile["current_thread"] += 1

    hadith_output = "\n\n".join([
        f"**Hadith:** {h['Hadith English']}  \n**Reference:** {h['Reference']}  \n**Similarity Score:** {h['similarity_score']:.3f}"
        for h in context_hadiths
    ])

    final_output = f"## 🧠 Response:\n{response}\n\n---\n\n### 📚 Hadiths Used:\n{hadith_output}"
    return emoji.emojize(final_output)

# ------------------- Streamlit UI -------------------
st.title(emoji.emojize("📜 Hadith Navigator Chatbot"))
st.markdown(emoji.emojize("🚀 **Ask Hadith-related questions and get authentic, referenced answers.**"))

# Sidebar
with st.sidebar:
    st.title(emoji.emojize("👤 User Info"))
    name = st.text_input("Enter your name:", value=st.session_state.user_profile.get("name", ""))
    st.session_state.user_profile["name"] = name

    st.markdown(emoji.emojize("### 💬 Chat History"))
    thread = st.session_state.user_profile["threads"][st.session_state.user_profile["current_thread"]]
    for chat in thread["chat_history"]:
        st.markdown(emoji.emojize(f"🖌️ {chat}"))

# User Query Input
query = st.text_input(
    emoji.emojize("🔣 Ask a Hadith-related question:"),
    placeholder="Ask something like: What does Islam say about patience?"
)

# Handle Query
if query:
    with st.spinner(emoji.emojize("⏳ Processing your query...")):
        result = generate_response(query)
    st.markdown(result, unsafe_allow_html=True)
