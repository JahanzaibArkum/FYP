import re
import os
import faiss
import emoji
import torch
import numpy as np
import pandas as pd
from groq import Groq
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv


# Load .env file
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load Hadith CSV
hadith_data = pd.read_csv("combined_file.csv")
hadith_data = hadith_data[["Hadith English", "Hadith Arabic", "Reference"]]
hadith_list = hadith_data.to_dict(orient="records")

# Load FAISS index
index = faiss.read_index("Full_hadith_index.faiss")


# Streamlit UI
st.title(emoji.emojize("🕌 Hadith Navigator Chatbot"))
st.markdown("Type your question below and get answers with references from Hadith.")




if "conversations" not in st.session_state:
    st.session_state.conversations = []

if "current_convo_index" not in st.session_state:
    st.session_state.current_convo_index = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.title("📚 Your Conversations")

st.sidebar.markdown("---")
# ➕ Start new chat option
if st.sidebar.button("➕ Start New Chat"):
    st.session_state.chat_history = [
        {
            "role": "system",
            "content": (
                "You are a Hadith scholar AI assistant. You must answer only using the retrieved Hadiths. "
                "Always include the reference for every response. Do not provide information that is not in the retrieved Hadiths."
            )
        }
    ]
    st.session_state.current_convo_index = None


for i, convo in enumerate(st.session_state.conversations):
    title_key = f"title_{i}"
    new_title = st.sidebar.text_input(" ", value=convo["title"], key=title_key)

    # Update title
    st.session_state.conversations[i]["title"] = new_title

    if st.sidebar.button(f"🔄 {new_title}", key=f"load_{i}"):
        st.session_state.chat_history = convo["history"]
        st.session_state.current_convo_index = i



# Initialize session history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "system",
            "content": (
                "You are a Hadith scholar AI assistant. You must answer only using the retrieved Hadiths. "
                "Always include the reference for every response. Do not provide information that is not in the retrieved Hadiths."
            )
        }
    ]


# 👇 Decide if current query is a follow-up
def decide_if_followup(chat_history, current_query):
    messages = [
        {"role": "system", "content": "You're a helpful assistant that determines whether a question is a follow-up or a standalone question."},
        {"role": "user", "content": f"""Chat history:
{chat_history}

Current question:
{current_query}

Does this current question depend on the above conversation? Answer with 'Yes' or 'No' only."""}
    ]

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        max_tokens=10,
        temperature=0,
    )

    answer = response.choices[0].message.content.strip().lower()
    return 'yes' in answer


def generate_title_from_question(question):
    prompt = [
        {"role": "system", "content": "Summarize the user question into 4-6 words that describe the topic."},
        {"role": "user", "content": question}
    ]
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=prompt,
        max_tokens=20,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip().capitalize()


# 🔍 Search function
def search_hadith(query, top_k=3):
    # Check if follow-up
    is_followup = decide_if_followup(st.session_state.chat_history[-5:], query)

    if is_followup:
        rephrase_prompt = [
            {"role": "system", "content": "You are a helpful assistant that rewrites follow-up questions as standalone questions."},
            {"role": "user", "content": f"Chat history:\n{st.session_state.chat_history[-5:]}\n\nCurrent question:\n{query}\n\nRephrase the current question using the context."}
        ]

        rephrase_response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=rephrase_prompt,
            max_tokens=700,
            temperature=0.5,
        )

        query = rephrase_response.choices[0].message.content.strip()

    # Embed and retrieve
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding).astype('float32'), k=top_k)
    top_hadiths = [hadith_list[i] for i in I[0]]
    return top_hadiths, query


user_input = st.chat_input("Ask Anything About Islam!")
with st.spinner("Generating answer..."):
    # 🧠 Handle user query
    if user_input:
        top_matches, processed_query = search_hadith(user_input)

        # Build context
        context = "\n\n".join([
            f"{i+1}. {h['Hadith English']} (Reference: {h['Reference']})"
            for i, h in enumerate(top_matches)
        ])

        # Update chat history with user + context
        user_message = f"User Question: {processed_query}\n\nUse the following Hadiths to answer:\n{context}"
        st.session_state.chat_history.append({"role": "user", "content": user_message})

        # Get LLM answer
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=st.session_state.chat_history,
            max_tokens=700,
            temperature=1.0,
        )

        assistant_reply = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
         
         # 🔖 If it's a new conversation
        if st.session_state.current_convo_index is None:
            title = generate_title_from_question(user_input)
            st.session_state.conversations.append({
                "title": title,
                "history": st.session_state.chat_history.copy()
            })
            st.session_state.current_convo_index = len(st.session_state.conversations) - 1
        else:
            # Update existing conversation
            st.session_state.conversations[st.session_state.current_convo_index]["history"] = st.session_state.chat_history.copy() 

# 💬 Show chat in order with scrollable style
for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f"**You:** {message['content']}")
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(f"**HadithBot:** {message['content']}")

