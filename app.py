import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Charger les données
data = pd.read_csv("FAQ.csv")

questions = data["question"]
reponses = data["reponse"]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)

def chatbot(user_input):
    user_tfidf = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_tfidf, tfidf_matrix)
    index = np.argmax(similarity)
    score = similarity[0][index]

    if score < 0.3:
        return "Je ne comprends pas la question."
    else:
        return reponses[index]

st.title("🎓 Chatbot Universitaire")

user_input = st.text_input("Posez votre question :")

if user_input:
    response = chatbot(user_input)
    st.write("🤖", response)
