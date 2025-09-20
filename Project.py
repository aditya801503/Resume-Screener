import streamlit as st
import pdfplumber
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Function: Extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function: Preprocess text and return tokens
def preprocess_and_get_tokens(text):
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if token.is_alpha and not token.is_stop]
    return tokens

# Function: Match Score
def calculate_similarity(resume_text, jd_text):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(similarity * 100, 2)

# Streamlit UI
st.title("📄 AI-Powered Resume Screener")
st.write("Upload resumes and compare with a Job Description (JD)")

# Job Description Input
jd_input = st.text_area("✍️ Paste Job Description here:")

# Resume Upload
uploaded_files = st.file_uploader("📂 Upload Resumes (PDF only)", type="pdf", accept_multiple_files=True)

if st.button("🔍 Analyze") and jd_input and uploaded_files:
    results = []
    jd_tokens = set(preprocess_and_get_tokens(jd_input))
    jd_processed = " ".join(jd_tokens)

    for uploaded_file in uploaded_files:
        resume_text = extract_text_from_pdf(uploaded_file)
        resume_tokens = set(preprocess_and_get_tokens(resume_text))
        resume_processed = " ".join(resume_tokens)

        score = calculate_similarity(resume_processed, jd_processed)

        # Missing Skills
        missing_skills = jd_tokens - resume_tokens
        missing_skills_str = ", ".join(missing_skills) if missing_skills else "None ✅"

        results.append({
            "Resume": uploaded_file.name,
            "Match %": score,
            "Missing Skills": missing_skills_str
        })

    # Convert to DataFrame
    df = pd.DataFrame(results).sort_values(by="Match %", ascending=False)
    st.dataframe(df)

    # Plotting
    st.subheader("📊 Resume Match Visualization")
    fig, ax = plt.subplots()
    ax.barh(df["Resume"], df["Match %"], color="skyblue")
    plt.xlabel("Match %")
    plt.ylabel("Resumes")
    st.pyplot(fig)
