
from fastapi import FastAPI, File, UploadFile
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import numpy as np
import io
import uvicorn
import os
# Initialize FastAPI
app = FastAPI()

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

df = pd.read_csv('linkedin_jobs_processed.csv')

model = SentenceTransformer('all-MiniLM-L6-v2')
lemmatizer = WordNetLemmatizer()

df['job_embeddings'] = df['cleaned_description'].apply(lambda x: model.encode(x))

def preprocess_text(text):
    if pd.isna(text):
        return ''
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfFileReader(io.BytesIO(pdf_file))
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text() + ' '
    return text.strip()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/match_jobs/")
async def match_jobs(file: UploadFile = File(...)):
    file_bytes = await file.read()
    resume_text = extract_text_from_pdf(file_bytes)
    
    resume_text = preprocess_text(resume_text)
    resume_embedding = model.encode(resume_text)
    
    df['similarity'] = df['job_embeddings'].apply(lambda x: util.cos_sim(resume_embedding, x).item())
    
    top_matches = df.sort_values(by='similarity', ascending=False).head(5)
    
    return top_matches[["title", "company_name", "location", "similarity"]].to_dict(orient="records")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Get PORT from Render, default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)