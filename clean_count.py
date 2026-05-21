import pandas as pd
import tiktoken
import re

data = [
    {
        "text": "<p>FastAPI <b>is a Python</b> framework.</p> It's very fast. <https://fastapi.tiangolo.com> look. Using for creat REST API.",
        "source": "docs",
        "category": "tech"
    },
    {
        "text": "   Machine Learning learns from data!!!   It has supervised, unsupervised methods.   Using in many places.   ",
        "source": "textbook",
        "category": "ai"
    }
]

df = pd.DataFrame(data)

def cleaning(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)       
    text = re.sub(r"https?://\S+", "", text)   
    text = re.sub(r"\s+", " ", text) 
    return text.lower().strip()

df["clean_text"] = df["text"].apply(cleaning)

def chunk_text(text: str, chunk_size: int = 150, overlap: int = 25) -> list:
    chunks, start = [], 0

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()

        if chunk:
            chunks.append(chunk)
        
        start += chunk_size - overlap
    
    return chunks

enc = tiktoken.get_encoding("cl100k_base")
chunk_enumerate = 1

for _, row in df.iterrows():
    data_row = chunk_text(row["clean_text"])

    for chunk in data_row:
        token_count = len(enc.encode(chunk))

    print(f"Chunk {chunk_enumerate} | source: {row['source']} | tokens: {token_count} | letters: {len(chunk)} | text: {chunk}")
    chunk_enumerate += 1