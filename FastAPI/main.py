# fastapi_serve.py

from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tokenization import tokenize
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your front-end URL(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Load TF-IDF vectorizer and cosine similarity matrix using pickle
with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf_vectorizer = pickle.load(file)

with open('cosine_sim.pkl', 'rb') as file:
    cosine_sim = pickle.load(file)

with open('df.pkl', 'rb') as file:
    df = pickle.load(file)
    
with open('tfidf_matrix.pkl', 'rb') as file:
    tfidf_matrix = pickle.load(file)

class Item(BaseModel):
    query: str
    top_n: int = 3

def get_top_similar_products(query: str, top_n: int = 3):
    query_vector = tfidf_vectorizer.transform([query])
    cosine_similarities = linear_kernel(query_vector, tfidf_matrix).flatten()
    top_indices = cosine_similarities.argsort()[:-top_n-1:-1]
    return df.iloc[top_indices][['title', 'description', 'price']]

@app.post("/recommend/")
def recommend(item: Item):
    try:
        recommendations = get_top_similar_products(item.query, item.top_n)
        return recommendations.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    
    uvicorn.run(app, host="127.0.0.1", port=8000)

