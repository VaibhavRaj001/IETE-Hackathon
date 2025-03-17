from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import random

# Initialize FastAPI app
app = FastAPI()

# Load sentiment analysis model
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)

# Mapping stars to IMDb rating and sentiment
star_to_imdb = {
    "1 star": (0, 2, "Very Negative"),
    "2 stars": (3, 4, "Negative"),
    "3 stars": (5, 6, "Neutral"),
    "4 stars": (7, 8, "Positive"),
    "5 stars": (9, 10, "Very Positive"),
}

# Define request body
class ReviewRequest(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_review(review: ReviewRequest):
    result = sentiment_pipeline(review.text)[0]
    star_label = result["label"]
    confidence = result["score"]

    # Convert to IMDb rating
    imdb_min, imdb_max, sentiment_category = star_to_imdb[star_label]
    imdb_rating = random.randint(imdb_min, imdb_max)  # Pick a random IMDb rating in the range

    return {
        "review": review.text,
        "sentiment": sentiment_category,
        "imdb_rating": imdb_rating,
        "stars": star_label,
        "confidence": round(confidence, 2)
    }

# Run the API using: uvicorn filename:app --reload
