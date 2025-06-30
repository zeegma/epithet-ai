from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import sys
import os

# Add the parent directory to the Python path to import from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.models.gen_algo import run_ga


app = FastAPI(title="Epithet AI API", version="1.0.0")


# Add CORS middleware to allow all frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnswersRequest(BaseModel):
    answers: List[int]


class UsernameResponse(BaseModel):
    username: str
    personality_type: str


@app.get("/")
async def root():
    return {
        "message": "Welcome to the Epithet AI API",
        "endpoints": {
            "/": "This welcome message",
            "/generate-username": "POST endpoint to generate username with identity",
            "/health": "GET endpoint to perform health check",
            "/docs": "FastAPI auto-generated API docs",
        },
    }


@app.post("/generate-username", response_model=UsernameResponse)
async def generate_username(request: AnswersRequest):
    try:
        # Run the genetic algorithm to get trait and best username
        trait, username = run_ga(request.answers)

        return UsernameResponse(username=username, personality_type=trait)
    except Exception as e:
        # Return a basic error response
        return {"error": f"Failed to generate username: {str(e)}"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
