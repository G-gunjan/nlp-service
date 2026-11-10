# SBERT Semantic Similarity Microservice
# This service compares candidate answers with reference answers
# and returns similarity scores for the PlaceReady interview evaluator.

# Import FastAPI to create API endpoints.
from fastapi import FastAPI

# Import BaseModel to validate incoming request data.
from pydantic import BaseModel

# Import SentenceTransformer to generate sentence embeddings.
# Import util to calculate cosine similarity between embeddings.
from sentence_transformers import SentenceTransformer, util


# Create the FastAPI application with the title "PlaceReady NLP".
app = FastAPI(title="PlaceReady NLP")


# Load the pre-trained Sentence-BERT model.
# all-MiniLM-L6-v2 converts sentences into numerical vector representations
# that capture aspects of their semantic meaning.
# The model is loaded once when the service starts.
model = SentenceTransformer("all-MiniLM-L6-v2")


# Define the structure of the incoming request using Pydantic.
class Pairs(BaseModel):

    # Store a list of answer pairs.
    # Each pair contains:
    # pairs[i][0] = candidate's answer
    # pairs[i][1] = reference or expected answer
    # Example: [["HTTP is a protocol", "HTTP is a communication protocol"]]
    pairs: list[list[str]]


# Create a GET endpoint to check whether the NLP service is responding.
@app.get("/health")
def health():

    # Return a JSON response indicating that the endpoint is accessible.
    return {"ok": True}


# Create a POST endpoint to calculate semantic similarity scores.
@app.post("/similarity")
def similarity(body: Pairs):

    # Encode all candidate answers into numerical embeddings.
    # p[0] selects the candidate answer from each pair.
    # "or ''" replaces an empty string or None-like value with an empty string.
    # convert_to_tensor=True returns embeddings as tensors.
    # normalize_embeddings=True normalizes the vectors for cosine similarity.
    a = model.encode(
        [p[0] or "" for p in body.pairs],
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    # Encode all reference answers into numerical embeddings.
    # p[1] selects the reference answer from each pair.
    # The same preprocessing options are used for both sets of answers.
    b = model.encode(
        [p[1] or "" for p in body.pairs],
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    # Calculate the cosine similarity between each matching pair
    # of candidate and reference embeddings.
    # a[i] is the candidate embedding and b[i] is its reference embedding.
    # float(...) converts each tensor result into a standard Python float.
    scores = [
        float(util.cos_sim(a[i], b[i]))
        for i in range(len(body.pairs))
    ]

    # Return the similarity scores as a JSON response.
    # Each score corresponds to the same-position pair in the input.
    return {"scores": scores}