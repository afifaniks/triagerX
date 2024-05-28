from loguru import logger
from fastapi import FastAPI

from dto.recommendation_request import RecommendationRequest
from dto.recommendation_response import RecommendationResponse

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/recommendation")
def read_item(request: RecommendationRequest) -> RecommendationResponse:
    logger.debug(f"Received request: {request}")
    return RecommendationResponse(
        recommended_components=["com1", "comp2"],
        recommended_developers=["dev1", "dev2", "dev3"]
    )