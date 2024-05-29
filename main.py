from loguru import logger
from fastapi import FastAPI

from dto.recommendation_request import RecommendationRequest
from dto.recommendation_response import RecommendationResponse
from service.recommendation_service import RecommendationService
from util.config_loader import ConfigLoader

app = FastAPI()
config_loader = ConfigLoader("config/triagerx_config.yaml")
recommendation_service = RecommendationService(config_loader.get_config())


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/recommendation")
def read_item(request: RecommendationRequest) -> RecommendationResponse:
    logger.debug(f"Received request: {request}")
    recommendations = recommendation_service.get_recommendation(
        request.issue_tile, request.issue_description
    )
    return RecommendationResponse(
        recommended_components=recommendations["predicted_components"],
        recommended_developers=recommendations["combined_ranking"],
    )
