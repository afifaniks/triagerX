from loguru import logger
from fastapi import FastAPI

from dto.recommendation_request import RecommendationRequest
from dto.recommendation_response import RecommendationResponse
from service.recommendation_service import RecommendationService
from util.config_loader import ConfigLoader


config_loader = ConfigLoader("config/triagerx_config.yaml")
configs = config_loader.get_config()

app = FastAPI(
    title=configs["project"]["title"], version=configs["project"]["api_version"]
)

logger.debug("Initializing recommendation service...")
recommendation_service = RecommendationService(configs)


@app.get("/")
def health_check():
    return {"status": 200}


@app.post(
    "/recommendation",
    description="Get component and developer recommendations for Openj9 Issues",
)
def get_recommendation(request: RecommendationRequest) -> RecommendationResponse:
    logger.debug(f"Received request: {request}")
    recommendations = recommendation_service.get_recommendation(
        request.issue_title, request.issue_description
    )
    return RecommendationResponse(
        recommended_components=recommendations["predicted_components"],
        recommended_developers=recommendations["combined_ranking"],
    )
