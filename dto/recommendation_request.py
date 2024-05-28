from pydantic import BaseModel


class RecommendationRequest(BaseModel):
    issue_tile: str
    issue_description: str
