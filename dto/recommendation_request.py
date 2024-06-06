from pydantic import BaseModel


class RecommendationRequest(BaseModel):
    issue_title: str
    issue_description: str
