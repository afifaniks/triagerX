from typing import List

from pydantic import BaseModel


class RecommendationResponse(BaseModel):
    recommended_developers: List[str]
