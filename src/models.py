from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

# Structure (from the most general to the most precise): Subject > Category > Tags

class Idea(BaseModel):
    content: str # the idea itself
    subject: str
    category: str
    tags: list[str] = []
    embedding: list[float] = []
    created_at: datetime = Field(default_factory=datetime.now)