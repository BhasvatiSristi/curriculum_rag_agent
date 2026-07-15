from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    mode: str = "hybrid"


class ChatResponse(BaseModel):
    answer: str