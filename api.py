from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from search import FoodSearchService, MAX_TOP_K


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.search_service = FoodSearchService()
    try:
        yield
    finally:
        app.state.search_service.close()


app = FastAPI(title="CraveAI Search API", version="1.0.0", lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=MAX_TOP_K)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/search",
    responses={422: {"description": "Invalid query or top_k value"}},
)
def search(request: SearchRequest, http_request: Request) -> list[dict]:
    try:
        return http_request.app.state.search_service.search(
            request.query,
            request.top_k,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error