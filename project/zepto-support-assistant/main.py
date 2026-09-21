"""FastAPI wrapper: POST /ask -> validated AskResponse."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from graph import run_query
from ingest import ensure_index
from schemas import AskRequest, AskResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_index()  # build the Chroma index on first start if it doesn't exist
    yield


app = FastAPI(title="Zepto Support Assistant", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return run_query(req.query)
