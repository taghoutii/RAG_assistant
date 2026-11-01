"""FastAPI app: POST /ask and GET /health over the frozen RAG system.

Run (PowerShell, from the repo root):
    uvicorn ragpapers.api.app:app --port 8000
then open http://localhost:8000/docs. Use ONE worker: the local Qdrant index can be opened
by a single process only.

The indexes and the generator are loaded once at startup (the lifespan below), not per
request, and one warm-up retrieval loads the embedding model before the first question.
Errors: 422 invalid question, 503 no API key or daily quota used up, 502 the generator API
failed after its retries.
"""

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from ragpapers.api.schemas import AskRequest, AskResponse, Health
from ragpapers.api.service import GeneratorFailed, RagService, ServiceUnavailable
from ragpapers.config import load_config

logger = logging.getLogger("uvicorn.error")  # uvicorn's logger: shown in the server output


def default_service() -> RagService:
    return RagService.from_config(load_config())


def create_app(service_factory: Callable[[], RagService] = default_service) -> FastAPI:
    """The app; tests pass a factory that builds a service over fake indexes."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service_factory()
        # Warm-up before serving: /health answers only after it, so Docker's healthcheck and
        # the first user both see a warm system.
        warm_ms = app.state.service.warm_up()
        logger.info("Warm-up retrieval done in %.0f ms (no LLM call, not logged)", warm_ms)
        try:
            yield
        finally:
            app.state.service.close()

    app = FastAPI(
        title="ml-papers-rag",
        description="Cited answers from a curated corpus of AI/ML papers (evaluated RAG).",
        lifespan=lifespan,
    )

    # Plain `def` endpoints: FastAPI runs them in a worker thread, so the CPU-bound retrieval
    # and the blocking LLM call never stall the event loop.
    @app.get("/health", response_model=Health)
    def health(request: Request) -> dict:
        return request.app.state.service.health()

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request) -> dict:
        try:
            return request.app.state.service.ask(body.question)
        except ServiceUnavailable as err:
            raise HTTPException(status_code=503, detail=str(err)) from err
        except GeneratorFailed as err:
            raise HTTPException(status_code=502, detail=str(err)) from err

    return app


app = create_app()  # loads nothing until the server starts
