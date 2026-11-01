"""Request and response shapes of the API (validated by Pydantic, shown at /docs)."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, StringConstraints

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class AskRequest(BaseModel):
    # No k, model or reranker options: the served system is the evaluated one, and any extra
    # field is rejected (422) rather than silently ignored.
    model_config = ConfigDict(extra="forbid")

    question: Question


class Citation(BaseModel):
    number: int  # the [n] used in the answer
    chunk_id: str
    paper_id: str
    title: str
    year: int
    section: str
    url: str


class RetrievedChunk(BaseModel):
    rank: int  # 1 = the generator's chunk [1]
    chunk_id: str
    paper_id: str
    title: str
    year: int
    section: str
    url: str
    text: str
    rrf_score: float | None
    bm25_rank: int | None  # None: not in BM25's top fusion_depth
    dense_rank: int | None  # None: not in dense's top fusion_depth


class AskResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    abstained: bool
    citations: list[Citation]  # only the chunks the answer cites, in first-cited order
    invalid_citations: list[int]  # [n] markers pointing to no shown chunk
    retrieved: list[RetrievedChunk]  # all chunks the generator saw
    latency_ms: dict[str, float]  # bm25, dense, fusion, generation, total
    model: str
    cached: bool  # True: answered from the disk cache, no API call


class Generator(BaseModel):
    provider: str
    model: str


class Health(BaseModel):
    status: str  # "ok", or "degraded" when /ask cannot call the generator
    detail: str | None
    evaluated_commit: str
    setup: str
    rrf_k: int
    context_k: int
    reranker: str | None
    embedding_model: str
    generator: Generator
    temperature: float
    chunk_config: str | None
    n_chunks: int
    api_key_set: bool
