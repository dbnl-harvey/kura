from kura.base_classes import BaseEmbeddingModel
from asyncio import Semaphore, gather
from tenacity import retry, wait_fixed, stop_after_attempt
from openai import AsyncOpenAI


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_batch_size: int = 50, n_concurrent_jobs: int = 5):
        self.client = AsyncOpenAI()
        self._model_batch_size = model_batch_size
        self._n_concurrent_jobs = n_concurrent_jobs
        self._semaphore = Semaphore(n_concurrent_jobs)

    @property
    def model_batch_size(self) -> int:
        return self._model_batch_size

    @property
    def n_concurrent_jobs(self) -> int:
        return self._n_concurrent_jobs

    @retry(wait=wait_fixed(3), stop=stop_after_attempt(3))
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a single batch of texts."""
        async with self._semaphore:
            resp = await self.client.embeddings.create(
                input=texts, model="text-embedding-3-small"
            )
            return [item.embedding for item in resp.data]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # Create batches
        batches = _batch_texts(texts, self.model_batch_size)

        # Process all batches concurrently
        tasks = [self._embed_batch(batch) for batch in batches]
        results_list_of_lists = await gather(*tasks)

        # Flatten results
        embeddings = []
        for result_batch in results_list_of_lists:
            embeddings.extend(result_batch)

        return embeddings


def _batch_texts(texts: list[str], batch_size: int) -> list[list[str]]:
    """Helper function to divide a list of texts into batches."""
    if not texts:
        return []

    batches = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batches.append(batch)
    return batches


class SentenceTransformerEmbeddingModel(BaseEmbeddingModel):
    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", model_batch_size: int = 128
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self._model_batch_size = model_batch_size

    @property
    def model_batch_size(self) -> int:
        return self._model_batch_size

    @retry(wait=wait_fixed(3), stop=stop_after_attempt(3))
    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # Create batches
        batches = _batch_texts(texts, self.model_batch_size)

        # Process all batches
        embeddings = []
        for batch in batches:
            batch_embeddings = self.model.encode(batch).tolist()
            embeddings.extend(batch_embeddings)

        return embeddings
