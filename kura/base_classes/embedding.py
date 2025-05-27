from abc import ABC, abstractmethod


class BaseEmbeddingModel(ABC):
    @property
    @abstractmethod
    def model_batch_size(self) -> int:
        """The default batch size for this embedding model instance."""
        pass

    @property
    @abstractmethod
    def n_concurrent_jobs(self) -> int:
        """The default number of concurrent jobs for this embedding model instance."""
        pass

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into a list of lists of floats"""
        pass
