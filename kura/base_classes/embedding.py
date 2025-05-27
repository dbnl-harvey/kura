from abc import ABC, abstractmethod


class BaseEmbeddingModel(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts into a list of lists of floats"""
        pass
