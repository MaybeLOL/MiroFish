"""
Voyage AI embedding service wrapper.
Uses voyage-3 model (1024-dim) for semantic search embeddings.
"""

import httpx
from typing import List
from ..utils.logger import get_logger

logger = get_logger(__name__)


class VoyageEmbedding:
    MODEL = "voyage-3"
    API_URL = "https://api.voyageai.com/v1/embeddings"
    MAX_BATCH = 128

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("VOYAGE_API_KEY not configured")
        self.api_key = api_key

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts. Batches automatically."""
        all_embeddings = []
        for i in range(0, len(texts), self.MAX_BATCH):
            batch = texts[i:i + self.MAX_BATCH]
            resp = httpx.post(
                self.API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.MODEL, "input": batch},
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()
            batch_embeddings = [item["embedding"] for item in data["data"]]
            all_embeddings.extend(batch_embeddings)
        return all_embeddings

    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.embed([text])[0]
