# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_model_foundry

from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer, util

from utils.logger import logger


class SemDeDup:
    """
    Semantic Deduplication Module.
    Embeds input text and clusters them to remove redundant examples.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.95):
        """
        Args:
            model_name: The SentenceTransformer model to use.
            threshold: Cosine similarity threshold for clustering.
        """
        self.model_name = model_name
        self.threshold = threshold
        # Lazy load model to avoid overhead if not used or during init
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def prune(self, data: List[Dict[str, Any]], key_fields: List[str]) -> List[Dict[str, Any]]:
        """
        Prunes the dataset by clustering semantically similar examples.

        Args:
            data: List of data items.
            key_fields: List of keys in the dict to combine for embedding (e.g. ["instruction", "input"]).

        Returns:
            A pruned list of data items (representatives).
        """
        if not data:
            return []

        logger.info(f"Starting SemDeDup on {len(data)} examples.")

        # 1. Extract texts to embed
        texts = []
        for item in data:
            # Join values of key fields to form the semantic representation
            text_parts = [str(item.get(k, "")) for k in key_fields if item.get(k)]
            texts.append(" ".join(text_parts))

        # 2. Embed
        logger.info("Embedding data...")
        embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)

        # 3. Cluster
        # We use community detection or simple pairwise clustering.
        # Fast clustering:
        # https://www.sbert.net/docs/package_reference/util.html#sentence_transformers.util.community_detection
        # But community_detection might be too aggressive or slow for huge datasets.
        # The prompt says: "Cluster: Identifies examples with Cosine Similarity > 0.95. Prune: Keeps only the top 1".

        logger.info(f"Clustering with threshold {self.threshold}...")
        clusters = util.community_detection(embeddings, threshold=self.threshold, min_community_size=1)

        # 4. Prune (Keep first element of each cluster)
        pruned_data = []
        for cluster in clusters:
            # cluster is a list of indices, we pick the first one as representative
            representative_idx = cluster[0]
            pruned_data.append(data[representative_idx])

        logger.info(f"Pruned {len(data)} -> {len(pruned_data)} examples.")
        return pruned_data
