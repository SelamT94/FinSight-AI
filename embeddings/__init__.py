"""Embeddings, semantic clustering, and pattern detection."""

from .clusterer import SemanticClusterer
from .embed_generator import OllamaEmbedder, TEIEmbedder, create_embedder
from .pattern_detector import PatternDetector

__all__ = [
    "SemanticClusterer",
    "TEIEmbedder",
    "OllamaEmbedder",
    "create_embedder",
    "PatternDetector",
]
