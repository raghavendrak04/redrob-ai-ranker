"""
semantic_scorer.py — Embedding-based semantic similarity scoring.

Loads pre-computed embeddings (from precompute_embeddings.py) and computes
cosine similarity between each candidate's text and the JD embedding.
Falls back to TF-IDF if embeddings are unavailable.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class SemanticScorer:
    """Manages embedding loading and similarity computation."""

    def __init__(self, embeddings_path: Optional[str] = None):
        """
        Args:
            embeddings_path: Path to pre-computed .npz file.
                             If None or missing, falls back to TF-IDF.
        """
        self.embeddings = None
        self.jd_embedding = None
        self.candidate_ids = None
        self.similarity_cache = {}
        self._fallback_mode = False

        if embeddings_path:
            self._load_embeddings(embeddings_path)

    def _load_embeddings(self, path: str) -> None:
        """Load pre-computed embeddings from .npz file."""
        p = Path(path)
        if not p.exists():
            logger.warning(
                f"Embeddings file not found at {path}. "
                "Will use TF-IDF fallback."
            )
            self._fallback_mode = True
            return

        try:
            data = np.load(path, allow_pickle=True)
            self.embeddings = data["candidate_embeddings"]
            self.jd_embedding = data["jd_embedding"]
            self.candidate_ids = list(data["candidate_ids"])
            logger.info(
                f"Loaded embeddings: {self.embeddings.shape[0]} candidates, "
                f"dim={self.embeddings.shape[1]}"
            )

            # Pre-compute all similarities at once (vectorized)
            jd_vec = self.jd_embedding.reshape(1, -1)
            similarities = cosine_similarity(jd_vec, self.embeddings)[0]

            # Build lookup: candidate_id → similarity
            self.similarity_cache = dict(
                zip(self.candidate_ids, similarities.tolist())
            )
            logger.info("Pre-computed all cosine similarities.")

        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            self._fallback_mode = True

    def get_similarity(self, candidate_id: str) -> float:
        """
        Get the semantic similarity score for a candidate.

        Args:
            candidate_id: The CAND_XXXXXXX identifier.

        Returns:
            Cosine similarity score in [0, 1] (shifted from [-1, 1]).
        """
        if candidate_id in self.similarity_cache:
            raw = self.similarity_cache[candidate_id]
            # Normalize from [-1, 1] to [0, 1]
            return max(0.0, min(1.0, (raw + 1.0) / 2.0))
        return 0.5  # neutral fallback

    def get_top_n_by_similarity(
        self, candidate_ids: list[str], n: int = 500
    ) -> list[tuple[str, float]]:
        """
        Return the top-N candidates from the given list, ranked by
        semantic similarity to the JD.

        Args:
            candidate_ids: List of candidate IDs to consider.
            n: Number of top candidates to return.

        Returns:
            List of (candidate_id, similarity_score) tuples, sorted
            descending by score.
        """
        scored = []
        for cid in candidate_ids:
            sim = self.get_similarity(cid)
            scored.append((cid, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    @property
    def is_fallback(self) -> bool:
        return self._fallback_mode


class TfidfFallbackScorer:
    """
    TF-IDF based scorer as fallback when embeddings are not available.
    Less accurate than sentence-transformers but runs without any
    pre-computation.
    """

    def __init__(self, jd_text: str):
        self.jd_text = jd_text
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._fitted = False
        self._jd_vec = None

    def fit_and_score(
        self, candidates: list[dict], jd_text: str
    ) -> dict[str, float]:
        """
        Compute TF-IDF similarity scores for all candidates.

        Args:
            candidates: List of candidate dicts.
            jd_text: The job description text.

        Returns:
            Dict mapping candidate_id → similarity score [0, 1].
        """
        texts = []
        ids = []

        for c in candidates:
            text = _build_candidate_text(c)
            texts.append(text)
            ids.append(c["candidate_id"])

        # Fit on all texts + JD
        all_texts = [jd_text] + texts
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)

        jd_vec = tfidf_matrix[0:1]
        candidate_vecs = tfidf_matrix[1:]

        similarities = cosine_similarity(jd_vec, candidate_vecs)[0]

        return dict(zip(ids, similarities.tolist()))


def _build_candidate_text(candidate: dict) -> str:
    """
    Build a composite text representation of a candidate for embedding
    or TF-IDF similarity computation.
    """
    parts = []
    profile = candidate.get("profile", {})

    # Headline and summary
    if profile.get("headline"):
        parts.append(profile["headline"])
    if profile.get("summary"):
        parts.append(profile["summary"])

    # Current role info
    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    industry = profile.get("current_industry", "")
    if title:
        parts.append(f"Current role: {title} at {company} in {industry}")

    # Career history
    for entry in candidate.get("career_history", []):
        role_text = f"{entry.get('title', '')} at {entry.get('company', '')}"
        desc = entry.get("description", "")
        if desc:
            role_text += f": {desc}"
        parts.append(role_text)

    # Skills
    skill_names = [
        s.get("name", "") for s in candidate.get("skills", [])
        if s.get("name")
    ]
    if skill_names:
        parts.append("Skills: " + ", ".join(skill_names))

    # Education
    for edu in candidate.get("education", []):
        edu_text = (
            f"{edu.get('degree', '')} in {edu.get('field_of_study', '')} "
            f"from {edu.get('institution', '')}"
        )
        parts.append(edu_text)

    # Certifications
    for cert in candidate.get("certifications", []):
        parts.append(f"Certified: {cert.get('name', '')} by {cert.get('issuer', '')}")

    return " . ".join(parts)


# Export for use in precompute_embeddings.py
build_candidate_text = _build_candidate_text
