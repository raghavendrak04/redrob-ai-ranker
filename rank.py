#!/usr/bin/env python3
"""
rank.py — Main ranking pipeline for the Redrob Intelligent Candidate
Discovery & Ranking Challenge.

Produces a top-100 CSV with candidate_id, rank, score, and reasoning.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Architecture:
    Stage 1: Honeypot detection (~80 removed)
    Stage 2: Hard filters (reduce to ~10K-20K relevant candidates)
    Stage 3: Semantic similarity pre-screening (narrow to top ~500)
    Stage 4: Deep multi-dimensional scoring
    Stage 5: Final ranking with reasoning generation

Constraints:
    - ≤5 minutes wall-clock on CPU
    - ≤16 GB RAM
    - No GPU, no network
    - Pre-computed embeddings loaded from embeddings.npz
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Ensure our project root is importable
sys.path.insert(0, str(Path(__file__).parent))

from jd_config import JD_TEXT_FOR_EMBEDDING, SCORING_WEIGHTS
from scoring.honeypot_detector import is_honeypot, compute_honeypot_score
from scoring.hard_filters import passes_hard_filter, compute_filter_metadata
from scoring.semantic_scorer import SemanticScorer, TfidfFallbackScorer, build_candidate_text
from scoring.deep_scorer import compute_deep_score
from scoring.behavioral_modifier import get_behavioral_breakdown
from scoring.reasoning_generator import generate_reasoning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_candidates(path: str) -> list[dict]:
    """Load candidates from JSONL or JSON file."""
    p = Path(path)
    candidates = []

    if p.suffix == ".jsonl":
        logger.info(f"Loading JSONL from {path}...")
        with open(p, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        candidates.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed line {i+1}: {e}")
    elif p.suffix == ".json":
        logger.info(f"Loading JSON from {path}...")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                candidates = data
            else:
                candidates = [data]
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")

    logger.info(f"Loaded {len(candidates)} candidates.")
    return candidates


def run_pipeline(
    candidates: list[dict],
    embeddings_path: str = "embeddings.npz",
    top_n_semantic: int = 500,
    output_count: int = 100,
) -> pd.DataFrame:
    """
    Run the full ranking pipeline.

    Args:
        candidates: List of candidate dicts.
        embeddings_path: Path to pre-computed embeddings.
        top_n_semantic: How many candidates to advance past semantic screening.
        output_count: Final output count (100 for submission).

    Returns:
        DataFrame with candidate_id, rank, score, reasoning columns.
    """
    total_start = time.time()

    # Build candidate lookup
    candidate_map = {c["candidate_id"]: c for c in candidates}

    # ════════════════════════════════════════════════════════════════
    # STAGE 1: Honeypot Detection
    # ════════════════════════════════════════════════════════════════
    logger.info("═══ Stage 1: Honeypot Detection ═══")
    stage1_start = time.time()

    honeypot_ids = set()
    for c in candidates:
        if is_honeypot(c):
            honeypot_ids.add(c["candidate_id"])

    remaining = [c for c in candidates if c["candidate_id"] not in honeypot_ids]
    logger.info(
        f"Honeypots detected: {len(honeypot_ids)} | "
        f"Remaining: {len(remaining)} | "
        f"Time: {time.time() - stage1_start:.1f}s"
    )

    # ════════════════════════════════════════════════════════════════
    # STAGE 2: Hard Filters
    # ════════════════════════════════════════════════════════════════
    logger.info("═══ Stage 2: Hard Filters ═══")
    stage2_start = time.time()

    filtered = []
    filter_reasons = {}

    for c in remaining:
        passes, reason = passes_hard_filter(c)
        if passes:
            filtered.append(c)
        else:
            filter_reasons[c["candidate_id"]] = reason

    logger.info(
        f"Passed filters: {len(filtered)} | "
        f"Filtered out: {len(remaining) - len(filtered)} | "
        f"Time: {time.time() - stage2_start:.1f}s"
    )

    # ════════════════════════════════════════════════════════════════
    # STAGE 3: Semantic Similarity Pre-screening
    # ════════════════════════════════════════════════════════════════
    logger.info("═══ Stage 3: Semantic Similarity ═══")
    stage3_start = time.time()

    # Try loading pre-computed embeddings
    semantic_scorer = SemanticScorer(embeddings_path)

    if semantic_scorer.is_fallback:
        # Fall back to TF-IDF if embeddings aren't available
        logger.warning("Using TF-IDF fallback (no pre-computed embeddings).")
        tfidf_scorer = TfidfFallbackScorer(JD_TEXT_FOR_EMBEDDING)
        tfidf_scores = tfidf_scorer.fit_and_score(filtered, JD_TEXT_FOR_EMBEDDING)

        # Sort by TF-IDF score and take top N
        scored_list = sorted(
            tfidf_scores.items(), key=lambda x: x[1], reverse=True
        )[:top_n_semantic]
        semantic_top = [
            (cid, score) for cid, score in scored_list
        ]
        # Store scores for later use
        for cid, score in tfidf_scores.items():
            semantic_scorer.similarity_cache[cid] = score * 2 - 1  # map to [-1,1]
    else:
        # Use pre-computed embeddings
        filtered_ids = [c["candidate_id"] for c in filtered]
        semantic_top = semantic_scorer.get_top_n_by_similarity(
            filtered_ids, n=top_n_semantic
        )

    semantic_top_ids = {cid for cid, _ in semantic_top}

    logger.info(
        f"Top {top_n_semantic} by semantic similarity selected | "
        f"Time: {time.time() - stage3_start:.1f}s"
    )

    # ════════════════════════════════════════════════════════════════
    # STAGE 4: Deep Multi-Dimensional Scoring
    # ════════════════════════════════════════════════════════════════
    logger.info("═══ Stage 4: Deep Scoring ═══")
    stage4_start = time.time()

    scored_candidates = []

    for cid in tqdm(semantic_top_ids, desc="Deep scoring"):
        candidate = candidate_map[cid]
        similarity = semantic_scorer.get_similarity(cid)
        score_breakdown = compute_deep_score(candidate, similarity)

        scored_candidates.append({
            "candidate_id": cid,
            "candidate": candidate,
            "score_breakdown": score_breakdown,
            "final_score": score_breakdown["final_score"],
        })

    # Sort by final score descending
    scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)

    logger.info(
        f"Deep scoring complete for {len(scored_candidates)} candidates | "
        f"Time: {time.time() - stage4_start:.1f}s"
    )

    # ════════════════════════════════════════════════════════════════
    # STAGE 5: Final Ranking & Reasoning
    # ════════════════════════════════════════════════════════════════
    logger.info("═══ Stage 5: Final Ranking & Reasoning ═══")
    stage5_start = time.time()

    # Take top N for output
    top_candidates = scored_candidates[:output_count]

    results = []
    for rank, entry in enumerate(top_candidates, start=1):
        reasoning = generate_reasoning(
            candidate=entry["candidate"],
            rank=rank,
            score_breakdown=entry["score_breakdown"],
        )

        results.append({
            "candidate_id": entry["candidate_id"],
            "rank": rank,
            "score": round(entry["final_score"], 4),
            "reasoning": reasoning,
        })

    logger.info(
        f"Reasoning generated for {len(results)} candidates | "
        f"Time: {time.time() - stage5_start:.1f}s"
    )

    total_time = time.time() - total_start
    logger.info(f"═══ Pipeline complete in {total_time:.1f}s ═══")

    return pd.DataFrame(results)


def write_submission(df: pd.DataFrame, output_path: str) -> None:
    """Write the submission CSV."""
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info(f"Submission written to {output_path}")
    logger.info(f"Rows: {len(df)} | Score range: [{df['score'].min():.4f}, {df['score'].max():.4f}]")


def main():
    parser = argparse.ArgumentParser(
        description="Rank candidates against the JD and produce a submission CSV."
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidates.jsonl or candidates.json",
    )
    parser.add_argument(
        "--out",
        default="submission.csv",
        help="Output CSV path (default: submission.csv)",
    )
    parser.add_argument(
        "--embeddings",
        default="embeddings.npz",
        help="Path to pre-computed embeddings (default: embeddings.npz)",
    )
    parser.add_argument(
        "--top-n-semantic",
        type=int,
        default=500,
        help="How many candidates to advance past semantic screening (default: 500)",
    )

    args = parser.parse_args()

    # Load candidates
    candidates = load_candidates(args.candidates)

    # Run pipeline
    results_df = run_pipeline(
        candidates,
        embeddings_path=args.embeddings,
        top_n_semantic=args.top_n_semantic,
    )

    # Validate
    if len(results_df) != 100:
        logger.warning(
            f"Expected 100 results, got {len(results_df)}. "
            f"Padding or truncating as needed."
        )

    # Write output
    write_submission(results_df, args.out)


if __name__ == "__main__":
    main()
