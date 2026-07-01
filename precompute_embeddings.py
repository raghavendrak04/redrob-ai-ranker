#!/usr/bin/env python3
"""
precompute_embeddings.py — Pre-compute sentence-transformer embeddings
for all candidates and the JD.

This runs ONCE (can take 15-30 min on CPU) and saves the embeddings
to a .npz file that the ranking step loads in seconds.

Usage:
    python precompute_embeddings.py \
        --candidates ./candidates.jsonl \
        --output ./embeddings.npz \
        --model all-MiniLM-L6-v2 \
        --batch-size 256

The ranking step (rank.py) loads these pre-computed embeddings and
completes in well under 5 minutes.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Ensure our project root is importable
sys.path.insert(0, str(Path(__file__).parent))

from scoring.semantic_scorer import build_candidate_text
from jd_config import JD_TEXT_FOR_EMBEDDING

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
            for line in f:
                line = line.strip()
                if line:
                    candidates.append(json.loads(line))
    elif p.suffix == ".json":
        logger.info(f"Loading JSON from {path}...")
        with open(p, "r", encoding="utf-8") as f:
            candidates = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")

    logger.info(f"Loaded {len(candidates)} candidates.")
    return candidates


def compute_embeddings(
    candidates: list[dict],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    output_path: str = "embeddings.npz",
) -> None:
    """
    Compute and save embeddings for all candidates and the JD.

    Args:
        candidates: List of candidate dicts.
        model_name: Sentence-transformer model name.
        batch_size: Batch size for encoding.
        output_path: Path to save the .npz file.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error(
            "sentence-transformers not installed. "
            "Run: pip install sentence-transformers"
        )
        sys.exit(1)

    logger.info(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)

    # ── Build candidate texts ──
    logger.info("Building candidate text representations...")
    candidate_ids = []
    candidate_texts = []

    for c in tqdm(candidates, desc="Building texts"):
        candidate_ids.append(c["candidate_id"])
        candidate_texts.append(build_candidate_text(c))

    # ── Encode JD ──
    logger.info("Encoding job description...")
    jd_embedding = model.encode(
        [JD_TEXT_FOR_EMBEDDING],
        show_progress_bar=False,
        normalize_embeddings=True,
    )[0]
    logger.info(f"JD embedding shape: {jd_embedding.shape}")

    # ── Encode candidates in batches ──
    logger.info(f"Encoding {len(candidate_texts)} candidates (batch_size={batch_size})...")
    start_time = time.time()

    candidate_embeddings = model.encode(
        candidate_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    elapsed = time.time() - start_time
    logger.info(
        f"Encoding complete in {elapsed:.1f}s "
        f"({len(candidate_texts) / elapsed:.0f} candidates/sec)"
    )
    logger.info(f"Candidate embeddings shape: {candidate_embeddings.shape}")

    # ── Save to .npz ──
    logger.info(f"Saving to {output_path}...")
    np.savez_compressed(
        output_path,
        candidate_embeddings=candidate_embeddings,
        jd_embedding=jd_embedding,
        candidate_ids=np.array(candidate_ids),
    )

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"Saved embeddings: {file_size_mb:.1f} MB")
    logger.info("Pre-computation complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute sentence-transformer embeddings for candidates."
    )
    parser.add_argument(
        "--candidates",
        required=True,
        help="Path to candidates.jsonl or candidates.json",
    )
    parser.add_argument(
        "--output",
        default="embeddings.npz",
        help="Output path for the .npz file (default: embeddings.npz)",
    )
    parser.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model name (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for encoding (default: 256)",
    )

    args = parser.parse_args()

    candidates = load_candidates(args.candidates)
    compute_embeddings(
        candidates,
        model_name=args.model,
        batch_size=args.batch_size,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
