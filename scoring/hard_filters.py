"""
hard_filters.py — First-pass filters that quickly eliminate obviously
irrelevant candidates before expensive scoring.

Based on explicit JD disqualifiers:
  - Non-tech roles with zero ML/AI career history
  - Experience far outside the 3–15 year window
  - Pure consulting-only careers (with no product company stint)
  - Primary expertise in irrelevant domains (robotics, speech, CV-only)
"""

import re
import logging
from typing import Optional

from jd_config import (
    EXPERIENCE_HARD_MIN,
    EXPERIENCE_HARD_MAX,
    TIER_1_TITLES,
    TIER_2_TITLES,
    NON_RELEVANT_TITLES,
    CONSULTING_COMPANIES,
    CORE_AI_ML_SKILLS,
)

logger = logging.getLogger(__name__)

# AI/ML keywords to search for in career descriptions
AI_ML_KEYWORDS = re.compile(
    r"\b("
    r"machine learning|deep learning|ml|ai|artificial intelligence|"
    r"nlp|natural language|embedding|retrieval|ranking|search|"
    r"recommendation|neural|transformer|bert|gpt|llm|"
    r"data science|data scientist|classification|regression|"
    r"model training|model serving|model deployment|inference|"
    r"feature engineering|vector|similarity|information retrieval|"
    r"pytorch|tensorflow|scikit|sklearn|hugging\s*face|"
    r"rag|fine.?tun|prompt|langchain|"
    r"data pipeline|data engineering|spark|airflow|"
    r"analytics|data.?driven|algorithm"
    r")\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    """Lowercase and strip whitespace."""
    return text.strip().lower() if text else ""


def _has_ai_ml_career_signals(candidate: dict) -> bool:
    """
    Check if the candidate has any AI/ML signals in their career
    history, skills, or profile.
    """
    # Check career descriptions
    for entry in candidate.get("career_history", []):
        desc = entry.get("description", "")
        title = entry.get("title", "")
        if AI_ML_KEYWORDS.search(desc) or AI_ML_KEYWORDS.search(title):
            return True

    # Check skills
    candidate_skills = {
        _normalize(s.get("name", "")) for s in candidate.get("skills", [])
    }
    if candidate_skills & CORE_AI_ML_SKILLS:
        return True

    # Check profile summary and headline
    summary = candidate.get("profile", {}).get("summary", "")
    headline = candidate.get("profile", {}).get("headline", "")
    if AI_ML_KEYWORDS.search(summary) or AI_ML_KEYWORDS.search(headline):
        return True

    return False


def _get_title_tier(title: str) -> int:
    """
    Classify a title into relevance tiers.
    Returns: 1 (best), 2 (adjacent), 3 (irrelevant), 0 (unknown).
    """
    t = _normalize(title)
    if not t:
        return 0

    for tier1 in TIER_1_TITLES:
        if tier1 in t or t in tier1:
            return 1

    for tier2 in TIER_2_TITLES:
        if tier2 in t or t in tier2:
            return 2

    for non_rel in NON_RELEVANT_TITLES:
        if non_rel in t or t in non_rel:
            return 3

    return 0  # unknown — don't filter


def _is_consulting_only(candidate: dict) -> bool:
    """
    Check if the candidate has ONLY worked at consulting/services firms.
    If they have at least one non-consulting stint, they pass.
    """
    career = candidate.get("career_history", [])
    if not career:
        return False

    for entry in career:
        company = _normalize(entry.get("company", ""))
        is_consulting = any(
            c_name in company for c_name in CONSULTING_COMPANIES
        )
        if not is_consulting:
            return False  # Found at least one non-consulting company

    return True  # All companies are consulting


def passes_hard_filter(candidate: dict) -> tuple[bool, Optional[str]]:
    """
    Apply hard filters to quickly eliminate obviously irrelevant candidates.

    Returns:
        (passes, reason): passes=True if candidate should advance.
                          reason explains why they were filtered if not.
    """
    profile = candidate.get("profile", {})
    cid = candidate.get("candidate_id", "UNKNOWN")

    # ── Experience range check ──
    yoe = profile.get("years_of_experience", 0)
    if yoe < EXPERIENCE_HARD_MIN:
        return False, f"Too junior ({yoe:.1f} yrs, need ≥{EXPERIENCE_HARD_MIN})"
    if yoe > EXPERIENCE_HARD_MAX:
        return False, f"Too senior ({yoe:.1f} yrs, need ≤{EXPERIENCE_HARD_MAX})"

    # ── Title relevance ──
    current_title = profile.get("current_title", "")
    title_tier = _get_title_tier(current_title)

    # If current title is clearly non-relevant (tier 3), check career for
    # any AI/ML signals before filtering
    if title_tier == 3:
        if not _has_ai_ml_career_signals(candidate):
            return False, f"Non-tech title '{current_title}' with no AI/ML career signals"
        # Has AI/ML signals despite non-relevant title — let through with caution

    # ── Check for ANY AI/ML relevance ──
    # Even tier-0 (unknown) titles pass if they have AI/ML signals
    if title_tier not in (1, 2):
        if not _has_ai_ml_career_signals(candidate):
            return False, f"No AI/ML relevance found in profile"

    return True, None


def compute_filter_metadata(candidate: dict) -> dict:
    """
    Compute and return metadata from the filtering stage that's useful
    for downstream scoring (avoids re-computation).
    """
    profile = candidate.get("profile", {})
    current_title = _normalize(profile.get("current_title", ""))

    return {
        "title_tier": _get_title_tier(current_title),
        "is_consulting_only": _is_consulting_only(candidate),
        "has_ai_ml_signals": _has_ai_ml_career_signals(candidate),
    }
