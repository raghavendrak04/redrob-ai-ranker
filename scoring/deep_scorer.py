"""
deep_scorer.py — Multi-dimensional weighted scoring engine.

Takes a candidate who passed hard filters + semantic pre-screening and
computes a detailed fit score against the JD across six dimensions:

    semantic     (0.25) — embedding cosine similarity
    title_career (0.25) — title relevance + career trajectory quality
    skills       (0.20) — skill match with trust multiplier
    experience   (0.10) — years-of-experience fit
    education    (0.05) — tier-weighted education relevance
    behavioral   (0.15) — platform signal modifier

The final score is a weighted sum of these components.
"""

import re
import logging
from typing import Optional

from jd_config import (
    SCORING_WEIGHTS,
    EXPERIENCE_RANGE,
    EXPERIENCE_SWEET_SPOT,
    TIER_1_TITLES,
    TIER_2_TITLES,
    NON_RELEVANT_TITLES,
    REQUIRED_SKILLS,
    DESIRED_SKILLS,
    CORE_AI_ML_SKILLS,
    CONSULTING_COMPANIES,
    NON_RELEVANT_PRIMARY_SKILLS,
)
from scoring.behavioral_modifier import compute_behavioral_modifier

logger = logging.getLogger(__name__)

# ── Proficiency weights ──
PROFICIENCY_WEIGHT = {
    "beginner": 0.25,
    "intermediate": 0.50,
    "advanced": 0.80,
    "expert": 1.00,
}

# AI/ML keywords for career description analysis
AI_CAREER_PATTERN = re.compile(
    r"\b("
    r"machine learning|deep learning|ml|neural|embedding|retrieval|"
    r"ranking|search|recommendation|nlp|natural language|transformer|"
    r"bert|gpt|llm|fine.?tun|model training|model serving|inference|"
    r"pytorch|tensorflow|data science|feature engineering|vector|"
    r"information retrieval|rag|similarity|classification|"
    r"production.{0,20}(?:model|ml|ai|deploy)|"
    r"a/b test|evaluation|metric|ndcg|precision|recall"
    r")\b",
    re.IGNORECASE,
)

PRODUCTION_PATTERN = re.compile(
    r"\b("
    r"production|deployed|shipped|live|real.?user|scale|"
    r"serving|pipeline|infrastructure|system|platform|"
    r"million|thousands|traffic|latency|availability"
    r")\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    return text.strip().lower() if text else ""


def _get_title_score(title: str) -> float:
    """Score a title's relevance to the JD. Range: 0.0 to 1.0."""
    t = _normalize(title)
    if not t:
        return 0.1

    # Tier 1: Direct match
    for tier1 in TIER_1_TITLES:
        if tier1 in t or t in tier1:
            return 1.0

    # Tier 2: Adjacent
    for tier2 in TIER_2_TITLES:
        if tier2 in t or t in tier2:
            return 0.55

    # Non-relevant
    for nr in NON_RELEVANT_TITLES:
        if nr in t or t in nr:
            return 0.05

    return 0.25  # unknown title


def compute_title_career_score(candidate: dict) -> float:
    """
    Score the candidate's title and career trajectory.
    Range: 0.0 to 1.0.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    score = 0.0

    # ── Current title relevance (40% of this component) ──
    current_title = profile.get("current_title", "")
    title_score = _get_title_score(current_title)
    score += title_score * 0.40

    # ── Career trajectory analysis (40% of this component) ──
    trajectory_score = 0.0
    ai_career_entries = 0
    production_entries = 0
    product_company_entries = 0
    total_entries = len(career)

    for entry in career:
        desc = entry.get("description", "")
        entry_title = entry.get("title", "")
        company = _normalize(entry.get("company", ""))

        # Check for AI/ML work in career descriptions
        if AI_CAREER_PATTERN.search(desc) or AI_CAREER_PATTERN.search(entry_title):
            ai_career_entries += 1

        # Check for production experience signals
        if PRODUCTION_PATTERN.search(desc):
            production_entries += 1

        # Check for product company experience (non-consulting)
        is_consulting = any(c in company for c in CONSULTING_COMPANIES)
        if not is_consulting and company not in ("", "acme corp", "dunder mifflin"):
            product_company_entries += 1

    if total_entries > 0:
        ai_ratio = ai_career_entries / total_entries
        trajectory_score += ai_ratio * 0.50

        prod_ratio = production_entries / total_entries
        trajectory_score += prod_ratio * 0.30

        # Product company experience bonus
        if product_company_entries > 0:
            trajectory_score += min(0.20, product_company_entries * 0.07)

    score += trajectory_score * 0.40

    # ── Consulting-only penalty (20% of this component) ──
    all_consulting = all(
        any(c in _normalize(entry.get("company", "")) for c in CONSULTING_COMPANIES)
        for entry in career
    ) if career else False

    if all_consulting:
        career_diversity_score = 0.10  # heavy penalty
    elif any(
        any(c in _normalize(entry.get("company", "")) for c in CONSULTING_COMPANIES)
        for entry in career
    ):
        career_diversity_score = 0.50  # mixed — acceptable
    else:
        career_diversity_score = 0.80  # all product companies

    score += career_diversity_score * 0.20

    return round(min(1.0, score), 4)


def compute_skills_score(candidate: dict) -> float:
    """
    Score skill match with endorsement/duration trust multiplier.
    Range: 0.0 to 1.0.

    Checks:
      - Required skills presence (weighted by trust)
      - Desired skills presence
      - Core AI/ML skill breadth
      - Penalizes suspicious patterns (keyword stuffing)
    """
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    assessments = signals.get("skill_assessment_scores", {})

    if not skills:
        return 0.0

    # Build skill lookup with trust scores
    candidate_skills = {}
    for s in skills:
        name = _normalize(s.get("name", ""))
        if not name:
            continue

        proficiency = PROFICIENCY_WEIGHT.get(s.get("proficiency", ""), 0.25)
        endorsements = s.get("endorsements", 0)
        duration_months = s.get("duration_months", 0)

        # Trust multiplier: cross-validates claimed proficiency
        trust = min(1.0, (
            endorsements * 0.03
            + duration_months * 0.015
            + proficiency * 0.30
        ))

        # If assessment score exists, factor it in
        if name in assessments:
            assessment = assessments[name] / 100.0
            trust = trust * 0.6 + assessment * 0.4

        candidate_skills[name] = {
            "proficiency": proficiency,
            "trust": trust,
            "effective": proficiency * trust,
        }

    # ── Required skills match (50% of skills score) ──
    required_matches = 0
    required_total_quality = 0.0
    for req_skill in REQUIRED_SKILLS:
        for cand_skill, data in candidate_skills.items():
            if req_skill in cand_skill or cand_skill in req_skill:
                required_matches += 1
                required_total_quality += data["effective"]
                break

    # Normalize: max reasonable required matches is ~8-10
    required_score = min(1.0, required_total_quality / 4.0)

    # ── Desired skills match (20% of skills score) ──
    desired_matches = 0
    for des_skill in DESIRED_SKILLS:
        for cand_skill in candidate_skills:
            if des_skill in cand_skill or cand_skill in des_skill:
                desired_matches += 1
                break
    desired_score = min(1.0, desired_matches / 3.0)

    # ── Core AI/ML breadth (20% of skills score) ──
    core_matches = 0
    for core_skill in CORE_AI_ML_SKILLS:
        for cand_skill in candidate_skills:
            if core_skill in cand_skill or cand_skill in core_skill:
                core_matches += 1
                break
    core_score = min(1.0, core_matches / 5.0)

    # ── Penalty for non-relevant primary skills (10% of skills score) ──
    non_relevant_count = 0
    for nr_skill in NON_RELEVANT_PRIMARY_SKILLS:
        for cand_skill, data in candidate_skills.items():
            if nr_skill in cand_skill or cand_skill in nr_skill:
                if data["proficiency"] >= 0.80:  # advanced or expert
                    non_relevant_count += 1
                break

    # If primary expertise is in non-relevant domains
    relevance_penalty = max(0.0, 1.0 - non_relevant_count * 0.25)

    final = (
        required_score * 0.50
        + desired_score * 0.20
        + core_score * 0.20
        + relevance_penalty * 0.10
    )

    return round(min(1.0, final), 4)


def compute_experience_score(candidate: dict) -> float:
    """
    Score experience-years fit to JD. Range: 0.0 to 1.0.
    Sweet spot: 6-8 years. Acceptable: 5-9. Allowed: 3-15.
    """
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)

    sweet_min, sweet_max = EXPERIENCE_SWEET_SPOT
    range_min, range_max = EXPERIENCE_RANGE

    if sweet_min <= yoe <= sweet_max:
        return 1.0
    elif range_min <= yoe <= range_max:
        return 0.85
    elif 3 <= yoe < range_min:
        # Scale from 0.4 at 3 yrs to 0.85 at 5 yrs
        return 0.4 + (yoe - 3) / (range_min - 3) * 0.45
    elif range_max < yoe <= 12:
        # Scale from 0.85 at 9 yrs to 0.55 at 12 yrs
        return 0.85 - (yoe - range_max) / (12 - range_max) * 0.30
    elif 12 < yoe <= 15:
        return 0.45
    else:
        return 0.2


def compute_education_score(candidate: dict) -> float:
    """
    Score education relevance. Range: 0.0 to 1.0.
    Considers institution tier, degree type, and field of study.
    """
    education = candidate.get("education", [])
    if not education:
        return 0.3  # no education listed — slight penalty

    best_score = 0.0

    for edu in education:
        score = 0.0
        tier = edu.get("tier", "unknown")
        degree = _normalize(edu.get("degree", ""))
        field = _normalize(edu.get("field_of_study", ""))

        # Tier scoring
        tier_scores = {
            "tier_1": 0.40,
            "tier_2": 0.30,
            "tier_3": 0.20,
            "tier_4": 0.10,
            "unknown": 0.15,
        }
        score += tier_scores.get(tier, 0.10)

        # Degree type
        if any(d in degree for d in ["m.tech", "m.s", "ms", "m.sc", "mtech", "master"]):
            score += 0.25
        elif any(d in degree for d in ["b.tech", "b.e", "btech", "b.sc", "bachelor"]):
            score += 0.15
        elif "ph" in degree or "doctor" in degree:
            score += 0.30
        else:
            score += 0.05

        # Field relevance
        relevant_fields = [
            "computer science", "cs", "artificial intelligence", "ai",
            "machine learning", "data science", "information technology",
            "it", "electronics", "ece", "electrical",
            "mathematics", "statistics", "applied math",
        ]
        if any(f in field for f in relevant_fields):
            score += 0.20
        else:
            score += 0.05

        best_score = max(best_score, score)

    return round(min(1.0, best_score), 4)


def compute_deep_score(
    candidate: dict,
    semantic_similarity: float,
) -> dict:
    """
    Compute the full multi-dimensional score for a candidate.

    Args:
        candidate: Full candidate dict.
        semantic_similarity: Pre-computed cosine similarity [0, 1].

    Returns:
        Dict with component scores and final weighted score.
    """
    try:
        # Compute each component
        title_career = compute_title_career_score(candidate)
        skills = compute_skills_score(candidate)
        experience = compute_experience_score(candidate)
        education = compute_education_score(candidate)
        behavioral = compute_behavioral_modifier(candidate)

        # Weighted combination (behavioral is multiplicative, not additive)
        w = SCORING_WEIGHTS
        base_score = (
            semantic_similarity * w["semantic"]
            + title_career * w["title_career"]
            + skills * w["skills"]
            + experience * w["experience"]
            + education * w["education"]
        )

        # Behavioral acts as a modifier on the base score
        # Range is 0.4–1.25, so it can boost or penalize
        base_weight = 1.0 - w["behavioral"]
        final_score = base_score * base_weight + (base_score * behavioral) * w["behavioral"]

        # Alternative: direct multiplicative approach
        # final_score = base_score * behavioral
        # This is cleaner but can compress scores too much

        return {
            "final_score": round(min(1.0, max(0.0, final_score)), 6),
            "semantic": round(semantic_similarity, 4),
            "title_career": round(title_career, 4),
            "skills": round(skills, 4),
            "experience": round(experience, 4),
            "education": round(education, 4),
            "behavioral_modifier": round(behavioral, 4),
            "base_score": round(base_score, 4),
        }

    except Exception as e:
        cid = candidate.get("candidate_id", "UNKNOWN")
        logger.error(f"Error scoring {cid}: {e}", exc_info=True)
        return {
            "final_score": 0.0,
            "semantic": semantic_similarity,
            "title_career": 0.0,
            "skills": 0.0,
            "experience": 0.0,
            "education": 0.0,
            "behavioral_modifier": 0.8,
            "base_score": 0.0,
            "error": str(e),
        }
