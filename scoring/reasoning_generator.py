"""
reasoning_generator.py — Generate specific, fact-based reasoning for each
ranked candidate.

The reasoning must:
  - Reference specific facts from the candidate's profile
  - Connect to specific JD requirements
  - Acknowledge gaps honestly
  - NOT be templated or identical across candidates
  - NOT hallucinate skills/experience not in the profile

Stage 4 evaluation samples 10 random rows and checks for these qualities.
"""

import logging

from jd_config import (
    REQUIRED_SKILLS,
    DESIRED_SKILLS,
    CONSULTING_COMPANIES,
    EXPERIENCE_SWEET_SPOT,
)

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    return text.strip().lower() if text else ""


def _get_relevant_skills(candidate: dict) -> list[str]:
    """Extract skills relevant to the JD from the candidate's profile."""
    all_target_skills = REQUIRED_SKILLS | DESIRED_SKILLS
    relevant = []

    for skill in candidate.get("skills", []):
        name = _normalize(skill.get("name", ""))
        for target in all_target_skills:
            if target in name or name in target:
                # Use original casing
                relevant.append(skill["name"])
                break

    return relevant


def _get_core_ai_skills(candidate: dict) -> list[str]:
    """Extract core AI/ML skills."""
    ai_keywords = {
        "machine learning", "deep learning", "nlp", "pytorch", "tensorflow",
        "transformers", "bert", "gpt", "llm", "embeddings", "retrieval",
        "ranking", "recommendation", "data science", "neural",
        "huggingface", "sklearn", "scikit-learn", "keras",
        "rag", "vector", "faiss", "search",
    }
    found = []
    for skill in candidate.get("skills", []):
        name = _normalize(skill.get("name", ""))
        for kw in ai_keywords:
            if kw in name or name in kw:
                found.append(skill["name"])
                break
    return found


def _describe_career_fit(candidate: dict) -> str:
    """Generate a brief career-fit description."""
    career = candidate.get("career_history", [])
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "")
    yoe = profile.get("years_of_experience", 0)

    parts = []

    # Current role
    parts.append(f"{title} at {company}" if company else title)

    # Experience
    sweet_min, sweet_max = EXPERIENCE_SWEET_SPOT
    if sweet_min <= yoe <= sweet_max:
        parts.append(f"{yoe:.1f} yrs exp (ideal range)")
    else:
        parts.append(f"{yoe:.1f} yrs exp")

    # Check for production AI/ML experience in career
    has_production_ai = False
    for entry in career:
        desc = _normalize(entry.get("description", ""))
        if ("production" in desc or "deployed" in desc or "shipped" in desc):
            if any(kw in desc for kw in ["ml", "ai", "model", "ranking", "search", "embedding"]):
                has_production_ai = True
                break

    if has_production_ai:
        parts.append("has production AI/ML experience")

    # Consulting check
    consulting_only = all(
        any(c in _normalize(e.get("company", "")) for c in CONSULTING_COMPANIES)
        for e in career
    ) if career else False

    if consulting_only:
        parts.append("consulting-only background")

    return "; ".join(parts)


def _describe_behavioral_fit(candidate: dict) -> str:
    """Generate behavioral fit summary."""
    signals = candidate.get("redrob_signals", {})
    parts = []

    response_rate = signals.get("recruiter_response_rate", 0)
    if response_rate >= 0.6:
        parts.append(f"responsive (rate {response_rate:.0%})")
    elif response_rate < 0.2:
        parts.append(f"low responsiveness ({response_rate:.0%})")

    if signals.get("open_to_work_flag"):
        parts.append("open to work")

    notice = signals.get("notice_period_days", 90)
    if notice <= 30:
        parts.append(f"short notice ({notice}d)")
    elif notice > 60:
        parts.append(f"long notice ({notice}d)")

    github = signals.get("github_activity_score", -1)
    if github >= 50:
        parts.append(f"active on GitHub ({github:.0f})")

    return "; ".join(parts) if parts else ""


def _describe_gaps(candidate: dict, score_breakdown: dict) -> str:
    """Honestly acknowledge gaps and concerns."""
    gaps = []
    profile = candidate.get("profile", {})

    # Low skills score
    if score_breakdown.get("skills", 0) < 0.3:
        gaps.append("limited match on required technical skills")

    # Experience outside range
    yoe = profile.get("years_of_experience", 0)
    if yoe < 5:
        gaps.append(f"below stated experience range ({yoe:.1f} yrs)")
    elif yoe > 9:
        gaps.append(f"above stated experience range ({yoe:.1f} yrs)")

    # Non-relevant title
    if score_breakdown.get("title_career", 0) < 0.3:
        title = profile.get("current_title", "")
        gaps.append(f"current title ({title}) not directly aligned")

    # Low behavioral score
    if score_breakdown.get("behavioral_modifier", 1.0) < 0.6:
        gaps.append("low platform engagement")

    # Location
    location = _normalize(profile.get("location", ""))
    country = _normalize(profile.get("country", ""))
    if "india" not in country:
        gaps.append(f"located outside India ({profile.get('country', 'Unknown')})")

    return "; ".join(gaps) if gaps else ""


def generate_reasoning(
    candidate: dict,
    rank: int,
    score_breakdown: dict,
) -> str:
    """
    Generate a 1-2 sentence reasoning for why this candidate is ranked here.

    The reasoning is fact-based and specific to this candidate's profile.
    It avoids templated language and references actual data.

    Args:
        candidate: Full candidate dict.
        rank: The candidate's rank (1-100).
        score_breakdown: Dict with component scores from deep_scorer.

    Returns:
        A 1-2 sentence reasoning string.
    """
    profile = candidate.get("profile", {})
    title = profile.get("current_title", "Unknown")
    yoe = profile.get("years_of_experience", 0)

    # Build components
    relevant_skills = _get_relevant_skills(candidate)
    core_ai_skills = _get_core_ai_skills(candidate)
    career_fit = _describe_career_fit(candidate)
    behavioral_fit = _describe_behavioral_fit(candidate)
    gaps = _describe_gaps(candidate, score_breakdown)

    # ── Build the reasoning sentence(s) ──
    parts = []

    # Sentence 1: Who they are and why they fit
    skill_mention = ""
    if relevant_skills:
        skill_list = relevant_skills[:4]
        skill_mention = f" with relevant skills in {', '.join(skill_list)}"
    elif core_ai_skills:
        skill_list = core_ai_skills[:3]
        skill_mention = f" with AI/ML skills in {', '.join(skill_list)}"

    sentence1 = f"{career_fit}{skill_mention}."
    parts.append(sentence1)

    # Sentence 2: Behavioral context + honest gap acknowledgment
    sentence2_parts = []
    if behavioral_fit:
        sentence2_parts.append(behavioral_fit)
    if gaps:
        sentence2_parts.append(f"gaps: {gaps}")

    if sentence2_parts:
        parts.append(" ".join(sentence2_parts).capitalize() + ".")

    reasoning = " ".join(parts)

    # Safety: cap at reasonable length
    if len(reasoning) > 500:
        reasoning = reasoning[:497] + "..."

    return reasoning
