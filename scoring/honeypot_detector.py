"""
honeypot_detector.py — Detect honeypot candidates with subtly impossible profiles.

The dataset contains ~80 honeypots with fabricated profiles. Examples of
impossible signals:
  - Claiming 8 years at a company founded 3 years ago
  - "Expert" in 10+ skills with 0 endorsements and <6 months duration each
  - Career history duration that wildly exceeds years_of_experience
  - Skill assessment scores that contradict claimed proficiency

Honeypots are forced to relevance tier 0 in the ground truth. If >10% of
our top 100 are honeypots, we are disqualified.
"""

import logging
from datetime import datetime, date
from typing import Optional

logger = logging.getLogger(__name__)

# Proficiency weights for scoring
PROFICIENCY_WEIGHT = {
    "beginner": 1,
    "intermediate": 2,
    "advanced": 3,
    "expert": 4,
}


def _parse_date(date_str: Optional[str]) -> Optional[date]:
    """Safely parse a date string in YYYY-MM-DD format."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _check_experience_inflation(candidate: dict) -> float:
    """
    Check if claimed years_of_experience is wildly inconsistent with
    actual career_history duration.

    Returns a suspicion score 0.0 (clean) to 1.0 (definitely honeypot).
    """
    try:
        claimed_years = candidate["profile"].get("years_of_experience", 0)
        career = candidate.get("career_history", [])
        if not career:
            return 0.0

        # Calculate total career months from history
        total_months = sum(
            entry.get("duration_months", 0) for entry in career
        )
        actual_years = total_months / 12.0

        # Flag if claimed experience is more than double what career shows
        # OR if career shows more than double what's claimed (also suspicious)
        if claimed_years > 0 and actual_years > 0:
            ratio = claimed_years / actual_years
            if ratio > 2.5 or ratio < 0.3:
                return 0.8
            if ratio > 2.0 or ratio < 0.4:
                return 0.5
        return 0.0
    except (KeyError, TypeError, ZeroDivisionError):
        return 0.0


def _check_keyword_stuffer(candidate: dict) -> float:
    """
    Detect keyword stuffers: candidates who claim expert proficiency in
    many skills but have no endorsements and very short durations.

    Returns a suspicion score 0.0 (clean) to 1.0 (honeypot).
    """
    try:
        skills = candidate.get("skills", [])
        if len(skills) < 5:
            return 0.0

        expert_skills = [
            s for s in skills
            if s.get("proficiency") == "expert"
        ]

        if len(expert_skills) < 4:
            return 0.0

        # Count experts with zero endorsements and very short duration
        suspicious_experts = 0
        for skill in expert_skills:
            endorsements = skill.get("endorsements", 0)
            duration = skill.get("duration_months", 0)
            if endorsements == 0 and duration < 6:
                suspicious_experts += 1

        if suspicious_experts >= 8:
            return 1.0
        if suspicious_experts >= 5:
            return 0.7
        if suspicious_experts >= 3:
            return 0.4
        return 0.0
    except (KeyError, TypeError):
        return 0.0


def _check_assessment_contradiction(candidate: dict) -> float:
    """
    Check if skill assessment scores contradict claimed proficiency levels.
    E.g., "expert" in Python but assessment score of 15/100.

    Returns a suspicion score 0.0 (clean) to 1.0 (honeypot).
    """
    try:
        skills = candidate.get("skills", [])
        signals = candidate.get("redrob_signals", {})
        assessments = signals.get("skill_assessment_scores", {})

        if not assessments or not skills:
            return 0.0

        contradictions = 0
        checked = 0

        for skill in skills:
            name = skill.get("name", "")
            proficiency = skill.get("proficiency", "")
            if name in assessments:
                checked += 1
                score = assessments[name]
                # Expert claiming but scoring < 30 is suspicious
                if proficiency == "expert" and score < 25:
                    contradictions += 1
                elif proficiency == "advanced" and score < 15:
                    contradictions += 1

        if checked == 0:
            return 0.0

        ratio = contradictions / checked
        if ratio > 0.6:
            return 0.9
        if ratio > 0.4:
            return 0.6
        if ratio > 0.2:
            return 0.3
        return 0.0
    except (KeyError, TypeError, ZeroDivisionError):
        return 0.0


def _check_career_timeline_impossible(candidate: dict) -> float:
    """
    Check for impossible career timelines:
    - Overlapping full-time roles that span years
    - Start dates in the far future
    - Duration_months doesn't match start/end dates at all
    """
    try:
        career = candidate.get("career_history", [])
        if len(career) < 2:
            return 0.0

        suspicion = 0.0

        for entry in career:
            start = _parse_date(entry.get("start_date"))
            end = _parse_date(entry.get("end_date"))
            claimed_duration = entry.get("duration_months", 0)

            if start and end:
                actual_months = (end.year - start.year) * 12 + (end.month - start.month)
                # If claimed duration is wildly off from date range
                if actual_months > 0 and claimed_duration > 0:
                    ratio = claimed_duration / actual_months
                    if ratio > 3.0 or ratio < 0.2:
                        suspicion = max(suspicion, 0.7)

            # Check if start_date is far in the future
            if start and start > date(2026, 12, 31):
                suspicion = max(suspicion, 0.9)

        return suspicion
    except (KeyError, TypeError, ZeroDivisionError):
        return 0.0


def _check_profile_completeness_mismatch(candidate: dict) -> float:
    """
    Check if profile_completeness_score is very high but the profile
    is actually sparse, or vice versa.
    """
    try:
        signals = candidate.get("redrob_signals", {})
        completeness = signals.get("profile_completeness_score", 50)
        profile = candidate.get("profile", {})
        skills = candidate.get("skills", [])
        career = candidate.get("career_history", [])
        education = candidate.get("education", [])

        # Count actual completeness indicators
        actual_filled = 0
        total_fields = 0

        for field in ["headline", "summary", "location", "country",
                      "current_title", "current_company"]:
            total_fields += 1
            if profile.get(field):
                actual_filled += 1

        if skills:
            actual_filled += 1
        total_fields += 1

        if career:
            actual_filled += 1
        total_fields += 1

        if education:
            actual_filled += 1
        total_fields += 1

        actual_pct = (actual_filled / total_fields) * 100 if total_fields > 0 else 50

        # If claimed completeness is 95+ but profile is actually sparse
        if completeness > 95 and actual_pct < 50:
            return 0.5
        return 0.0
    except (KeyError, TypeError, ZeroDivisionError):
        return 0.0


def compute_honeypot_score(candidate: dict) -> float:
    """
    Compute an aggregate honeypot suspicion score for a candidate.

    Returns:
        float: 0.0 (definitely clean) to 1.0 (definitely honeypot).
               Candidates with score >= 0.6 should be excluded.
    """
    scores = [
        _check_experience_inflation(candidate) * 0.25,
        _check_keyword_stuffer(candidate) * 0.30,
        _check_assessment_contradiction(candidate) * 0.20,
        _check_career_timeline_impossible(candidate) * 0.15,
        _check_profile_completeness_mismatch(candidate) * 0.10,
    ]
    return sum(scores)


def is_honeypot(candidate: dict, threshold: float = 0.45) -> bool:
    """
    Determine if a candidate is a honeypot.

    Args:
        candidate: Full candidate dict from the JSONL.
        threshold: Score threshold above which a candidate is flagged.

    Returns:
        True if the candidate is likely a honeypot.
    """
    score = compute_honeypot_score(candidate)
    if score >= threshold:
        cid = candidate.get("candidate_id", "UNKNOWN")
        logger.info(f"Honeypot detected: {cid} (score={score:.3f})")
        return True
    return False
