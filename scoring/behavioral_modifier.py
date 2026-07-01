"""
behavioral_modifier.py — Platform behavioral signal analysis.

Computes a multiplicative modifier (0.4 to 1.25) based on Redrob platform
activity signals. A perfect-on-paper candidate who hasn't logged in for
6 months and has a 5% response rate is NOT actually available.

Signal categories:
  - Availability & responsiveness (open_to_work, response_rate, last_active)
  - Engagement quality (interview completion, offer acceptance)
  - Verification & trust (verified email/phone, LinkedIn)
  - Market demand signals (profile views, search appearances, saved by recruiters)
  - Logistics fit (notice period, work mode, relocation, salary)
"""

import logging
from datetime import datetime, date

from jd_config import (
    IDEAL_NOTICE_PERIOD_DAYS,
    MAX_ACCEPTABLE_NOTICE_DAYS,
    PREFERRED_WORK_MODES,
    PREFERRED_LOCATIONS,
    PREFERRED_COUNTRIES,
)

logger = logging.getLogger(__name__)

# Reference date for "how recently active" calculations
REFERENCE_DATE = date(2026, 6, 1)


def _parse_date_safe(date_str) -> date:
    """Parse date string, return old date on failure."""
    if not date_str:
        return date(2020, 1, 1)
    try:
        return datetime.strptime(str(date_str), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return date(2020, 1, 1)


def compute_availability_score(signals: dict) -> float:
    """
    Score based on how available and responsive the candidate is.
    Range: 0.0 to 1.0
    """
    score = 0.5  # neutral baseline

    # Open to work flag — strong signal
    if signals.get("open_to_work_flag", False):
        score += 0.15

    # Recruiter response rate
    response_rate = signals.get("recruiter_response_rate", 0.5)
    if response_rate >= 0.7:
        score += 0.15
    elif response_rate >= 0.4:
        score += 0.08
    elif response_rate < 0.15:
        score -= 0.20

    # Average response time
    avg_response_hrs = signals.get("avg_response_time_hours", 48)
    if avg_response_hrs <= 12:
        score += 0.05
    elif avg_response_hrs <= 48:
        score += 0.02
    elif avg_response_hrs > 168:  # > 1 week
        score -= 0.10

    # Last active date
    last_active = _parse_date_safe(signals.get("last_active_date"))
    days_since_active = (REFERENCE_DATE - last_active).days
    if days_since_active <= 14:
        score += 0.10
    elif days_since_active <= 30:
        score += 0.05
    elif days_since_active <= 90:
        pass  # neutral
    elif days_since_active <= 180:
        score -= 0.10
    else:
        score -= 0.25  # inactive for 6+ months — heavy penalty

    return max(0.0, min(1.0, score))


def compute_engagement_score(signals: dict) -> float:
    """
    Score based on interview and hiring engagement quality.
    Range: 0.0 to 1.0
    """
    score = 0.5

    # Interview completion rate
    interview_rate = signals.get("interview_completion_rate", 0.5)
    if interview_rate >= 0.85:
        score += 0.15
    elif interview_rate >= 0.65:
        score += 0.08
    elif interview_rate < 0.3:
        score -= 0.15

    # Offer acceptance rate (-1 means no history)
    offer_rate = signals.get("offer_acceptance_rate", -1)
    if offer_rate >= 0:
        if offer_rate >= 0.7:
            score += 0.10
        elif offer_rate >= 0.4:
            score += 0.05
        elif offer_rate < 0.2:
            score -= 0.10

    # Applications submitted (shows active job seeking)
    apps = signals.get("applications_submitted_30d", 0)
    if 1 <= apps <= 10:
        score += 0.08  # actively looking but not spraying
    elif apps > 20:
        score -= 0.05  # spray-and-pray

    return max(0.0, min(1.0, score))


def compute_verification_score(signals: dict) -> float:
    """
    Score based on profile verification and trust signals.
    Range: 0.0 to 1.0
    """
    score = 0.5

    if signals.get("verified_email", False):
        score += 0.12
    if signals.get("verified_phone", False):
        score += 0.12
    if signals.get("linkedin_connected", False):
        score += 0.10

    # Profile completeness
    completeness = signals.get("profile_completeness_score", 50)
    if completeness >= 85:
        score += 0.10
    elif completeness >= 70:
        score += 0.05
    elif completeness < 40:
        score -= 0.10

    return max(0.0, min(1.0, score))


def compute_market_demand_score(signals: dict) -> float:
    """
    Score based on how much external demand exists for this candidate.
    Range: 0.0 to 1.0
    """
    score = 0.5

    # Saved by recruiters in last 30d
    saved = signals.get("saved_by_recruiters_30d", 0)
    if saved >= 10:
        score += 0.15
    elif saved >= 5:
        score += 0.10
    elif saved >= 2:
        score += 0.05

    # Search appearances
    appearances = signals.get("search_appearance_30d", 0)
    if appearances >= 100:
        score += 0.10
    elif appearances >= 30:
        score += 0.05

    # Profile views
    views = signals.get("profile_views_received_30d", 0)
    if views >= 20:
        score += 0.08
    elif views >= 10:
        score += 0.04

    return max(0.0, min(1.0, score))


def compute_logistics_score(signals: dict, candidate: dict) -> float:
    """
    Score based on logistical fit: notice period, work mode, location,
    relocation willingness.
    Range: 0.0 to 1.0
    """
    score = 0.5
    profile = candidate.get("profile", {})

    # Notice period
    notice_days = signals.get("notice_period_days", 60)
    if notice_days <= IDEAL_NOTICE_PERIOD_DAYS:
        score += 0.15
    elif notice_days <= 60:
        score += 0.05
    elif notice_days > MAX_ACCEPTABLE_NOTICE_DAYS:
        score -= 0.10

    # Work mode preference
    work_mode = signals.get("preferred_work_mode", "flexible")
    if work_mode in PREFERRED_WORK_MODES:
        score += 0.08
    elif work_mode == "remote":
        score -= 0.03  # slight negative, not dealbreaker

    # Location
    location = (profile.get("location", "") or "").lower()
    country = (profile.get("country", "") or "").lower()

    in_preferred_location = any(
        loc in location for loc in PREFERRED_LOCATIONS
    )
    in_preferred_country = any(
        c in country for c in PREFERRED_COUNTRIES
    )

    if in_preferred_location:
        score += 0.12
    elif in_preferred_country:
        score += 0.05
        # Check relocation willingness for non-preferred locations
        if signals.get("willing_to_relocate", False):
            score += 0.05
    else:
        # Outside India
        if signals.get("willing_to_relocate", False):
            score += 0.02
        else:
            score -= 0.08

    return max(0.0, min(1.0, score))


def compute_github_score(signals: dict) -> float:
    """
    GitHub activity score for a technical AI role.
    Range: 0.0 to 1.0
    """
    github = signals.get("github_activity_score", -1)
    if github < 0:
        return 0.4  # no GitHub linked — slight negative for AI role
    if github >= 70:
        return 0.95
    if github >= 50:
        return 0.80
    if github >= 30:
        return 0.65
    if github >= 10:
        return 0.50
    return 0.40


def compute_behavioral_modifier(candidate: dict) -> float:
    """
    Compute the final behavioral modifier for a candidate.

    This is a multiplicative factor applied to the candidate's skill/career
    score. Range: 0.4 (heavily penalized) to 1.25 (boosted).

    Args:
        candidate: Full candidate dict.

    Returns:
        Multiplicative modifier.
    """
    signals = candidate.get("redrob_signals", {})

    if not signals:
        return 0.8  # no signals → slight penalty

    # Compute sub-scores
    availability = compute_availability_score(signals)
    engagement = compute_engagement_score(signals)
    verification = compute_verification_score(signals)
    market_demand = compute_market_demand_score(signals)
    logistics = compute_logistics_score(signals, candidate)
    github = compute_github_score(signals)

    # Weighted combination → produces [0, 1] range
    combined = (
        availability * 0.30
        + engagement * 0.15
        + verification * 0.10
        + market_demand * 0.10
        + logistics * 0.20
        + github * 0.15
    )

    # Map [0, 1] → [0.4, 1.25] modifier range
    modifier = 0.4 + combined * 0.85

    return round(modifier, 4)


def get_behavioral_breakdown(candidate: dict) -> dict:
    """
    Return a detailed breakdown of behavioral scores for debugging
    and reasoning generation.
    """
    signals = candidate.get("redrob_signals", {})
    if not signals:
        return {"modifier": 0.8, "note": "No behavioral signals available"}

    return {
        "availability": round(compute_availability_score(signals), 3),
        "engagement": round(compute_engagement_score(signals), 3),
        "verification": round(compute_verification_score(signals), 3),
        "market_demand": round(compute_market_demand_score(signals), 3),
        "logistics": round(compute_logistics_score(signals, candidate), 3),
        "github": round(compute_github_score(signals), 3),
        "modifier": compute_behavioral_modifier(candidate),
    }
