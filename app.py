#!/usr/bin/env python3
"""
app.py — Streamlit UI for the Redrob Intelligent Candidate
Discovery & Ranking System.

A polished, interactive dashboard that serves as the sandbox/demo
for the hackathon submission.

Usage:
    streamlit run app.py

Features:
    - Job description viewer with extracted requirements
    - Upload or use bundled candidate data
    - Run the full ranking pipeline interactively
    - Visual score breakdowns and radar charts
    - Drill into individual candidate profiles
    - Export ranked results to CSV
"""

import json
import sys
import time
import logging
from pathlib import Path
from io import StringIO

import pandas as pd
import numpy as np
import streamlit as st

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

from jd_config import (
    JD_TEXT_FOR_EMBEDDING,
    JOB_TITLE,
    COMPANY,
    EXPERIENCE_RANGE,
    EXPERIENCE_SWEET_SPOT,
    PREFERRED_LOCATIONS,
    REQUIRED_SKILLS,
    DESIRED_SKILLS,
    SCORING_WEIGHTS,
)
from scoring.honeypot_detector import is_honeypot, compute_honeypot_score
from scoring.hard_filters import passes_hard_filter
from scoring.semantic_scorer import (
    SemanticScorer,
    TfidfFallbackScorer,
    build_candidate_text,
)
from scoring.deep_scorer import compute_deep_score
from scoring.behavioral_modifier import (
    get_behavioral_breakdown,
    compute_behavioral_modifier,
)
from scoring.reasoning_generator import generate_reasoning

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Page config & theme
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Redrob AI Ranker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium styling
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ── */
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.15), transparent 70%);
        border-radius: 50%;
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.03em;
    }
    .main-header p {
        color: #a5b4fc;
        font-size: 1.05rem;
        margin: 0.4rem 0 0 0;
        font-weight: 400;
    }

    /* ── Stat cards ── */
    .stat-card {
        background: linear-gradient(135deg, #1e1b4b, #312e81);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 14px;
        padding: 1.3rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(99, 102, 241, 0.2);
    }
    .stat-card .stat-value {
        font-size: 2rem;
        font-weight: 800;
        color: #a5b4fc;
        line-height: 1;
    }
    .stat-card .stat-label {
        font-size: 0.82rem;
        color: #94a3b8;
        margin-top: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 500;
    }

    /* ── Candidate card ── */
    .candidate-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(99, 102, 241, 0.18);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.12);
        transition: all 0.25s ease;
    }
    .candidate-card:hover {
        border-color: rgba(99, 102, 241, 0.45);
        box-shadow: 0 6px 24px rgba(99, 102, 241, 0.12);
    }
    .candidate-card h3 {
        color: #e2e8f0;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0 0 0.3rem 0;
    }
    .candidate-card .subtitle {
        color: #818cf8;
        font-size: 0.92rem;
        font-weight: 500;
    }
    .candidate-card .meta {
        color: #94a3b8;
        font-size: 0.82rem;
        margin-top: 0.5rem;
    }

    /* ── Score badge ── */
    .score-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 0.02em;
    }
    .score-high {
        background: linear-gradient(135deg, #065f46, #047857);
        color: #6ee7b7;
    }
    .score-mid {
        background: linear-gradient(135deg, #78350f, #92400e);
        color: #fbbf24;
    }
    .score-low {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        color: #fca5a5;
    }

    /* ── Rank badge ── */
    .rank-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
        border-radius: 50%;
        font-weight: 800;
        font-size: 1rem;
        margin-right: 0.8rem;
    }
    .rank-1 { background: linear-gradient(135deg, #fbbf24, #f59e0b); color: #1a1a2e; }
    .rank-2 { background: linear-gradient(135deg, #d1d5db, #9ca3af); color: #1a1a2e; }
    .rank-3 { background: linear-gradient(135deg, #cd7f32, #b87333); color: #1a1a2e; }
    .rank-other { background: rgba(99, 102, 241, 0.15); color: #a5b4fc; }

    /* ── Progress bar ── */
    .score-bar-outer {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        height: 10px;
        overflow: hidden;
        margin: 4px 0;
    }
    .score-bar-inner {
        height: 100%;
        border-radius: 8px;
        transition: width 0.6s ease;
    }

    /* ── Pill tag ── */
    .skill-pill {
        display: inline-block;
        padding: 0.2rem 0.65rem;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 500;
        margin: 0.15rem;
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #c7d2fe;
        background: rgba(99, 102, 241, 0.08);
    }
    .skill-pill.matched {
        border-color: rgba(52, 211, 153, 0.5);
        color: #6ee7b7;
        background: rgba(52, 211, 153, 0.1);
    }

    /* ── Section divider ── */
    .section-divider {
        border: none;
        border-top: 1px solid rgba(99, 102, 241, 0.15);
        margin: 1.5rem 0;
    }

    /* ── Pipeline stage ── */
    .pipeline-stage {
        background: rgba(99, 102, 241, 0.06);
        border-left: 3px solid #6366f1;
        padding: 0.8rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .pipeline-stage .stage-num {
        color: #818cf8;
        font-weight: 700;
    }
    .pipeline-stage .stage-result {
        color: #6ee7b7;
        font-weight: 600;
        float: right;
    }

    /* ── Hide default streamlit hamburger and footer ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ── Sidebar styling ── */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #1a1a2e);
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════

def load_candidates_from_file(uploaded_file=None, default_path=None):
    """Load candidates from uploaded file or default path."""
    if uploaded_file is not None:
        content = uploaded_file.getvalue().decode("utf-8")
        name = uploaded_file.name

        if name.endswith(".jsonl"):
            candidates = []
            for line in content.strip().split("\n"):
                if line.strip():
                    candidates.append(json.loads(line))
            return candidates
        elif name.endswith(".json"):
            return json.loads(content)
        else:
            st.error("Please upload a .json or .jsonl file.")
            return None

    if default_path and Path(default_path).exists():
        p = Path(default_path)
        if p.suffix == ".jsonl":
            candidates = []
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        candidates.append(json.loads(line))
            return candidates
        elif p.suffix == ".json":
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def score_badge_html(score: float) -> str:
    """Create a color-coded score badge."""
    if score >= 0.7:
        cls = "score-high"
    elif score >= 0.4:
        cls = "score-mid"
    else:
        cls = "score-low"
    return f'<span class="score-badge {cls}">{score:.4f}</span>'


def rank_badge_html(rank: int) -> str:
    """Create a rank badge."""
    if rank == 1:
        cls = "rank-1"
    elif rank == 2:
        cls = "rank-2"
    elif rank == 3:
        cls = "rank-3"
    else:
        cls = "rank-other"
    return f'<span class="rank-badge {cls}">{rank}</span>'


def score_bar_html(score: float, color: str = "#6366f1", label: str = "") -> str:
    """Create a visual score bar."""
    pct = min(100, max(0, score * 100))
    return f"""
    <div style="display: flex; align-items: center; gap: 8px; margin: 3px 0;">
        <span style="color: #94a3b8; font-size: 0.78rem; min-width: 90px;">{label}</span>
        <div class="score-bar-outer" style="flex: 1;">
            <div class="score-bar-inner" style="width: {pct}%; background: {color};"></div>
        </div>
        <span style="color: #e2e8f0; font-size: 0.82rem; font-weight: 600; min-width: 45px;">{score:.3f}</span>
    </div>
    """


def run_ranking_pipeline(candidates, embeddings_path="embeddings.npz"):
    """Run the full ranking pipeline with progress tracking."""
    pipeline_log = []
    total_start = time.time()

    # Stage 1: Honeypot Detection
    stage_start = time.time()
    honeypot_ids = set()
    honeypot_scores = {}

    progress_bar = st.progress(0, text="🔍 Stage 1: Detecting honeypots...")
    for i, c in enumerate(candidates):
        hp_score = compute_honeypot_score(c)
        if hp_score >= 0.45:
            honeypot_ids.add(c["candidate_id"])
            honeypot_scores[c["candidate_id"]] = hp_score
        if i % 500 == 0:
            progress_bar.progress(
                min(0.15, i / len(candidates) * 0.15),
                text=f"🔍 Stage 1: Scanning candidate {i+1}/{len(candidates)}..."
            )

    remaining = [c for c in candidates if c["candidate_id"] not in honeypot_ids]
    pipeline_log.append({
        "stage": "Honeypot Detection",
        "input": len(candidates),
        "output": len(remaining),
        "removed": len(honeypot_ids),
        "time": time.time() - stage_start,
    })

    # Stage 2: Hard Filters
    stage_start = time.time()
    progress_bar.progress(0.20, text="⚡ Stage 2: Applying hard filters...")

    filtered = []
    filter_stats = {"too_junior": 0, "too_senior": 0, "non_tech": 0, "no_ai": 0}

    for i, c in enumerate(remaining):
        passes, reason = passes_hard_filter(c)
        if passes:
            filtered.append(c)
        else:
            if reason and "Too junior" in reason:
                filter_stats["too_junior"] += 1
            elif reason and "Too senior" in reason:
                filter_stats["too_senior"] += 1
            elif reason and "Non-tech" in reason:
                filter_stats["non_tech"] += 1
            else:
                filter_stats["no_ai"] += 1

        if i % 500 == 0:
            progress_bar.progress(
                min(0.40, 0.20 + i / len(remaining) * 0.20),
                text=f"⚡ Stage 2: Filtering {i+1}/{len(remaining)}..."
            )

    pipeline_log.append({
        "stage": "Hard Filters",
        "input": len(remaining),
        "output": len(filtered),
        "removed": len(remaining) - len(filtered),
        "time": time.time() - stage_start,
        "stats": filter_stats,
    })

    # Stage 3: Semantic Similarity
    stage_start = time.time()
    progress_bar.progress(0.45, text="🧠 Stage 3: Computing semantic similarity...")

    semantic_scorer = SemanticScorer(embeddings_path)
    top_n = min(500, len(filtered))

    if semantic_scorer.is_fallback:
        tfidf = TfidfFallbackScorer(JD_TEXT_FOR_EMBEDDING)
        scores = tfidf.fit_and_score(filtered, JD_TEXT_FOR_EMBEDDING)
        scored_list = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        semantic_top_ids = {cid for cid, _ in scored_list}
        for cid, s in scores.items():
            semantic_scorer.similarity_cache[cid] = s * 2 - 1
    else:
        filtered_ids = [c["candidate_id"] for c in filtered]
        semantic_top = semantic_scorer.get_top_n_by_similarity(filtered_ids, n=top_n)
        semantic_top_ids = {cid for cid, _ in semantic_top}

    pipeline_log.append({
        "stage": "Semantic Similarity",
        "input": len(filtered),
        "output": len(semantic_top_ids),
        "time": time.time() - stage_start,
        "method": "TF-IDF Fallback" if semantic_scorer.is_fallback else "Sentence-Transformers",
    })

    # Stage 4: Deep Scoring
    stage_start = time.time()
    progress_bar.progress(0.60, text="📊 Stage 4: Deep multi-dimensional scoring...")

    candidate_map = {c["candidate_id"]: c for c in candidates}
    scored_candidates = []

    for i, cid in enumerate(semantic_top_ids):
        candidate = candidate_map[cid]
        similarity = semantic_scorer.get_similarity(cid)
        breakdown = compute_deep_score(candidate, similarity)

        scored_candidates.append({
            "candidate_id": cid,
            "candidate": candidate,
            "score_breakdown": breakdown,
            "final_score": breakdown["final_score"],
        })

        if i % 50 == 0:
            progress_bar.progress(
                min(0.85, 0.60 + i / len(semantic_top_ids) * 0.25),
                text=f"📊 Stage 4: Deep scoring {i+1}/{len(semantic_top_ids)}..."
            )

    scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)

    pipeline_log.append({
        "stage": "Deep Scoring",
        "input": len(semantic_top_ids),
        "output": len(scored_candidates),
        "time": time.time() - stage_start,
    })

    # Stage 5: Reasoning
    stage_start = time.time()
    progress_bar.progress(0.88, text="💬 Stage 5: Generating reasoning...")

    top_100 = scored_candidates[:100]
    results = []

    for rank, entry in enumerate(top_100, start=1):
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
            "score_breakdown": entry["score_breakdown"],
            "candidate": entry["candidate"],
        })

    pipeline_log.append({
        "stage": "Reasoning Generation",
        "input": len(top_100),
        "output": len(results),
        "time": time.time() - stage_start,
    })

    total_time = time.time() - total_start
    progress_bar.progress(1.0, text=f"✅ Pipeline complete in {total_time:.1f}s")

    return results, pipeline_log, honeypot_ids


# ═══════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # Data source
    st.markdown("### 📁 Candidate Data")
    data_source = st.radio(
        "Data source:",
        ["Sample (20 candidates)", "Full dataset (100K)", "Upload custom"],
        index=0,
        help="Start with the sample for quick testing, then run on full data.",
    )

    uploaded_file = None
    if data_source == "Upload custom":
        uploaded_file = st.file_uploader(
            "Upload .json or .jsonl",
            type=["json", "jsonl"],
        )

    st.markdown("---")

    # Embeddings
    st.markdown("### 🧠 Embeddings")
    embeddings_path = st.text_input(
        "Embeddings .npz path:",
        value="embeddings.npz",
        help="Path to pre-computed embeddings. Leave default if not generated yet.",
    )

    st.markdown("---")

    # Scoring weights (adjustable)
    st.markdown("### ⚖️ Scoring Weights")
    w_semantic = st.slider("Semantic", 0.0, 1.0, 0.25, 0.05)
    w_title = st.slider("Title & Career", 0.0, 1.0, 0.25, 0.05)
    w_skills = st.slider("Skills", 0.0, 1.0, 0.20, 0.05)
    w_experience = st.slider("Experience", 0.0, 1.0, 0.10, 0.05)
    w_education = st.slider("Education", 0.0, 1.0, 0.05, 0.05)
    w_behavioral = st.slider("Behavioral", 0.0, 1.0, 0.15, 0.05)

    total_weight = w_semantic + w_title + w_skills + w_experience + w_education + w_behavioral
    if abs(total_weight - 1.0) > 0.01:
        st.warning(f"Weights sum to {total_weight:.2f}, not 1.0")

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color: #64748b; font-size: 0.75rem;'>"
        "Built for the Redrob Hackathon<br/>AI Candidate Discovery Challenge"
        "</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════
# Main content
# ═══════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="main-header">
    <h1>🎯 Redrob AI Candidate Ranker</h1>
    <p>Intelligent Candidate Discovery & Ranking System — Beyond Keywords</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab_jd, tab_rank, tab_results, tab_arch = st.tabs([
    "📋 Job Description",
    "🚀 Run Ranking",
    "📊 Results & Analysis",
    "🏗️ Architecture",
])


# ═══════════════════════════════════════════════════════════════════
# Tab 1: Job Description
# ═══════════════════════════════════════════════════════════════════

with tab_jd:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🏢 Senior AI Engineer — Founding Team")
        st.markdown(f"**Company:** {COMPANY} (Series A)")
        st.markdown(f"**Experience:** {EXPERIENCE_RANGE[0]}–{EXPERIENCE_RANGE[1]} years (sweet spot: {EXPERIENCE_SWEET_SPOT[0]}–{EXPERIENCE_SWEET_SPOT[1]})")
        st.markdown(f"**Location:** Pune/Noida, India (Hybrid)")

        st.markdown("---")
        st.markdown("#### What the JD Actually Needs")
        st.markdown("""
        > The role requires someone who can **own the intelligence layer** — ranking,
        > retrieval, and matching systems. They need **deep technical depth** in modern
        > ML systems combined with a **scrappy product-engineering attitude**.
        """)

        with st.expander("📄 Full JD Text (for embedding)", expanded=False):
            st.text(JD_TEXT_FOR_EMBEDDING)

    with col2:
        st.markdown("#### ✅ Required Skills")
        required_display = sorted(list(REQUIRED_SKILLS))[:15]
        for s in required_display:
            st.markdown(f"- `{s}`")

        st.markdown("#### 💡 Desired Skills")
        desired_display = sorted(list(DESIRED_SKILLS))[:10]
        for s in desired_display:
            st.markdown(f"- `{s}`")

        st.markdown("#### ⚠️ Disqualifiers")
        st.markdown("""
        - Pure research (no production)
        - Recent-only LangChain experience
        - Hasn't coded in 18 months
        - Title-chasers
        - Consulting-only careers
        - Primary CV/speech/robotics expertise
        """)


# ═══════════════════════════════════════════════════════════════════
# Tab 2: Run Ranking
# ═══════════════════════════════════════════════════════════════════

with tab_rank:
    st.markdown("### 🚀 Run the Ranking Pipeline")

    # Determine data path
    data_dir = Path(__file__).parent / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge"

    if data_source == "Sample (20 candidates)":
        default_path = str(data_dir / "sample_candidates.json")
        st.info(f"📂 Using sample dataset: `{default_path}`")
    elif data_source == "Full dataset (100K)":
        default_path = str(data_dir / "candidates.jsonl")
        st.info(f"📂 Using full dataset: `{default_path}`")
    else:
        default_path = None
        if uploaded_file:
            st.info(f"📂 Using uploaded file: `{uploaded_file.name}`")
        else:
            st.warning("⬆️ Please upload a candidate file in the sidebar.")

    # Pipeline diagram
    with st.expander("🔄 Pipeline Architecture", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown("""
            <div style="text-align:center; padding: 10px;">
                <div style="font-size: 2rem;">🛡️</div>
                <div style="color: #a5b4fc; font-weight: 700; font-size: 0.85rem;">Stage 1</div>
                <div style="color: #94a3b8; font-size: 0.75rem;">Honeypot<br/>Detection</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div style="text-align:center; padding: 10px;">
                <div style="font-size: 2rem;">⚡</div>
                <div style="color: #a5b4fc; font-weight: 700; font-size: 0.85rem;">Stage 2</div>
                <div style="color: #94a3b8; font-size: 0.75rem;">Hard<br/>Filters</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div style="text-align:center; padding: 10px;">
                <div style="font-size: 2rem;">🧠</div>
                <div style="color: #a5b4fc; font-weight: 700; font-size: 0.85rem;">Stage 3</div>
                <div style="color: #94a3b8; font-size: 0.75rem;">Semantic<br/>Similarity</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown("""
            <div style="text-align:center; padding: 10px;">
                <div style="font-size: 2rem;">📊</div>
                <div style="color: #a5b4fc; font-weight: 700; font-size: 0.85rem;">Stage 4</div>
                <div style="color: #94a3b8; font-size: 0.75rem;">Deep<br/>Scoring</div>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            st.markdown("""
            <div style="text-align:center; padding: 10px;">
                <div style="font-size: 2rem;">💬</div>
                <div style="color: #a5b4fc; font-weight: 700; font-size: 0.85rem;">Stage 5</div>
                <div style="color: #94a3b8; font-size: 0.75rem;">Reasoning<br/>Generation</div>
            </div>
            """, unsafe_allow_html=True)

    # Run button
    if st.button("▶️  Run Ranking Pipeline", type="primary", use_container_width=True):
        # Load data
        with st.spinner("Loading candidate data..."):
            if uploaded_file:
                candidates = load_candidates_from_file(uploaded_file=uploaded_file)
            else:
                candidates = load_candidates_from_file(default_path=default_path)

        if candidates is None or len(candidates) == 0:
            st.error("❌ No candidates loaded. Check your data source.")
        else:
            st.success(f"✅ Loaded {len(candidates)} candidates")

            # Update scoring weights
            import scoring.deep_scorer as ds_module
            import jd_config
            jd_config.SCORING_WEIGHTS = {
                "semantic": w_semantic,
                "title_career": w_title,
                "skills": w_skills,
                "experience": w_experience,
                "education": w_education,
                "behavioral": w_behavioral,
            }

            # Run pipeline
            results, pipeline_log, honeypot_ids = run_ranking_pipeline(
                candidates, embeddings_path
            )

            # Store in session state
            st.session_state["results"] = results
            st.session_state["pipeline_log"] = pipeline_log
            st.session_state["honeypot_ids"] = honeypot_ids
            st.session_state["candidates"] = candidates

            # Show pipeline summary
            st.markdown("### 📊 Pipeline Summary")

            for log in pipeline_log:
                st.markdown(
                    f"""<div class="pipeline-stage">
                        <span class="stage-num">{log['stage']}</span>:
                        {log['input']} → {log['output']} candidates
                        <span class="stage-result">{log['time']:.1f}s</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

            st.markdown("---")
            st.success("🎉 **Ranking complete!** Switch to the **Results & Analysis** tab to explore.")


# ═══════════════════════════════════════════════════════════════════
# Tab 3: Results & Analysis
# ═══════════════════════════════════════════════════════════════════

with tab_results:
    if "results" not in st.session_state:
        st.info("🔄 Run the ranking pipeline first to see results here.")
    else:
        results = st.session_state["results"]
        pipeline_log = st.session_state.get("pipeline_log", [])
        honeypot_ids = st.session_state.get("honeypot_ids", set())

        # ── Summary stats ──
        st.markdown("### 📈 Overview")
        s1, s2, s3, s4 = st.columns(4)

        with s1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{len(results)}</div>
                <div class="stat-label">Ranked Candidates</div>
            </div>
            """, unsafe_allow_html=True)
        with s2:
            avg_score = np.mean([r["score"] for r in results]) if results else 0
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{avg_score:.3f}</div>
                <div class="stat-label">Avg Score</div>
            </div>
            """, unsafe_allow_html=True)
        with s3:
            top_score = results[0]["score"] if results else 0
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{top_score:.3f}</div>
                <div class="stat-label">Top Score</div>
            </div>
            """, unsafe_allow_html=True)
        with s4:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{len(honeypot_ids)}</div>
                <div class="stat-label">Honeypots Caught</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Results table ──
        st.markdown("### 🏆 Ranked Candidates")

        # Build display dataframe
        display_data = []
        for r in results:
            profile = r["candidate"].get("profile", {})
            signals = r["candidate"].get("redrob_signals", {})
            breakdown = r.get("score_breakdown", {})
            display_data.append({
                "Rank": r["rank"],
                "Candidate ID": r["candidate_id"],
                "Score": r["score"],
                "Title": profile.get("current_title", ""),
                "Company": profile.get("current_company", ""),
                "YoE": profile.get("years_of_experience", 0),
                "Location": profile.get("location", ""),
                "Semantic": breakdown.get("semantic", 0),
                "Title/Career": breakdown.get("title_career", 0),
                "Skills": breakdown.get("skills", 0),
                "Behavioral": breakdown.get("behavioral_modifier", 0),
                "Reasoning": r["reasoning"],
            })

        df_display = pd.DataFrame(display_data)
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score", min_value=0, max_value=1, format="%.4f",
                ),
                "Semantic": st.column_config.ProgressColumn(
                    "Semantic", min_value=0, max_value=1, format="%.3f",
                ),
                "Title/Career": st.column_config.ProgressColumn(
                    "Title/Career", min_value=0, max_value=1, format="%.3f",
                ),
                "Skills": st.column_config.ProgressColumn(
                    "Skills", min_value=0, max_value=1, format="%.3f",
                ),
                "Behavioral": st.column_config.NumberColumn(
                    "Behavioral", format="%.3f",
                ),
                "YoE": st.column_config.NumberColumn("YoE", format="%.1f"),
            },
        )

        st.markdown("---")

        # ── Score Distribution Chart ──
        st.markdown("### 📊 Score Distribution")
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            scores = [r["score"] for r in results]
            chart_df = pd.DataFrame({"Score": scores, "Rank": range(1, len(scores) + 1)})
            st.line_chart(chart_df.set_index("Rank"), y="Score", use_container_width=True)

        with chart_col2:
            # Component averages
            components = ["Semantic", "Title/Career", "Skills", "Behavioral"]
            avg_values = [df_display[c].mean() for c in components]
            comp_df = pd.DataFrame({"Component": components, "Avg Score": avg_values})
            st.bar_chart(comp_df.set_index("Component"), y="Avg Score", use_container_width=True)

        st.markdown("---")

        # ── Candidate Deep Dive ──
        st.markdown("### 🔍 Candidate Deep Dive")

        selected_rank = st.selectbox(
            "Select a candidate to inspect:",
            options=[f"Rank {r['rank']} — {r['candidate_id']} ({r['candidate']['profile'].get('current_title', '')})" for r in results],
            index=0,
        )

        if selected_rank:
            idx = int(selected_rank.split("Rank ")[1].split(" —")[0]) - 1
            r = results[idx]
            candidate = r["candidate"]
            profile = candidate.get("profile", {})
            signals = candidate.get("redrob_signals", {})
            breakdown = r.get("score_breakdown", {})

            # Profile header
            col_profile, col_scores = st.columns([1, 1])

            with col_profile:
                st.markdown(f"""
                <div class="candidate-card">
                    <div style="display: flex; align-items: center;">
                        {rank_badge_html(r['rank'])}
                        <div>
                            <h3>{profile.get('anonymized_name', 'Unknown')}</h3>
                            <div class="subtitle">{profile.get('headline', '')}</div>
                        </div>
                    </div>
                    <div class="meta">
                        📍 {profile.get('location', 'Unknown')}, {profile.get('country', '')} •
                        🏢 {profile.get('current_company', '')} ({profile.get('current_industry', '')}) •
                        📅 {profile.get('years_of_experience', 0):.1f} years
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Summary
                st.markdown("**Summary**")
                st.caption(profile.get("summary", "No summary available."))

                # Skills
                st.markdown("**Skills**")
                skills_html = ""
                for s in candidate.get("skills", []):
                    name = s.get("name", "")
                    # Check if matched
                    n_lower = name.lower()
                    is_matched = any(
                        req in n_lower or n_lower in req
                        for req in REQUIRED_SKILLS | DESIRED_SKILLS
                    )
                    cls = "skill-pill matched" if is_matched else "skill-pill"
                    prof = s.get("proficiency", "")
                    skills_html += f'<span class="{cls}" title="{prof} | {s.get("endorsements", 0)} endorsements">{name}</span> '
                st.markdown(skills_html, unsafe_allow_html=True)

                # Career History
                st.markdown("**Career History**")
                for entry in candidate.get("career_history", []):
                    is_current = "🟢" if entry.get("is_current") else "⚪"
                    st.markdown(
                        f"{is_current} **{entry.get('title', '')}** at {entry.get('company', '')} "
                        f"({entry.get('duration_months', 0)} months, {entry.get('industry', '')})"
                    )
                    with st.expander("Role description", expanded=False):
                        st.caption(entry.get("description", ""))

            with col_scores:
                st.markdown("**Score Breakdown**")
                st.markdown(
                    score_bar_html(breakdown.get("semantic", 0), "#818cf8", "Semantic")
                    + score_bar_html(breakdown.get("title_career", 0), "#6366f1", "Title/Career")
                    + score_bar_html(breakdown.get("skills", 0), "#a78bfa", "Skills")
                    + score_bar_html(breakdown.get("experience", 0), "#c084fc", "Experience")
                    + score_bar_html(breakdown.get("education", 0), "#e879f9", "Education")
                    + score_bar_html(breakdown.get("behavioral_modifier", 0), "#22d3ee", "Behavioral"),
                    unsafe_allow_html=True,
                )

                st.markdown(f"**Final Score:** {score_badge_html(r['score'])}", unsafe_allow_html=True)

                # Behavioral breakdown
                st.markdown("---")
                st.markdown("**Behavioral Signals**")
                beh = get_behavioral_breakdown(candidate)
                for key, val in beh.items():
                    if key != "modifier" and key != "note":
                        st.markdown(
                            score_bar_html(val, "#22d3ee", key.replace("_", " ").title()),
                            unsafe_allow_html=True,
                        )

                # Key signals
                st.markdown("---")
                st.markdown("**Key Platform Signals**")
                sig_cols = st.columns(2)
                with sig_cols[0]:
                    st.metric("Response Rate", f"{signals.get('recruiter_response_rate', 0):.0%}")
                    st.metric("Notice Period", f"{signals.get('notice_period_days', 0)}d")
                    st.metric("GitHub Score", f"{signals.get('github_activity_score', -1):.0f}")
                with sig_cols[1]:
                    st.metric("Open to Work", "Yes" if signals.get("open_to_work_flag") else "No")
                    st.metric("Interview Rate", f"{signals.get('interview_completion_rate', 0):.0%}")
                    st.metric("Work Mode", signals.get("preferred_work_mode", "N/A"))

            # Reasoning
            st.markdown("---")
            st.markdown("**📝 Ranking Reasoning**")
            st.info(r["reasoning"])

        st.markdown("---")

        # ── Export ──
        st.markdown("### 💾 Export Submission")
        export_df = df_display[["Candidate ID", "Rank", "Score", "Reasoning"]].copy()
        export_df.columns = ["candidate_id", "rank", "score", "reasoning"]

        csv_data = export_df.to_csv(index=False, quoting=1)
        st.download_button(
            label="📥 Download submission.csv",
            data=csv_data,
            file_name="submission.csv",
            mime="text/csv",
            type="primary",
            use_container_width=True,
        )


# ═══════════════════════════════════════════════════════════════════
# Tab 4: Architecture
# ═══════════════════════════════════════════════════════════════════

with tab_arch:
    st.markdown("### 🏗️ System Architecture")

    st.markdown("""
    #### Hybrid Scoring Pipeline

    Our system uses a **5-stage funnel architecture** that progressively narrows
    100K candidates down to a ranked top 100:

    ```
    100K Candidates
     ├─► Stage 1: Honeypot Detection (~80 removed)
     │    └─ 5 heuristic checks: experience inflation, keyword stuffing,
     │       assessment contradictions, impossible timelines, completeness mismatch
     │
     ├─► Stage 2: Hard Filters (→ ~10-20K)
     │    └─ Experience range, title relevance, AI/ML career signals
     │
     ├─► Stage 3: Semantic Similarity (→ ~500)
     │    └─ Pre-computed sentence-transformer embeddings (all-MiniLM-L6-v2)
     │       or TF-IDF fallback if embeddings unavailable
     │
     ├─► Stage 4: Deep Multi-Dimensional Scoring
     │    └─ 6 weighted components:
     │       • Semantic similarity (0.25)
     │       • Title & career trajectory (0.25)
     │       • Skills with trust multiplier (0.20)
     │       • Experience fit (0.10)
     │       • Education tier (0.05)
     │       • Behavioral modifier (0.15)
     │
     └─► Stage 5: Reasoning Generation (→ Top 100)
          └─ Fact-based, specific reasoning referencing actual profile data
    ```

    #### Why This Architecture?

    | Approach | Pros | Cons | Our Verdict |
    |---|---|---|---|
    | Keyword/BM25 | Fast | Falls for keyword stuffers | ❌ |
    | Pure LLM | Deepest understanding | Can't run offline in 5 min | ❌ |
    | Pure embeddings | Captures semantics | Misses behavioral signals | ⚠️ Partial |
    | **Hybrid funnel** | Fast + semantic + behavioral | More complex | ✅ Chosen |

    #### Key Design Decisions

    1. **Honeypot detection is the first stage**, not an afterthought. We identify
       ~80 fake profiles using career timeline analysis, not keyword matching.

    2. **Behavioral signals are multiplicative**, not additive. A candidate with a
       perfect skills match but 5% response rate gets a 0.5× modifier, effectively
       dropping them 50 ranks.

    3. **Skills trust multiplier** cross-validates claimed proficiency against
       endorsements, duration, and Redrob assessment scores. This catches keyword
       stuffers that honeypot detection might miss.

    4. **The reasoning generator is fact-based**, pulling specific data points from
       each candidate's profile. No templates, no hallucination.

    #### Compute Budget

    | Step | Time (100K on CPU) |
    |---|---|
    | Pre-computation (one-time) | ~20 min |
    | Stage 1: Honeypots | ~3s |
    | Stage 2: Hard filters | ~5s |
    | Stage 3: Semantic similarity | ~2s (pre-computed) |
    | Stage 4: Deep scoring | ~5s |
    | Stage 5: Reasoning | ~1s |
    | **Total ranking** | **~16s** ✅ |
    """)
