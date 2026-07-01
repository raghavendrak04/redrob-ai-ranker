"""
jd_config.py — Structured representation of the Senior AI Engineer JD.

This file contains the parsed job description as structured data so the ranking
system doesn't need to re-parse the docx at runtime. Every field here is
derived from the actual job_description.docx in the challenge bundle.
"""

# ============================================================================
# Role identity
# ============================================================================
JOB_TITLE = "Senior AI Engineer"
COMPANY = "Redrob AI"
EMPLOYMENT_TYPE = "Full-time"
EXPERIENCE_RANGE = (5, 9)         # stated range
EXPERIENCE_SWEET_SPOT = (6, 8)    # "ideal candidate" section
EXPERIENCE_HARD_MIN = 3           # generous lower bound we allow
EXPERIENCE_HARD_MAX = 15          # generous upper bound we allow

# ============================================================================
# Location preferences
# ============================================================================
PREFERRED_LOCATIONS = [
    "pune", "noida", "hyderabad", "mumbai", "delhi", "delhi ncr",
    "bengaluru", "bangalore", "gurugram", "gurgaon",
]
PREFERRED_COUNTRIES = ["india"]
LOCATION_FLEXIBLE = True  # open to relocation candidates

# ============================================================================
# Work mode
# ============================================================================
PREFERRED_WORK_MODES = ["hybrid", "onsite", "flexible"]

# ============================================================================
# Skills — tiered by JD importance
# ============================================================================

# "Things you absolutely need"
REQUIRED_SKILLS = {
    # Embeddings & retrieval
    "embeddings", "sentence-transformers", "sentence transformers",
    "openai embeddings", "bge", "e5", "embedding", "vector embeddings",
    "retrieval", "information retrieval", "semantic search", "dense retrieval",

    # Vector DBs & hybrid search
    "pinecone", "weaviate", "qdrant", "milvus", "opensearch",
    "elasticsearch", "faiss", "vector database", "hybrid search",
    "vector search", "annoy", "chromadb",

    # Python (strong)
    "python",

    # Evaluation frameworks
    "ndcg", "mrr", "map", "evaluation", "a/b testing", "ranking evaluation",
    "offline evaluation", "ranking metrics",
}

# "Things we'd like you to have"
DESIRED_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning", "fine tuning",
    "fine-tuning llms", "llm fine-tuning",
    "xgboost", "learning to rank", "learning-to-rank", "lambdamart",
    "hr-tech", "recruiting", "marketplace",
    "distributed systems", "inference optimization",
    "open-source", "open source",
}

# Core AI/ML skills that indicate genuine technical depth
CORE_AI_ML_SKILLS = {
    "machine learning", "deep learning", "nlp",
    "natural language processing", "neural networks",
    "pytorch", "tensorflow", "transformers", "huggingface",
    "bert", "gpt", "llm", "large language models",
    "reinforcement learning", "recommendation systems",
    "search ranking", "ranking", "retrieval",
    "data science", "mlops", "ml engineering",
    "scikit-learn", "sklearn", "keras",
    "computer vision",  # acceptable if combined with NLP
    "rag", "retrieval augmented generation",
    "langchain",  # acceptable but not sufficient alone
    "model deployment", "model serving",
    "feature engineering", "data pipelines",
    "spark", "airflow", "data engineering",
    "sql", "nosql", "mongodb", "postgresql",
}

# Skills that indicate the candidate is NOT a fit (primary expertise)
NON_RELEVANT_PRIMARY_SKILLS = {
    "robotics", "speech", "speech recognition",
    "tts", "text to speech", "image classification",
    "object detection",
}

# ============================================================================
# Title relevance tiers
# ============================================================================

# Tier 1: Directly relevant titles
TIER_1_TITLES = [
    "ai engineer", "senior ai engineer", "lead ai engineer",
    "ml engineer", "senior ml engineer", "machine learning engineer",
    "senior machine learning engineer", "lead ml engineer",
    "data scientist", "senior data scientist", "lead data scientist",
    "applied scientist", "research engineer",
    "nlp engineer", "search engineer", "ranking engineer",
    "recommendation engineer", "retrieval engineer",
    "ml platform engineer", "mlops engineer",
]

# Tier 2: Adjacent titles with potential relevance
TIER_2_TITLES = [
    "backend engineer", "software engineer", "senior software engineer",
    "full stack engineer", "platform engineer",
    "data engineer", "senior data engineer", "analytics engineer",
    "technical lead", "tech lead", "engineering manager",
    "staff engineer", "principal engineer",
]

# Tier 3: Roles that are typically NOT relevant
NON_RELEVANT_TITLES = [
    "marketing manager", "hr manager", "human resources",
    "accountant", "sales executive", "content writer",
    "graphic designer", "operations manager", "customer support",
    "civil engineer", "mechanical engineer", "electrical engineer",
    "project manager", "business analyst", "product manager",
    "financial analyst", "legal counsel",
]

# ============================================================================
# Consulting / services companies (explicit JD disqualifier)
# ============================================================================
CONSULTING_COMPANIES = {
    "tcs", "tata consultancy services",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "hcl", "hcl technologies",
    "tech mahindra",
    "mindtree",  # now part of LTIMindtree
    "ltimindtree",
    "mphasis",
    "persistent systems",
    "l&t infotech", "lti",
}

# Product companies (positive signal)
PRODUCT_COMPANIES_EXAMPLES = {
    "google", "meta", "facebook", "amazon", "microsoft", "apple",
    "netflix", "uber", "airbnb", "stripe", "shopify",
    "flipkart", "zomato", "swiggy", "razorpay", "cred",
    "meesho", "zerodha", "dream11", "ola", "phonepe",
    "paytm", "byju's", "unacademy", "groww", "slice",
    "freshworks", "zoho", "postman", "browserstack",
    "thoughtspot", "hasura", "chargebee",
}

# ============================================================================
# Notice period preferences
# ============================================================================
IDEAL_NOTICE_PERIOD_DAYS = 30
MAX_ACCEPTABLE_NOTICE_DAYS = 90

# ============================================================================
# JD text for semantic embedding (condensed version for matching)
# ============================================================================
JD_TEXT_FOR_EMBEDDING = """
Senior AI Engineer — Founding Team at Redrob AI, an AI-native talent intelligence platform.

Core mandate: Own the intelligence layer — ranking, retrieval, and matching systems
that decide what recruiters see when they search for candidates.

Required technical depth:
- Production experience with embeddings-based retrieval systems (sentence-transformers,
  OpenAI embeddings, BGE, E5) deployed to real users
- Production experience with vector databases or hybrid search (Pinecone, Weaviate,
  Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS)
- Strong Python, code quality matters
- Evaluation frameworks for ranking systems: NDCG, MRR, MAP, A/B testing

Day-to-day work: Audit BM25 + rule-based scoring, ship v2 ranking with embeddings
and hybrid retrieval, set up evaluation infrastructure with offline benchmarks
and online A/B testing.

Ideal candidate: 6-8 years total experience, 4-5 in applied ML/AI at product companies.
Has shipped end-to-end ranking, search, or recommendation system to real users.
Strong opinions on retrieval (hybrid vs dense), evaluation (offline vs online),
LLM integration (when to fine-tune vs prompt).

Must be a "shipper" not just a "researcher". Comfortable with scrappy product engineering.
Located in India, preferably Pune or Noida. Active on job market.

Disqualifiers: pure research without production, recent-only LangChain experience,
hasn't written code in 18 months, title-chasers, framework enthusiasts,
consulting-only careers (TCS, Infosys, Wipro, etc.), primary expertise in
computer vision/speech/robotics without NLP/IR exposure.
"""

# ============================================================================
# Scoring weights (final composite)
# ============================================================================
SCORING_WEIGHTS = {
    "semantic": 0.25,
    "title_career": 0.25,
    "skills": 0.20,
    "experience": 0.10,
    "education": 0.05,
    "behavioral": 0.15,
}
