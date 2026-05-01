"""
PackPal Configuration & Environment Loader
Loads secrets securely and defines app-wide constants.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")

# --- API Keys & Credentials ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GOOGLE_VISION_API_KEY = os.getenv("GOOGLE_VISION_API_KEY", "")

# --- Supabase ---
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# --- Neo4j Aura ---
NEO4J_URI = os.getenv("NEO4J_URI", "")      # e.g., neo4j+s://<id>.databases.neo4j.io
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# --- Cloudflare R2 (S3 Compatible) ---
R2_ENDPOINT_URL = f"https://{os.getenv('R2_ACCOUNT_ID')}.r2.cloudflarestorage.com"
R2_PUBLIC_BASE_URL = "https://pub-bd367cd29e634c4c8c5abf04e574a1e4.r2.dev"
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET", "")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME", "packpal")

# --- App Constants ---
DEFAULT_FARE_CLASS = "economy"
DEFAULT_COLD_TOLERANCE = "standard"
LOCAL_UPLOAD_DIR = ROOT_DIR / "data" / "local_uploads"
LOCAL_UPLOAD_DIR.mkdir(parents=True, exist_ok=True) # Ensure exists for fallback mode

# Dependency Availability Flags (Checked once at startup)
DEPS_AVAILABLE = {
    "neo4j": False,
    "supabase": False,
    "boto3": False,
    "groq": False,
}

try:
    import neo4j
    DEPS_AVAILABLE["neo4j"] = True
except ImportError: pass

try:
    import supabase
    DEPS_AVAILABLE["supabase"] = True
except ImportError: pass

try:
    import boto3
    DEPS_AVAILABLE["boto3"] = True
except ImportError: pass

try:
    import langchain_groq
    DEPS_AVAILABLE["groq"] = True
except ImportError: pass

    
