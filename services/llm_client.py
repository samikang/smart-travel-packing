"""
LLM Client (LangChain + Groq)
Initializes the chat model for slot filling and XAI explanations.
"""
from config.settings import GROQ_API_KEY, DEPS_AVAILABLE

_llm = None

def get_llm(temperature: float = 0.1):
    """
    Returns a LangChain ChatGroq instance.
    Temperature 0.1 for strict slot filling, higher for XAI narratives.
    """
    global _llm
    if _llm is None:
        if DEPS_AVAILABLE["groq"] and GROQ_API_KEY:
            from langchain_groq import ChatGroq
            _llm = ChatGroq(
                model="llama-3.3-70b-versatile", # Fast, excellent reasoning
                api_key=GROQ_API_KEY,
                temperature=temperature
            )
        else:
            # We return None here. The slot_detection.py will catch this
            # and route to a strict Regex/Keyword fallback.
            pass
    return _llm

def is_llm_available() -> bool:
    return get_llm() is not None