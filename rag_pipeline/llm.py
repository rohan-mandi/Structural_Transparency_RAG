"""LLM chat API wrapper with robust retry logic for Together AI.

This module provides a thin wrapper around Together's chat completions API with:
- Automatic retry on transient errors (5xx, 429, timeouts) using exponential backoff
- Long backoff strategy (~3 min total) to handle API availability issues
- Clean error propagation for non-transient errors
- Stateful client singleton to reuse connections

Transient errors cause retries; non-transient errors fail immediately so the caller
can decide whether to resume or abort.
"""
import time
from together import Together

from .config import GENERATION_MODEL, TOGETHER_API_KEY

# Global client instance (singleton pattern to reuse HTTP connections)
_client: Together | None = None

# Retry backoff schedule in seconds (exponential backoff up to 60s)
TRANSIENT_BACKOFF = [2, 5, 10, 20, 40, 60, 60]


def client() -> Together:
    """Get or initialize the global Together API client (singleton).
    
    Returns:
        Together: Authenticated API client
        
    Raises:
        RuntimeError: If TOGETHER_API_KEY is not set
    """
    global _client
    if _client is None:
        if not TOGETHER_API_KEY:
            raise RuntimeError("TOGETHER_API_KEY not set.")
        _client = Together(api_key=TOGETHER_API_KEY)
    return _client


def _is_transient(msg: str) -> bool:
    """Check if an error message indicates a transient/retryable error.
    
    Transient errors include:
    - 5xx server errors (502, 503, 504)
    - Rate limiting (429)
    - Network issues (connection, timeout, DNS)
    
    Args:
        msg: Error message string to check
        
    Returns:
        bool: True if the error is transient and should be retried
    """
    m = msg.lower()
    return (any(s in msg for s in ("502", "503", "504", "429"))
            or "connection" in m or "timeout" in m or "getaddrinfo" in m
            or "temporarily unavailable" in m)


def chat(
    messages: list[dict],
    *,
    model: str = GENERATION_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """Send a message to the LLM and get a response, with automatic retries.
    
    Args:
        messages: List of message dicts with 'role' and 'content' keys
                 (standard OpenAI format)
        model: Model ID to use (default: gpt-oss-120b)
        temperature: Sampling temperature [0, 2]. 0 = deterministic, higher = more creative
        max_tokens: Maximum tokens in the response
        
    Returns:
        str: The LLM's response text
        
    Raises:
        RuntimeError: If transient errors persist after all retries, or for non-transient errors
    """
    last_err = None
    # Try with exponential backoff
    for attempt, wait in enumerate(TRANSIENT_BACKOFF):
        try:
            resp = client().chat.completions.create(
                model=model, messages=messages,
                temperature=temperature, max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            msg = str(e)
            # If error is transient, retry after sleeping
            if _is_transient(msg):
                print(f"  transient chat error (attempt {attempt+1}, sleep {wait}s): {e.__class__.__name__}")
                last_err = e
                time.sleep(wait)
                continue
            # Non-transient error: fail immediately
            raise
    # All retries exhausted
    raise RuntimeError(
        f"Together chat API kept failing with transient errors after extended retry. "
        f"Last error: {last_err}. Try again later; eval will resume from where it left off."
    )
