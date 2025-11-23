"""
Runtime state package for Robot Savo LLM server.

This package is responsible for tracking per-session conversation state,
so the robot can have multi-turn conversations without mixing users.

Typical usage (e.g. in pipeline.py):

    from app.runtime_state import session_store

    session = session_store.get_or_create_session(session_id)
    history_messages = session_store.get_history_as_messages(session_id)
    # ... pass `history_messages` into Tier1/Tier2 prompts ...

    session_store.update_from_interaction(
        session_id=session_id,
        user_text=user_text,
        assistant_text=reply_text,
        intent=final_intent,
        nav_goal=final_nav_goal,
    )
"""

from .sessions import (
    SessionTurn,
    SessionData,
    RuntimeState,
    SessionStore,
    session_store,
)

__all__ = [
    "SessionTurn",
    "SessionData",
    "RuntimeState",
    "SessionStore",
    "session_store",
]
