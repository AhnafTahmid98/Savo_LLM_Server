# app/runtime_state/sessions.py
# -*- coding: utf-8 -*-
"""
Robot Savo — Runtime Session State
----------------------------------

This module implements a simple, file-backed session store for the LLM server.

Purpose
~~~~~~~
- Track per-session conversation history so Robot Savo can have multi-turn
  conversations without mixing different users.
- Remember the last intent and navigation goal for each active session.
- Persist state to disk so a server restart does not immediately lose context.

Design notes
~~~~~~~~~~~~
- Backed by a single JSON file (sessions.json) located next to this module.
- Assumes a single FastAPI worker process (no multi-process concurrency).
  If you later scale out, move this to Redis/PostgreSQL instead.
- History is automatically truncated to keep token usage under control.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from app.utils import get_logger, read_json_safely, write_json_atomic


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = get_logger("robot_savo.runtime_state")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SessionTurn(BaseModel):
    """One turn in the conversation history."""

    role: Literal["user", "assistant"]
    text: str
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionData(BaseModel):
    """
    Per-session state.

    Attributes
    ----------
    session_id:
        Unique key for the session (provided by the Pi/client).
    created_at:
        When this session was first created.
    last_seen:
        Last time we updated this session (used for pruning stale sessions).
    last_intent:
        Last classified intent (NAVIGATE, FOLLOW, STOP, STATUS, CHATBOT).
    last_nav_goal:
        Last resolved navigation goal (e.g. "A201") or None.
    history:
        Recent conversation turns, truncated to keep token usage small.
    summary:
        Optional short summary of the conversation so far (for future use).
    """

    session_id: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_seen: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_intent: Optional[str] = None
    last_nav_goal: Optional[str] = None
    history: List[SessionTurn] = Field(default_factory=list)
    summary: Optional[str] = None


class RuntimeState(BaseModel):
    """Top-level container for all sessions stored on disk."""

    sessions: Dict[str, SessionData] = Field(default_factory=dict)


# Default JSON path lives next to this file
DEFAULT_SESSIONS_PATH = Path(__file__).with_name("sessions.json")


# ---------------------------------------------------------------------------
# Session store implementation
# ---------------------------------------------------------------------------


class SessionStore:
    """
    File-backed session store for Robot Savo LLM server.

    This implementation keeps all state in memory and syncs it to a JSON file
    on disk. It is intentionally simple and single-process only.

    Parameters
    ----------
    path:
        Optional override for the path to the JSON file. By default, it uses
        `sessions.json` next to this module.
    max_history_turns:
        Maximum number of turns (user + assistant) to keep per session.
        Older turns are dropped from the front of the history list.
    auto_persist:
        If True, writes to disk after each update. You can set this to False
        in tests and call `_sync()` manually.
    """

    def __init__(
        self,
        path: Optional[Union[Path, str]] = None,
        max_history_turns: int = 8,
        auto_persist: bool = True,
    ) -> None:
        self.path: Path = Path(path) if path is not None else DEFAULT_SESSIONS_PATH
        self.max_history_turns = max_history_turns
        self.auto_persist = auto_persist

        self.state: RuntimeState = self._load_from_disk()

    # ------------------------------------------------------------------
    # Low-level I/O
    # ------------------------------------------------------------------

    def _load_from_disk(self) -> RuntimeState:
        """
        Load runtime state from disk.

        Behaviour:
        - If the file does not exist:
            - create an empty RuntimeState
            - persist it to disk
        - If loading/parsing fails:
            - log a warning
            - start with an empty RuntimeState (do NOT crash the server)
        """
        if not self.path.exists():
            logger.info(
                "[SessionStore] No existing sessions file at %s, creating a new one.",
                self.path,
            )
            empty = RuntimeState()
            try:
                write_json_atomic(self.path, empty.model_dump(mode="json"))
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(
                    "[SessionStore] Failed to create initial sessions file %s: %s",
                    self.path,
                    exc,
                )
            return empty

        raw: Dict[str, Any] = read_json_safely(
            self.path,
            default={"sessions": {}},
            log_missing=False,
        ) or {"sessions": {}}

        try:
            state = RuntimeState.model_validate(raw)
            logger.info(
                "[SessionStore] Loaded %d sessions from %s",
                len(state.sessions),
                self.path,
            )
            return state
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "[SessionStore] Failed to validate sessions from %s: %s; "
                "starting with empty state.",
                self.path,
                exc,
            )
            return RuntimeState()

    def _write_to_disk(self, state: RuntimeState) -> None:
        """
        Atomically write the given state to disk.

        Uses the shared write_json_atomic helper to avoid partial writes.
        """
        payload = state.model_dump(mode="json")
        write_json_atomic(self.path, payload)

    def _sync(self) -> None:
        """Persist the current in-memory state to disk."""
        try:
            self._write_to_disk(self.state)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("[SessionStore] Failed to sync sessions to disk: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_create_session(self, session_id: str) -> SessionData:
        """
        Retrieve an existing session or create a new one.

        Updates `last_seen` to now.
        """
        if session_id not in self.state.sessions:
            logger.info("[SessionStore] Creating new session %s", session_id)
            self.state.sessions[session_id] = SessionData(session_id=session_id)

        session = self.state.sessions[session_id]
        session.last_seen = datetime.now(timezone.utc)
        return session

    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Return the SessionData for `session_id`, or None if not found."""
        return self.state.sessions.get(session_id)

    def delete_session(self, session_id: str) -> None:
        """Delete a session and persist the change."""
        if session_id in self.state.sessions:
            logger.info("[SessionStore] Deleting session %s", session_id)
            del self.state.sessions[session_id]
            if self.auto_persist:
                self._sync()

    def update_from_interaction(
        self,
        session_id: str,
        user_text: Optional[str],
        assistant_text: Optional[str],
        intent: Optional[str],
        nav_goal: Optional[str],
        summary: Optional[str] = None,
    ) -> SessionData:
        """
        Update session after a /chat call has been processed.

        - Adds user + assistant turns (if provided).
        - Updates last_intent / last_nav_goal / summary.
        - Trims history to at most `max_history_turns`.
        - Persists state to disk (if auto_persist=True).
        """
        session = self.get_or_create_session(session_id)

        now = datetime.now(timezone.utc)
        session.last_seen = now

        if user_text is not None:
            session.history.append(
                SessionTurn(role="user", text=user_text, ts=now)
            )

        if assistant_text is not None:
            session.history.append(
                SessionTurn(role="assistant", text=assistant_text, ts=now)
            )

        if intent is not None:
            session.last_intent = intent

        if nav_goal is not None:
            session.last_nav_goal = nav_goal

        if summary is not None:
            session.summary = summary

        # Keep only the last N turns to control token usage
        if len(session.history) > self.max_history_turns:
            session.history = session.history[-self.max_history_turns :]

        self.state.sessions[session_id] = session

        if self.auto_persist:
            self._sync()

        return session

    def get_history_as_messages(self, session_id: str) -> List[Dict[str, str]]:
        """
        Return history in OpenAI-style message format:

            [
              {"role": "user", "content": "..."},
              {"role": "assistant", "content": "..."},
              ...
            ]

        If no session exists, returns an empty list.
        """
        session = self.get_session(session_id)
        if not session:
            return []

        messages: List[Dict[str, str]] = []
        for turn in session.history:
            messages.append({"role": turn.role, "content": turn.text})
        return messages

    def prune_stale_sessions(self, max_age_seconds: int) -> int:
        """
        Remove sessions that have not been seen for more than `max_age_seconds`.

        Returns
        -------
        int
            Number of deleted sessions.
        """
        if max_age_seconds <= 0:
            return 0

        cutoff = datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)
        to_delete = [
            sid
            for sid, sess in self.state.sessions.items()
            if sess.last_seen < cutoff
        ]
        for sid in to_delete:
            logger.info(
                "[SessionStore] Pruning stale session %s (last_seen=%s)",
                sid,
                self.state.sessions[sid].last_seen,
            )
            del self.state.sessions[sid]

        if to_delete and self.auto_persist:
            self._sync()

        return len(to_delete)

    def to_dict(self) -> Dict[str, Any]:
        """Return the full runtime state as a plain dict (for debugging / admin)."""
        return self.state.model_dump(mode="json")


# Global instance used by the rest of the app
session_store = SessionStore()
