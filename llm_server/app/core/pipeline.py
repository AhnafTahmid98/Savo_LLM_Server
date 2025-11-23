# app/core/pipeline.py
# -*- coding: utf-8 -*-
"""
Robot Savo — LLM Pipeline
-------------------------
High-level pipeline for handling a single /chat request:

    ChatRequest -> (intent + context + prompts) -> generate.py -> ChatResponse

This version:
- Uses deterministic intent classification (core/intent.py).
- Uses map_lookup + KnownLocations to get a canonical nav_goal.
- Loads live telemetry via:
    - NavState     (nav_state.json)
    - RobotStatus  (robot_status.json)
- Attaches a locations summary (known locations).
- Attaches live web context (weather / local time / crypto prices) via tools_web.
- Injects all of this into ChatRequest.meta so generate.py can see it.
- Calls generate.generate_reply_text(...) (3-tier chain).
- Parses the final JSON block and returns ChatResponse.
- Tracks per-session conversation history using runtime_state.session_store.

IMPORTANT:
- For non-navigation intents (CHATBOT, STATUS without movement), nav_goal is
  forced to None in the final ChatResponse. Only NAVIGATE/FOLLOW may carry
  a nav_goal (STOP is special but still not a "new" goal).
- If nav_state.json or robot_status.json do NOT exist yet, we send EMPTY
  dicts for nav_state / robot_status so the model cannot invent fake
  battery or temperature from defaults. Prompts must treat empty dict as
  "no telemetry available".
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set

from app.core.config import settings
from app.core.intent import classify_intent, extract_nav_goal, is_nav_intent
from app.core import safety
from app.core.map_lookup import resolve_nav_goal, get_known_locations
from app.core.generate import generate_reply_text
from app.core.types import ModelCallResult
from app.core import tools_web  # live web tools (weather / time / crypto)
from app.models.chat_request import ChatRequest
from app.models.chat_response import ChatResponse
from app.models.nav_state_model import NavState
from app.models.robot_status_model import RobotStatus
from app.runtime_state import session_store  # session tracking

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Telemetry + locations helpers
# ---------------------------------------------------------------------------


def _load_nav_state_for_meta() -> Dict[str, Any]:
    """
    Load NavState snapshot as JSON dict for meta.

    Behaviour:
    - If nav_state.json exists and loads: return its JSON.
    - If nav_state.json is missing: return {} (no fake "idle" state).
    - If any error occurs: log and return {}.
    """
    try:
        nav_state = NavState.load()
        data = nav_state.to_json_dict()
        # If the snapshot is completely empty, treat as no telemetry.
        if not isinstance(data, dict) or not data:
            return {}
        return data
    except FileNotFoundError:
        logger.info("NavState snapshot missing (no nav_state.json yet).")
        return {}
    except Exception:  # noqa: BLE001
        logger.exception("Failed to load NavState; using empty dict.")
        return {}


def _load_robot_status_for_meta() -> Dict[str, Any]:
    """
    Load RobotStatus snapshot as JSON dict for meta.

    Behaviour:
    - If robot_status.json exists and loads: return its JSON.
    - If robot_status.json is missing: return {} (NO default/fake values).
    - If any error occurs: log and return {}.

    This is important so that the model only sees *real* battery/temperature
    when the Pi has actually pushed telemetry.
    """
    try:
        status = RobotStatus.load()
        data = status.to_json_dict()
        if not isinstance(data, dict) or not data:
            return {}
        return data
    except FileNotFoundError:
        logger.info("RobotStatus snapshot missing (no robot_status.json yet).")
        return {}
    except Exception:  # noqa: BLE001
        logger.exception("Failed to load RobotStatus; using empty dict.")
        return {}


def _build_locations_summary() -> str:
    """
    Build a short textual summary of known locations.

    Example:
        "Known locations: A201 (Room A201 (Lab)), Info Desk (Information Desk)"
    """
    try:
        known = get_known_locations()
    except Exception:  # noqa: BLE001
        logger.exception("Failed to load known locations; using empty summary.")
        return ""

    names = known.list_canonical_names()
    if not names:
        return ""

    parts: List[str] = []
    for key in names:
        loc = known.get(key)
        if not loc:
            continue
        parts.append(f"{key} ({loc.display_name})")
    return "Known locations: " + ", ".join(parts)


# ---------------------------------------------------------------------------
# Live web tools helper (weather / time / crypto)
# ---------------------------------------------------------------------------


def _attach_live_context(
    clean_text: str,
    base_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Look at the cleaned user text and decide which live web tools to call.

    - Weather questions   -> tools_web.get_weather_current(...)
    - Time questions      -> tools_web.get_local_time(...)
    - Crypto questions    -> tools_web.get_crypto_price(...) for a small
                             allow-list of coins/fiats (BTC/ETH/DOGE/LINK vs EUR/USD).

    The result is stored under meta["live_context"] so generate.py can
    pass it to the LLM via the META: {...} line in the user prompt.
    """
    meta: Dict[str, Any] = dict(base_meta or {})
    text = clean_text.lower()

    live: Dict[str, Any] = {}

    # --- Weather triggers -------------------------------------------------
    weather_triggers = [
        "weather",
        "temperature outside",
        "outside weather",
        "cold outside",
        "hot outside",
        "rain outside",
        "snow outside",
        "forecast",
    ]
    if any(trig in text for trig in weather_triggers):
        # Kuopio approx coordinates (Savonia region)
        weather = tools_web.get_weather_current(62.89, 27.68)
        if weather is not None:
            live["weather"] = weather

    # --- Local time triggers ---------------------------------------------
    time_triggers = [
        "what time is it",
        "current time",
        "time now",
        "time in helsinki",
        "local time",
    ]
    if any(trig in text for trig in time_triggers):
        live["time"] = tools_web.get_local_time("Europe/Helsinki")

    # --- Crypto price triggers (multi-coin, allow-listed) -----------------
    # We support BTC, ETH, DOGE, LINK vs EUR/USD via tools_web.get_crypto_price.
    crypto_coin_aliases: Dict[str, List[str]] = {
        "btc": ["btc", "bitcoin", "xbt"],
        "eth": ["eth", "ethereum"],
        "doge": ["doge", "dogecoin"],
        "link": ["link", "chainlink"],
    }
    crypto_generic_triggers = [
        "crypto",
        "coin price",
        "coin prices",
        "crypto price",
        "crypto prices",
        "cryptocurrency",
    ]
    fiat_aliases: Dict[str, List[str]] = {
        "eur": ["eur", "euro", "€"],
        "usd": ["usd", "dollar", "dollars", "$", "us dollar", "us dollars"],
    }

    # Detect if the user is talking about crypto at all.
    mentions_any_crypto = any(
        alias in text for aliases in crypto_coin_aliases.values() for alias in aliases
    ) or any(trig in text for trig in crypto_generic_triggers)

    if mentions_any_crypto:
        # Decide which fiat the user seems to care about (default EUR).
        vs_currency = "eur"
        for fiat_code, aliases in fiat_aliases.items():
            if any(alias in text for alias in aliases):
                vs_currency = fiat_code
                break

        # Collect which coins are explicitly requested.
        coins_requested: Set[str] = set()
        for coin_symbol, aliases in crypto_coin_aliases.items():
            if any(alias in text for alias in aliases):
                coins_requested.add(coin_symbol)

        # If only generic "crypto" is mentioned, default to BTC.
        if not coins_requested:
            coins_requested.add("btc")

        crypto_block: Dict[str, Any] = {}

        for sym in sorted(coins_requested):
            try:
                info = tools_web.get_crypto_price(sym, vs_currency)
            except Exception:  # noqa: BLE001
                logger.exception("tools_web.get_crypto_price failed for %s/%s", sym, vs_currency)
                info = None

            if info is None:
                # Either unsupported symbol/fiat or API issue.
                continue

            key = f"{info.get('symbol', sym)}_{info.get('vs_currency', vs_currency)}"
            crypto_block[key] = info

        if crypto_block:
            live["crypto"] = crypto_block

    # Only add live_context if we actually have something
    if live:
        meta["live_context"] = live

    return meta


# ---------------------------------------------------------------------------
# JSON extraction from model output
# ---------------------------------------------------------------------------


def _extract_final_json_block(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract the last JSON object from the model's raw text output.

    We assume the model ends with something like:
        {
          "reply_text": "...",
          "intent": "NAVIGATE",
          "nav_goal": "A201"
        }
    """
    if not text:
        return None

    end = text.rfind("}")
    if end == -1:
        return None

    start = text.rfind("{", 0, end)
    if start == -1:
        return None

    candidate = text[start : end + 1]
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        return None

    return None


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


async def run_pipeline(chat_req: ChatRequest) -> ChatResponse:
    """
    Main entry point for handling a single chat request.

    Steps:
    1. Resolve session_id and load conversation history.
    2. Sanitize user text.
    3. Classify intent (STOP/FOLLOW/NAVIGATE/STATUS/CHATBOT).
    4. Resolve canonical nav_goal using map_lookup (if navigation-related).
    5. Load live telemetry (NavState + RobotStatus) and locations summary.
    6. Attach live web context (weather / time / crypto) into meta.
    7. Build a new ChatRequest with enriched .meta and cleaned text.
    8. Call generate.generate_reply_text(...) (Tier1/Tier2/Tier3).
    9. Extract final JSON block from model output.
    10. Combine deterministic intent/nav_goal with model JSON.
    11. Update runtime session state and return ChatResponse.
    """
    # ------------------------------------------------------------------
    # 1) Resolve session_id and load conversation history
    # ------------------------------------------------------------------
    # Prefer explicit field; fall back to legacy meta["session_id"]; else "default".
    session_id: str = (
        chat_req.session_id
        or (
            chat_req.meta.get("session_id")
            if isinstance(chat_req.meta, dict)
            else None
        )
        or "default"
    )

    # Pull recent history to feed into the model via meta if needed.
    history_messages: List[Dict[str, str]] = session_store.get_history_as_messages(
        session_id
    )

    # ------------------------------------------------------------------
    # 2) Sanitize user text using safety.SanitizedTextResult
    # ------------------------------------------------------------------
    raw_text = chat_req.user_text or ""

    try:
        res = safety.sanitize_user_text(raw_text)
    except Exception:  # noqa: BLE001
        logger.exception("sanitize_user_text failed; falling back to raw_text.")
        # Fallback behaviour: treat raw_text as already clean
        from app.core.safety import SanitizedTextResult  # local import to avoid cycle

        res = SanitizedTextResult(
            original=str(raw_text),
            sanitized=str(raw_text),
            truncated=False,
            too_short=(not str(raw_text).strip()),
        )

    clean_user_text = res.sanitized

    # ------------------------------------------------------------------
    # 3) Deterministic intent classification
    # ------------------------------------------------------------------
    intent: str = classify_intent(clean_user_text)

    # ------------------------------------------------------------------
    # 4) Resolve nav_goal using location model (if nav-related)
    # ------------------------------------------------------------------
    canonical_nav_goal: Optional[str] = None
    rough_goal: Optional[str] = None

    if is_nav_intent(intent):
        # Rough phrase from the sentence (e.g. "A201", "info desk")
        rough_goal = extract_nav_goal(clean_user_text)
        # Map that phrase to canonical location if possible
        if rough_goal:
            canonical_nav_goal = resolve_nav_goal(rough_goal)
        # As extra fallback, try using the whole text
        if canonical_nav_goal is None:
            canonical_nav_goal = resolve_nav_goal(clean_user_text)

    # ------------------------------------------------------------------
    # 5) Load live telemetry + locations summary
    # ------------------------------------------------------------------
    nav_state_meta = _load_nav_state_for_meta()
    robot_status_meta = _load_robot_status_for_meta()
    locations_summary = _build_locations_summary()

    # ------------------------------------------------------------------
    # 6) Attach live web context and other meta
    # ------------------------------------------------------------------
    # Start from client meta and enrich with live tools.
    base_meta = dict(chat_req.meta or {})
    # Ensure session_id is always present in meta for downstream logging/prompts.
    base_meta["session_id"] = session_id

    meta_with_live = _attach_live_context(clean_user_text, base_meta)

    # Attach telemetry + locations + canonical goal
    meta_with_live["nav_state"] = nav_state_meta
    meta_with_live["robot_status"] = robot_status_meta
    if locations_summary:
        meta_with_live["locations_summary"] = locations_summary
    if canonical_nav_goal:
        meta_with_live["canonical_nav_goal"] = canonical_nav_goal

    # Telemetry flags so prompts can see if data is real or missing.
    meta_with_live["telemetry_flags"] = {
        "has_nav_state": bool(nav_state_meta),
        "has_robot_status": bool(robot_status_meta),
    }

    # Attach recent conversation history as optional context.
    # generate.py can decide how to use this (or ignore it).
    if history_messages:
        meta_with_live["conversation_history"] = history_messages

    # Debug-only: show just the keys so logs are not spammy.
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[Robot Savo PIPELINE] META keys sent to LLM: %s",
            list(meta_with_live.keys()),
        )

    # ------------------------------------------------------------------
    # 7) Build a new ChatRequest with enriched meta and cleaned text
    # ------------------------------------------------------------------
    enriched_request = ChatRequest(
        user_text=clean_user_text,
        source=chat_req.source,
        language=chat_req.language,
        session_id=session_id,
        meta=meta_with_live,
    )

    # ------------------------------------------------------------------
    # 8) Call Tier1/Tier2/Tier3 via generate_reply_text(...)
    # ------------------------------------------------------------------
    nav_goal_hint = canonical_nav_goal or rough_goal
    result: ModelCallResult = generate_reply_text(
        request=enriched_request,
        intent=intent,
        nav_goal_guess=nav_goal_hint,
    )

    model_output: str = result.text
    logger.info(
        "[Robot Savo PIPELINE] used_tier=%s backend=%r",
        result.used_tier,
        result.raw,
    )

    # ------------------------------------------------------------------
    # 9) Extract final JSON block from model output
    # ------------------------------------------------------------------
    parsed: Dict[str, Any] = {}
    json_obj = _extract_final_json_block(model_output)
    if json_obj is not None:
        parsed = json_obj
    else:
        logger.warning("No valid JSON block found in model output; using fallback.")

    # ------------------------------------------------------------------
    # 10) Combine deterministic intent/nav_goal with model JSON
    # ------------------------------------------------------------------
    reply_text = parsed.get("reply_text") or parsed.get("reply") or ""

    model_intent = parsed.get("intent")
    model_nav_goal = parsed.get("nav_goal")

    # Intent: deterministic classifier wins, log if model disagrees
    final_intent = intent
    if model_intent and model_intent != intent:
        logger.info(
            "Model intent (%s) differs from classified intent (%s); "
            "keeping classifier result.",
            model_intent,
            intent,
        )

    # nav_goal: prefer canonical; otherwise allow model's suggestion
    final_nav_goal: Optional[str] = canonical_nav_goal or model_nav_goal

    # Only keep nav_goal when the intent is navigation-related.
    # For CHATBOT / STATUS (pure explanation) we force nav_goal=None so that
    # Pi does not accidentally think a new navigation target was requested.
    if not is_nav_intent(final_intent):
        final_nav_goal = None

    # ------------------------------------------------------------------
    # 11) Clamp reply length, update session state, and build ChatResponse
    # ------------------------------------------------------------------
    # clamp_reply_text() already uses settings.max_reply_chars
    reply_text = safety.clamp_reply_text(reply_text)

    # Update runtime session store (best-effort; failures should not break reply).
    try:
        session_store.update_from_interaction(
            session_id=session_id,
            user_text=clean_user_text,
            assistant_text=reply_text,
            intent=final_intent,
            nav_goal=final_nav_goal,
        )
    except Exception:  # noqa: BLE001
        logger.exception(
            "[Robot Savo PIPELINE] Failed to update session store for %s",
            session_id,
        )

    # ChatResponse.intent expects Enum, but Pydantic will coerce from string.
    response = ChatResponse(
        reply_text=reply_text,
        intent=final_intent,      # e.g. "NAVIGATE"
        nav_goal=final_nav_goal,  # e.g. "A201" or "Info Desk" or None
        session_id=session_id,
        tier_used=result.used_tier,
    )
    return response


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


async def _self_test() -> int:
    """
    Simple self-test for the pipeline.

    This will actually call generate.py (Tier1/Tier2/Tier3), so make sure
    .env and models are configured before running.
    """
    print("Robot Savo — pipeline.py self-test")
    print("----------------------------------")

    dummy_req = ChatRequest(
        user_text="Can you take me to room A201?",
        session_id="pipeline-self-test",
        meta={"client": "self-test"},
    )

    resp = await run_pipeline(dummy_req)
    print("ChatResponse:")
    try:
        print(resp.model_dump(mode="json", exclude_none=True))
    except AttributeError:
        # Pydantic v1 fallback
        print(resp.dict(exclude_none=True))

    print("\nSelf-test OK.")
    return 0


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(_self_test()))
