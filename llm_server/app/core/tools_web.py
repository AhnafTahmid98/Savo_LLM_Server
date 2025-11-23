# app/core/tools_web.py
# -*- coding: utf-8 -*-
"""
Robot Savo LLM Server — Web Tools
---------------------------------
Small helper functions for **live external data** that the LLM can use:
- Weather (Open-Meteo)
- Local time (worldtimeapi.org with safe fallback)
- Crypto prices (CoinGecko; limited, safe allow-list of coins/fiats)

This module is used by:
- app/core/pipeline.py  (via _attach_live_context)

Design:
- Each helper is pure: input params -> small dict or None.
- Any network/JSON problems raise ToolsWebError OR return None (for
  non-critical data) so the robot can still answer safely.
- Crypto helpers use explicit allow-lists (BTC/ETH/DOGE/LINK, EUR/USD)
  so the LLM cannot "invent" unknown symbols or random coins.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)


class ToolsWebError(Exception):
    """Raised when a live web tool fails in a way we care about."""


# ---------------------------------------------------------------------------
# Crypto symbol / fiat allow-lists
# ---------------------------------------------------------------------------

# Map short symbols and aliases -> CoinGecko IDs
COIN_ID_MAP: Dict[str, str] = {
    # Bitcoin
    "btc": "bitcoin",
    "xbt": "bitcoin",
    "bitcoin": "bitcoin",
    # Ethereum
    "eth": "ethereum",
    "ethereum": "ethereum",
    # Dogecoin
    "doge": "dogecoin",
    "dogecoin": "dogecoin",
    # Chainlink
    "link": "chainlink",
    "chainlink": "chainlink",
}

# Map various fiat notations -> CoinGecko vs_currency codes
FIAT_MAP: Dict[str, str] = {
    "eur": "eur",
    "€": "eur",
    "euro": "eur",
    "usd": "usd",
    "$": "usd",
    "dollar": "usd",
}


def _resolve_coin_id(symbol: str) -> Optional[str]:
    """
    Resolve a user-facing symbol/alias (e.g. 'btc', 'Bitcoin') to
    a CoinGecko coin id (e.g. 'bitcoin').

    Returns None if the symbol is not in the allow-list.
    """
    key = symbol.strip().lower()
    coin_id = COIN_ID_MAP.get(key)
    if coin_id is None:
        logger.warning("Unknown coin symbol requested: %r", symbol)
    return coin_id


def _resolve_fiat_code(code: str) -> Optional[str]:
    """
    Resolve a user-facing fiat notation (e.g. 'eur', '€', 'usd', '$')
    to a CoinGecko vs_currency code ('eur', 'usd').

    Returns None if the fiat is not in the allow-list.
    """
    key = code.strip().lower()
    fiat = FIAT_MAP.get(key)
    if fiat is None:
        logger.warning("Unknown fiat currency requested: %r", code)
    return fiat


# ---------------------------------------------------------------------------
# Internal HTTP helper
# ---------------------------------------------------------------------------

def _safe_get_json(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    timeout: float = 5.0,
) -> Dict[str, Any]:
    """
    Tiny wrapper around requests.get that:
    - logs errors
    - raises ToolsWebError if HTTP/JSON parsing fails.
    """
    try:
        resp = requests.get(url, params=params, timeout=timeout)
    except requests.RequestException as exc:
        msg = f"HTTP request failed for '{url}': {exc}"
        logger.warning(msg)
        raise ToolsWebError(msg) from exc

    if resp.status_code != 200:
        preview = resp.text[:200].replace("\n", " ")
        msg = f"HTTP {resp.status_code} for '{url}': {preview}"
        logger.warning(msg)
        raise ToolsWebError(msg)

    try:
        return resp.json()
    except ValueError as exc:
        msg = f"Non-JSON response from '{url}'"
        logger.warning(msg)
        raise ToolsWebError(msg) from exc


# ---------------------------------------------------------------------------
# Weather (Open-Meteo)
# ---------------------------------------------------------------------------

def get_weather_current(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """
    Get simple current weather using Open-Meteo.

    Returns a dict like:
        {
            "time": "...",
            "temperature_2m": -5.8,
            "weathercode": 3,
            "windspeed_10m": 13.3,
            "winddirection_10m": 323,
            "is_day": 0,
        }
    or None if it fails (so the robot can still answer without weather).
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": (
            "temperature_2m,weathercode,"
            "windspeed_10m,winddirection_10m,is_day"
        ),
        "timezone": "auto",
    }

    try:
        data = _safe_get_json(url, params=params, timeout=5.0)
    except ToolsWebError:
        return None

    current = data.get("current")
    if not isinstance(current, dict):
        logger.warning("Open-Meteo JSON missing 'current' block: %r", data)
        return None

    # We just return the "current" dict as-is; pipeline will pass it as
    # META.live_context.weather.
    return current


# ---------------------------------------------------------------------------
# Local time (worldtimeapi.org with fallback)
# ---------------------------------------------------------------------------

def get_local_time(timezone: str = "Europe/Helsinki") -> Dict[str, Any]:
    """
    Get local time using worldtimeapi.org.

    If the HTTPS call fails for any reason (SSL, network, etc.) we fall
    back to the system clock and still return a small dict:

        {
            "datetime": "2025-11-22T01:20:53+02:00",
            "timezone": "EET",
        }
    """
    url = f"https://worldtimeapi.org/api/timezone/{timezone}"
    try:
        data = _safe_get_json(url, timeout=5.0)
        # worldtimeapi returns many fields; we only keep what we care about.
        dt = data.get("datetime")
        tz = data.get("abbreviation") or timezone
        if not isinstance(dt, str):
            raise ToolsWebError("worldtimeapi JSON missing 'datetime' string")
        return {"datetime": dt, "timezone": tz}
    except ToolsWebError as exc:
        # Fallback: system time (always available, no network)
        now = _dt.datetime.now(_dt.timezone.utc).astimezone()
        logger.warning(
            "get_local_time: failed (%s), falling back to system time.",
            exc,
        )
        return {
            "datetime": now.isoformat(),
            "timezone": now.tzname() or "local",
        }


# ---------------------------------------------------------------------------
# Crypto price (CoinGecko, with allow-lists)
# ---------------------------------------------------------------------------

def get_crypto_price(
    symbol: str = "bitcoin",
    vs_currency: str = "eur",
    use_aliases: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Get a simple crypto price using the free CoinGecko API.

    Parameters
    ----------
    symbol:
        User-facing coin symbol or CoinGecko id. If `use_aliases` is True,
        we resolve things like "btc"/"Bitcoin" -> "bitcoin" using COIN_ID_MAP.
    vs_currency:
        User-facing fiat code or CoinGecko vs_currency. If `use_aliases`
        is True, we resolve things like "€"/"usd"/"$" -> "eur"/"usd".
    use_aliases:
        If True (default), apply the allow-lists COIN_ID_MAP and FIAT_MAP
        to normalise the requested assets. If either cannot be resolved,
        we log a warning and return None.

    Returns
    -------
    Optional[Dict[str, Any]]
        Example for BTC/EUR:
            {
                "symbol": "bitcoin",
                "vs_currency": "eur",
                "price": 73123.0
            }

        Returns None if:
        - the asset/fiat is not in the allow-list (with use_aliases=True), or
        - the HTTP/JSON request fails, or
        - the JSON shape is unexpected.
    """
    if use_aliases:
        # Normalise coin symbol and fiat to our safe allow-lists.
        norm_symbol = _resolve_coin_id(symbol)
        if norm_symbol is None:
            # Unknown coin: treat as unsupported.
            return None
        norm_vs = _resolve_fiat_code(vs_currency)
        if norm_vs is None:
            # Unknown fiat: treat as unsupported.
            return None
    else:
        # Assume caller already passes valid CoinGecko ids/codes.
        norm_symbol = symbol.strip().lower()
        norm_vs = vs_currency.strip().lower()

    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {
        "ids": norm_symbol,
        "vs_currencies": norm_vs,
    }

    try:
        data = _safe_get_json(url, params=params, timeout=5.0)
    except ToolsWebError:
        return None

    try:
        price = data.get(norm_symbol, {}).get(norm_vs)
    except AttributeError:
        logger.warning("CoinGecko unexpected JSON shape: %r", data)
        return None

    if price is None:
        logger.warning(
            "CoinGecko missing price for %s/%s: %r",
            norm_symbol,
            norm_vs,
            data,
        )
        return None

    try:
        price_f = float(price)
    except (TypeError, ValueError):
        logger.warning(
            "CoinGecko price not numeric for %s/%s: %r",
            norm_symbol,
            norm_vs,
            price,
        )
        return None

    return {
        "symbol": norm_symbol,
        "vs_currency": norm_vs,
        "price": price_f,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Minimal manual self-test.

    Run from project root:

        cd ~/robot_savo_LLM/llm_server
        source .venv/bin/activate
        python3 -m app.core.tools_web
    """
    print("Robot Savo — tools_web.py self-test\n")

    # 1) Weather (Kuopio approx lat/lon)
    w = get_weather_current(62.89, 27.68)
    print("[Weather] current:", w)
    print("-" * 60)

    # 2) Local time (Helsinki timezone)
    t = get_local_time("Europe/Helsinki")
    print("[Time] local:", t)
    print("-" * 60)

    # 3) Crypto (BTC/EUR, ETH/EUR, DOGE/EUR, LINK/EUR) via allow-list
    for sym in ("btc", "eth", "doge", "link"):
        c = get_crypto_price(sym, "eur")
        print(f"[Crypto] {sym.upper()}/EUR:", c["price"] if c else None)

    print("\nSelf-test finished.")
