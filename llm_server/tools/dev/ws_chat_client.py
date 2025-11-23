#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot Savo — Dev WebSocket Chat Client (/ws/chat)
-------------------------------------------------
Interactive console tool for talking to the LLM server over WebSocket.

Features:
- Simple REPL: you type, Robot Savo answers.
- Sends ChatRequest-shaped JSON to /ws/chat.
- Reads ChatResponse-shaped JSON back.
- Adds optional meta: session_id, dev_mode.
- AUTO-RECONNECT when the connection drops (fixed small delay).

This client is meant for development / testing on your laptop.
The Pi will normally call the HTTP /chat endpoint instead.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any, Dict

import websockets
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK,
)

DEFAULT_SERVER = "ws://127.0.0.1:8000/ws/chat"


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robot Savo — Dev WebSocket Chat Client (/ws/chat)",
    )
    parser.add_argument(
        "--server",
        type=str,
        default=DEFAULT_SERVER,
        help=f"WebSocket server URL (default: {DEFAULT_SERVER})",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Optional session_id to include in meta (e.g. 'dev-console-01').",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="keyboard",
        choices=["keyboard", "mic", "system", "test"],
        help=(
            "ChatRequest.source value. Must match ChatRequest enum "
            "(keyboard|mic|system|test). Default: keyboard."
        ),
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language hint for ChatRequest (default: en).",
    )
    parser.add_argument(
        "--no-meta",
        action="store_true",
        help="Do not send any meta field (ignore session/dev_mode).",
    )
    parser.add_argument(
        "--dev-mode",
        dest="dev_mode",
        action="store_true",
        default=True,
        help=(
            "Mark this client as dev_mode so the LLM knows it is a test console "
            "(default: enabled)."
        ),
    )
    parser.add_argument(
        "--no-dev-mode",
        dest="dev_mode",
        action="store_false",
        help="Disable dev_mode flag in meta.",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=3.0,
        help="Fixed delay (seconds) before reconnect after a drop (default: 3.0).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Payload helper
# ---------------------------------------------------------------------------


def build_payload(text: str, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Build a minimal ChatRequest-shaped dict.

    Only user_text is required by the server. source/language/meta are
    optional but useful for debugging and routing.
    """
    payload: Dict[str, Any] = {
        "user_text": text,
    }

    if args.language:
        payload["language"] = args.language

    if args.source:
        # MUST be one of: 'keyboard', 'mic', 'system', 'test'
        payload["source"] = args.source

    if not args.no_meta:
        meta: Dict[str, Any] = {}

        if args.session:
            meta["session_id"] = args.session

        if args.dev_mode:
            # Important: tells the prompts this is console / dev mode,
            # not the real Pi driving in a hallway.
            meta["dev_mode"] = True

        if meta:
            payload["meta"] = meta

    return payload


# ---------------------------------------------------------------------------
# Core chat loop (for one connection)
# ---------------------------------------------------------------------------


async def run_single_session(args: argparse.Namespace) -> None:
    """
    Handles one connect→chat→disconnect cycle.

    Called inside an outer auto-reconnect loop.
    """
    print("Type a message and press Enter. Type /quit to exit.\n")
    print(f"[client] server   : {args.server}")
    print(f"[client] source   : {args.source}")
    print(f"[client] session  : {args.session or '-'}")
    print(f"[client] dev_mode : {args.dev_mode}")
    print()

    # IMPORTANT:
    # - ping_interval=None, ping_timeout=None disables client keepalive pings.
    #   This avoids client-side "keepalive ping timeout" when the server is busy.
    async with websockets.connect(
        args.server,
        ping_interval=None,
        ping_timeout=None,
    ) as ws:
        print("Connected.\n")

        while True:
            try:
                text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                raise KeyboardInterrupt

            if not text:
                continue
            if text.lower() in {"/quit", "/exit"}:
                print("Bye.")
                raise KeyboardInterrupt

            payload = build_payload(text, args)

            # Send ChatRequest
            await ws.send(json.dumps(payload))

            # Wait for ChatResponse (or error frame)
            try:
                raw = await ws.recv()
            except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK) as exc:
                print(f"\nConnection dropped while waiting for reply: {exc}")
                # Re-raise so outer loop can auto-reconnect.
                raise

            # Try to parse JSON
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                print(f"Raw response (not JSON): {raw}")
                continue

            # If server sent a structured error frame
            if isinstance(data, dict) and data.get("type") == "error":
                print(f"Server error: {data.get('code')} - {data.get('message')}")
                details = data.get("details")
                if details:
                    print(f"  details: {details}")
                print()
                continue

            # Normal ChatResponse
            reply_text = data.get("reply_text")
            intent = data.get("intent")
            nav_goal = data.get("nav_goal")

            print(f"\nRobot Savo: {reply_text}")
            print(f"  intent = {intent}, nav_goal = {nav_goal}\n")


# ---------------------------------------------------------------------------
# Auto-reconnect wrapper
# ---------------------------------------------------------------------------


async def run_with_reconnect(args: argparse.Namespace) -> None:
    """
    Outer loop that auto-reconnects when connection fails.

    Behaviour:
    - Tries to connect and run_single_session().
    - On error/disconnect, waits a bit and retries.
    - Fixed delay (default 3s) between reconnect attempts.
    - Ctrl+C at any time to exit.
    """
    attempt = 0
    delay = max(float(args.reconnect_delay), 0.5)  # guard: at least 0.5s

    while True:
        attempt += 1
        try:
            print(f"Connecting to '{args.server}' (attempt {attempt}) ...")
            await run_single_session(args)
            # If run_single_session returns normally (user /quit),
            # we exit the reconnect loop.
            return
        except KeyboardInterrupt:
            print("\nInterrupted. Bye.")
            return
        except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK) as exc:
            print(f"\nConnection closed: {exc}")
        except OSError as exc:
            print(f"\nConnection error: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"\nUnexpected error: {exc}")

        # Auto-reconnect with fixed delay
        print(f"Reconnecting in {delay} seconds... (Ctrl+C to stop)")
        try:
            await asyncio.sleep(delay)
        except KeyboardInterrupt:
            print("\nInterrupted during backoff. Bye.")
            return


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run_with_reconnect(args))
    except KeyboardInterrupt:
        print("\nBye.")
        sys.exit(0)


if __name__ == "__main__":
    main()
