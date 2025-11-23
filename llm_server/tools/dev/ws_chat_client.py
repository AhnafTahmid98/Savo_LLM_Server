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
- AUTO-RECONNECT when the connection drops (with backoff).
- NEW: If the connection drops after sending a question but before
  receiving the answer, the client will remember that question and
  automatically resend it after reconnect. You do NOT need to type it again.

This client is meant for development / testing on your laptop.
The Pi will normally call the HTTP /chat endpoint instead.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any, Dict, Optional

import websockets
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK,
)

DEFAULT_SERVER = "ws://127.0.0.1:8000/ws/chat"


# ---------------------------------------------------------------------------
# Custom exception to carry a "pending" message across reconnects
# ---------------------------------------------------------------------------


class PendingMessage(Exception):
    """
    Raised when the connection drops while we are waiting for a reply
    to a message that was already sent.

    The `payload` attribute holds the ChatRequest-shaped dict that
    should be resent after reconnect.
    """

    def __init__(self, payload: Dict[str, Any]) -> None:
        super().__init__("Connection lost with a pending message.")
        self.payload = payload


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


async def run_single_session(
    args: argparse.Namespace,
    pending_payload: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Handles one connect→chat→disconnect cycle.

    If `pending_payload` is provided, it will be resent immediately after
    connecting, and we will wait for its reply before entering the normal
    REPL loop.

    If the connection drops AFTER we send a payload but BEFORE we receive
    the reply, we raise PendingMessage(payload) so the outer loop can
    reconnect and resend it.
    """
    print("Type a message and press Enter. Type /quit to exit.\n")
    print(f"[client] server   : {args.server}")
    print(f"[client] source   : {args.source}")
    print(f"[client] session  : {args.session or '-'}")
    print(f"[client] dev_mode : {args.dev_mode}")
    print()

    # IMPORTANT:
    # - ping_interval=None, ping_timeout=None disables client keepalive pings.
    #   This avoids client-side "keepalive ping timeout" if the server is busy.
    async with websockets.connect(
        args.server,
        ping_interval=None,
        ping_timeout=None,
    ) as ws:
        print("Connected.\n")

        # --------------------------------------------------------------
        # 1) If we had a pending message from a previous connection,
        #    resend it immediately and wait for the answer.
        # --------------------------------------------------------------
        if pending_payload is not None:
            print(
                "[client] Re-sending last unanswered message after reconnect...\n"
            )
            # Send the old payload again
            await ws.send(json.dumps(pending_payload))

            try:
                raw = await ws.recv()
            except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK) as exc:
                print(
                    "\nConnection dropped again while waiting for the "
                    "pending reply."
                )
                # Still pending → propagate so outer loop keeps it.
                raise PendingMessage(pending_payload) from exc

            # Got a reply for the pending message: parse and show it.
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                print(f"Raw response (not JSON) for pending message: {raw}")
            else:
                if isinstance(data, dict) and data.get("type") == "error":
                    print(
                        f"Server error (pending message): "
                        f"{data.get('code')} - {data.get('message')}"
                    )
                    details = data.get("details")
                    if details:
                        print(f"  details: {details}")
                    print()
                else:
                    reply_text = data.get("reply_text")
                    intent = data.get("intent")
                    nav_goal = data.get("nav_goal")
                    print("Robot Savo (pending reply):", reply_text)
                    print(f"  intent = {intent}, nav_goal = {nav_goal}\n")

            # Pending payload has now been answered; clear it.
            pending_payload = None

        # --------------------------------------------------------------
        # 2) Normal REPL loop
        # --------------------------------------------------------------
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
            # Mark this payload as "in flight" until we get a reply.
            in_flight_payload: Dict[str, Any] = payload

            # Send ChatRequest
            await ws.send(json.dumps(payload))

            # Wait for ChatResponse (or error frame)
            try:
                raw = await ws.recv()
            except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK) as exc:
                print(
                    "\nConnection dropped while waiting for reply. "
                    "Your last question will be resent after reconnect."
                )
                # Raise PendingMessage so the outer loop can reconnect
                # and resend this exact payload.
                raise PendingMessage(in_flight_payload) from exc

            # Reached here → we got a reply, so nothing is pending.
            in_flight_payload = {}

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
    - If the error was a PendingMessage, we store its payload and resend
      that message first after reconnect.
    - Backoff: 3s, 6s, 9s, ... capped at 30s.
    - Ctrl+C at any time to exit.
    """
    attempt = 0
    base_delay = 3  # seconds
    pending_payload: Optional[Dict[str, Any]] = None

    while True:
        attempt += 1
        try:
            print(f"Connecting to '{args.server}' (attempt {attempt}) ...")
            await run_single_session(args, pending_payload=pending_payload)
            # If run_single_session returns normally (user /quit),
            # we exit the reconnect loop.
            return

        except KeyboardInterrupt:
            print("\nInterrupted. Bye.")
            return

        except PendingMessage as exc:
            # Keep the payload so we can resend it after reconnect.
            pending_payload = exc.payload
            print(
                "\n[client] Connection closed with a pending question. "
                "Will resend it after reconnect."
            )

        except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK) as exc:
            # Generic close without a known pending message; nothing to resend.
            pending_payload = None
            print(f"\nConnection closed: {exc}")

        except OSError as exc:
            pending_payload = None
            print(f"\nConnection error: {exc}")

        except Exception as exc:  # noqa: BLE001
            pending_payload = None
            print(f"\nUnexpected error: {exc}")

        # Auto-reconnect delay
        delay = min(base_delay * attempt, 30)
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
