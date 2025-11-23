# Robot Savo LLM Server

FastAPI-based LLM server for **Robot Savo**, the indoor campus guide robot.

This service runs on a PC / laptop (Ubuntu or macOS) and exposes HTTP + WebSocket
APIs for the Raspberry Pi 5:

- `POST /chat` – main endpoint for navigation + chatbot.
- `POST /map/navstate` – Pi pushes Nav2 state snapshot.
- `POST /map/status` – Pi pushes robot health snapshot.
- `GET  /map/known_locations` – list of canonical destinations.
- `GET  /status/*` – read-only debug views.
- `WS  /ws/chat` – persistent chat.
- `WS  /ws/telemetry` – streaming telemetry.

The server uses a 3-tier LLM chain (online → local → template) and keeps
conversation history per `session_id`.

---

## Quick start

```bash
cd llm_server
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
# or: pip install .
