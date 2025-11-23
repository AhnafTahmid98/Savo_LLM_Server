# Robot Savo LLM Server

FastAPI-based LLM server for **Robot Savo**, the indoor campus guide robot.

This service runs on a PC / laptop and exposes a clean API
for the Raspberry Pi 5:

- `/chat` (HTTP / WebSocket) for navigation + chat.
- `/map/*` for live telemetry snapshots (Nav2 state + robot status).
- `/status/*` for read-only debug views.
- `/ws/*` for persistent chat + telemetry streams.

The server uses a **3-tier LLM chain** (online → local → template) and
tracks **per-session conversation history** so multiple users do not get mixed.

---

## Features

- **Deterministic intent classifier**  
  Classifies each utterance as: `STOP`, `FOLLOW`, `NAVIGATE`, `STATUS`, `CHATBOT`.

- **Navigation-aware**  
  Uses `known_locations.json` + `map_lookup.py` to turn free text
  (e.g. “info desk”, “room A201”) into canonical navigation goals.

- **Real telemetry only**  
  Uses `nav_state.json` and `robot_status.json` pushed by the Pi.  
  If files are missing → treated as “no data” (no fake battery/temperature).

- **Session-aware chat**  
  Uses `runtime_state/sessions.json` to keep history per `session_id`
  so each user can have a multi-turn conversation.

- **3-tier generation chain**  
  1. **Tier1** – Online LLM (OpenRouter)  
  2. **Tier2** – Local LLM via Ollama  
  3. **Tier3** – Offline template fallback

---

## Project Layout (LLM server only)

```txt
llm_server/
  app/
    core/
      config.py           # Settings, .env loader, model config
      intent.py           # Deterministic intent classifier
      generate.py         # Prompt builder + 3-tier generation
      pipeline.py         # High-level /chat pipeline
      map_lookup.py       # Known locations resolver
      safety.py           # Sanitization + length limiting
      tools_web.py        # Weather / time / BTC helpers
      types.py            # ModelCallResult etc.
    models/
      chat_request.py     # ChatRequest schema
      chat_response.py    # ChatResponse schema
      nav_state_model.py  # NavState snapshot model
      robot_status_model.py # RobotStatus snapshot model
      # (possibly location models later)
    runtime_state/
      sessions.py         # SessionStore + sessions.json
    routers/
      chat.py             # POST /chat
      map.py              # /map/navstate, /map/status, /map/known_locations
      status.py           # /status/* debug views
      ws.py               # /ws/chat and /ws/telemetry
    utils/
      __init__.py
      file_io.py          # read_json_safely, write_json_atomic
      logging.py          # setup_logging, get_logger
      timers.py           # simple timing helpers (optional)
    map_data/
      known_locations.json  # canonical destinations (A201, Info Desk, ...)
      nav_state.json        # live Nav2 state (from Pi)
      robot_status.json     # live status (from Pi)
    prompts/
      system_prompt.txt
      style_guidelines.txt
      navigation_prompt.txt
      chatbot_prompt.txt
      status_prompt.txt
    providers/
      tier1_online.py     # OpenRouter provider
      tier2_local.py      # Ollama provider
      tier3_pi.py         # Template fallback
    main.py               # FastAPI app factory + routes
  .env
  requirements.txt
  pyproject.toml
  README.md
