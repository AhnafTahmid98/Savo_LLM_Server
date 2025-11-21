"""
intent.py
---------
This module decides what the human wants the robot to do, based on the
speech text that came from STT (speech-to-text).

The output is one high-level INTENT string that controls robot behavior:
    - "STOP"     : User wants the robot to stop / freeze / wait.
    - "FOLLOW"   : User wants the robot to follow them.
    - "NAVIGATE" : User wants the robot to guide them to a place.
    - "STATUS"   : User is asking for explanation / status / battery.
    - "CHATBOT"  : Normal talk (default fallback, safe).

"""

from typing import Literal, Optional

# -------------------------------------------------------------------------
# Public types
# -------------------------------------------------------------------------

IntentType = Literal["STOP", "FOLLOW", "NAVIGATE", "STATUS", "CHATBOT"]


# -------------------------------------------------------------------------
# Keyword catalogs (English-only for now)
# Keep them grouped so teammates can edit easily.
# These are substrings, not full-word matches.
# -------------------------------------------------------------------------

STOP_KEYWORDS = [
    # Soft stop
    "stop",
    "stop now",
    "stop please",
    "please stop",
    "can you stop",

    # Freeze / don't move / wait
    "halt",
    "freeze",
    "wait here",
    "wait there",
    "stay here",
    "stay there",
    "stay still",
    "stay right there",

    # Negative motion commands
    "do not move",
    "don't move",
    "do not go",
    "don't go",
    "do not follow",
    "don't follow",
]

FOLLOW_KEYWORDS = [
    # Ask robot to follow the human
    "follow me",
    "follow my",
    "follow us",
    "come with me",
    "come with us",
    "come with",
    "come here",
    "come behind me",
    "come behind us",
    "walk with me",
    "walk behind me",
    "stay with me",
    "stay close to me",
]

NAVIGATE_KEYWORDS = [
    # Ask robot to lead the way
    "take me to",
    "take me",
    "can you take me",
    "can u take me",

    "bring me to",
    "bring me",
    "can you bring me",
    "can u bring me",

    "guide me to",
    "guide me",
    "can you guide",
    "can u guide",

    "show me the way",
    "show me where",
    "show me",

    # Asking for directions / location
    "i need to go",
    "i need go",
    "i want to go",
    "i want go",
    "help me find",
    "help us find",

    # Navigation phrasing
    "go to",
    "go with me to",
    "take us to",
    "take us",

    # Questions like "where is A201?"
    "where is",
    "where's",
    "how do i get to",
    "how can i get to",
]

STATUS_KEYWORDS = [
    # Why robot is stopped / not moving
    "why did you stop",
    "why you stop",
    "why you stopped",
    "why are you stopped",
    "why did u stop",
    "why you not moving",
    "why you not move",
    "why you are not moving",
    "why you are not moving?",

    # Robot condition
    "are you ok",
    "are you okay",
    "are you fine",
    "are you broken",
    "are you damaged",

    # What are you doing?
    "what are you doing",
    "what are you doing?",
    "what are you doing right now",
    "what is happening",
    "what is going on",
    "what is your status",
    "tell me status",
    "status please",

    # Battery / temp / health
    "what is your battery",
    "battery level",
    "battery status",
    "how much battery",
    "how much battery you have",
    "battery percent",
    "battery percentage",

    "temperature",
    "what is your temperature",
    "are you hot",
    "are you overheating",
]


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------

def normalize(text: str) -> str:
    """
    Normalize user text for simpler matching.
    - lowercases
    - trims
    - collapses multiple spaces to single space
    We do NOT remove punctuation fully because sometimes we care about "to".
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")

    lowered = text.lower().strip()
    parts = lowered.split()  # split() auto-removes extra spaces/newlines
    return " ".join(parts)


def _contains_any(text: str, keywords: list[str]) -> bool:
    """
    Returns True if any keyword in `keywords` exists as a substring
    inside the normalized text.
    We use substring and not exact word match. Why?
    - "can you stop now please" still contains "stop".
    - "can u take me to a201" still contains "take me to".
    """
    for k in keywords:
        if k in text:
            return True
    return False


def _extract_after_keywords(text: str, triggers: list[str]) -> Optional[str]:
    """
    Try to extract a navigation target (room / location name) from phrases
    like:
        "take me to A201"
        "can you bring me to info desk"
        "go to room a201"
        "i want to go info desk"

    We do this with a dumb heuristic: look for each trigger,
    take the text that comes AFTER it, and return the first chunk
    that looks like a "goal phrase".

    This is not AI, it's just string slicing. Good enough for early demo.
    Later Tier1/Tier2/Tier3 pipeline will do smarter extraction.

    Returns:
        "A201", "info desk", etc.
        or None if not found.
    """
    t = text  # already normalized outside
    for trig in triggers:
        idx = t.find(trig)
        if idx != -1:
            # Cut everything after trigger
            after = t[idx + len(trig):].strip()

            # If user says "take me to the info desk please"
            #   after = "the info desk please"
            # Heuristic: stop at polite words like "please"
            stop_words = ["please", "now", "thanks", "thank you"]
            for stopper in stop_words:
                cut_idx = after.find(stopper)
                if cut_idx != -1:
                    after = after[:cut_idx].strip()

            # Also remove leading filler words like "the", "room", "to", "our"
            fillers = ["the", "room", "to", "our", "my", "me", "us"]
            # We'll repeatedly trim fillers from the start
            tokens = after.split()
            while tokens and tokens[0] in fillers:
                tokens.pop(0)

            if not tokens:
                continue

            # Now rebuild string from remaining tokens
            candidate = " ".join(tokens)

            # Safety: we don't want to return huge paragraphs
            # If it's longer than, say, 4 words, it's likely junk.
            # ex: "the big classroom near stairs" (5 words) might be real,
            # but for MVP we'll cut it to first 4 words to stay clean.
            candidate_tokens = candidate.split()
            if len(candidate_tokens) > 4:
                candidate = " ".join(candidate_tokens[:4])

            # Final small cleanup, strip punctuation on edges
            candidate = candidate.strip(",.?!:;")

            if candidate:
                return candidate

    return None


# -------------------------------------------------------------------------
# Intent classification
# -------------------------------------------------------------------------

def classify_intent(user_text: str) -> IntentType:
    """
    Classify the user's request into one of the intent labels.

    SAFETY:
    - If input is empty or nonsense, we do NOT assume an action like FOLLOW.
      We fall back to "CHATBOT".
    - We never guess navigation if not obvious.

    Priority logic is enforced in order.
    """

    if not isinstance(user_text, str):
        raise TypeError("Input must be a string")

    if not user_text.strip():
        # Empty string -> just talk like a chatbot
        return "CHATBOT"

    t = normalize(user_text)

    # 1. STOP (highest priority)
    if _contains_any(t, STOP_KEYWORDS):
        return "STOP"

    # 2. FOLLOW
    if _contains_any(t, FOLLOW_KEYWORDS):
        return "FOLLOW"

    # 3. NAVIGATE
    if _contains_any(t, NAVIGATE_KEYWORDS):
        return "NAVIGATE"

    # 4. STATUS
    if _contains_any(t, STATUS_KEYWORDS):
        return "STATUS"

    # 5. CHATBOT (fallback, safe, non-moving)
    return "CHATBOT"


def is_nav_intent(intent: IntentType) -> bool:
    """
    True if this intent requires physical navigation / body motion.
    NOTE:
    - FOLLOW and NAVIGATE require path planning / cmd_vel.
    - STOP is not "movement", but it *affects* motion urgently.
    """
    return intent in ("FOLLOW", "NAVIGATE")


def extract_nav_goal(user_text: str) -> Optional[str]:
    """
    Try to pull out a target location (like 'A201', 'info desk') from the
    user's sentence.

    This is only *attempted* for navigation-like phrases.
    If nothing confident is found, return None.

    WARNING:
    - This is heuristic. It can be wrong.
    - The caller (generate.py / pipeline.py) MUST still validate the goal
      against known_locations.json before trusting it.
    """
    if not isinstance(user_text, str):
        return None
    if not user_text.strip():
        return None

    t = normalize(user_text)

    # We try multiple trigger patterns in order of most specific to general.
    # Example:
    #   "take me to a201"
    #   "can you take me to the info desk please"
    #   "go to room a201"
    trigger_phrases = [
        "take me to",
        "can you take me to",
        "can u take me to",
        "bring me to",
        "can you bring me to",
        "guide me to",
        "can you guide me to",
        "go to",
        "i need to go to",
        "i want to go to",
        "i need to go",
        "i want to go",
        "help me find",
        "help us find",
        "show me the way to",
        "show me the way",
        "show me where",
        "where is",
        "where's",
        "how do i get to",
        "how can i get to",
    ]

    goal = _extract_after_keywords(t, trigger_phrases)

    # We return whatever we got (ex: "a201", "info desk")
    # Caller can uppercase, titlecase, validate, etc.
    return goal


def classify_intent_debug(user_text: str) -> dict:
    """
    Debug helper for logging / unit tests.
    This does not drive the robot directly.
    """
    intent = classify_intent(user_text)
    goal = extract_nav_goal(user_text) if intent == "NAVIGATE" else None

    return {
        "user_text": user_text,
        "normalized": normalize(user_text) if isinstance(user_text, str) else "",
        "intent": intent,
        "nav_goal_guess": goal,
        "is_nav_intent": is_nav_intent(intent),
        "priority_rule": "STOP > FOLLOW > NAVIGATE > STATUS > CHATBOT",
    }

if __name__ == "__main__":
    """
    Minimal manual self-test.

    Run from project root:

        cd ~/robot_savo_LLM/llm_server
        source .venv/bin/activate
        python3 -m app.core.intent
    """
    samples = [
        "stop please",
        "can you follow me",
        "can you take me to A201",
        "i want to go to the info desk please",
        "why did you stop",
        "hello robot savo, how are you",
        "",
    ]

    print("Robot Savo — intent.py self-test\n")
    for text in samples:
        info = classify_intent_debug(text)
        print(f"Text     : {text!r}")
        print(f"Intent   : {info['intent']}")
        print(f"Nav goal : {info['nav_goal_guess']!r}")
        print(f"is_nav   : {info['is_nav_intent']}")
        print(f"Norm     : {info['normalized']!r}")
        print("-" * 60)
