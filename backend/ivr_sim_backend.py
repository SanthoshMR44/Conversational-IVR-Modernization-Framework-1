# ivr_sim_backend.py
# Conversational IVR simulator backend (FastAPI)
# - Keyword-based intent mapping
# - Optional OpenAI conversational fallback if OPENAI_API_KEY is set

import os
import random
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Optional OpenAI
try:
    import openai
except Exception:
    openai = None

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # optional
if OPENAI_API_KEY and openai:
    openai.api_key = OPENAI_API_KEY

app = FastAPI(title="Conversational IVR Simulator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # keep open for local dev; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Simple Menu / Data Models ----------
class CallStart(BaseModel):
    caller_number: str

class ConversationRequest(BaseModel):
    input_text: str
    call_id: Optional[str] = None

class DTMFInput(BaseModel):
    call_id: str
    digit: str
    current_menu: Optional[str] = "main"

class CallLog(BaseModel):
    call_id: str
    caller_number: str
    start_time: str
    end_time: Optional[str] = None
    menu_path: List[str] = []
    inputs: List[str] = []

# In-memory state (demo)
active_calls = {}
call_history = []

# MENU - reuseable prompts & options
MENU_STRUCTURE = {
    "main": {
        "prompt": "Welcome to Indian Railways Customer Support. You can say 'book ticket', 'train status' or 'refund'.",
        "options": {
            "booking": {"message": "You selected Ticket Booking.", "next": "booking"},
            "train_status": {"message": "You selected Train Status.", "next": "train_status"},
            "refund": {"message": "You selected Refund Enquiry.", "next": "refund"},
            "agent": {"message": "Connecting you to an agent.", "next": None}
        }
    },
    "booking": {
        "prompt": "Booking Menu. Say 'sleeper' or 'ac' or say 'back' to return.",
        "options": {}
    },
    "train_status": {
        "prompt": "Please say or enter your 6-digit PNR number.",
        "options": {}
    },
    "refund": {
        "prompt": "Refund Menu. Say 'cancelled' or 'tatkal' or say 'back' to return.",
        "options": {}
    }
}

# ---------- Utility: simple keyword-based intent detection ----------
def detect_intent(text: str):
    t = (text or "").lower()
    # very basic rules
    if any(k in t for k in ["book", "ticket", "booking", "reserve"]):
        return "booking", "Sure — I can help with booking. Do you want Sleeper or AC class?"
    if any(k in t for k in ["status", "pnr", "flight status", "train status"]):
        return "train_status", "Please tell me your 6-digit PNR number."
    if any(k in t for k in ["refund", "cancel", "tatkal", "money back"]):
        return "refund", "I can help with refunds. Is this for a cancelled train or a tatkal booking?"
    if any(k in t for k in ["agent", "human", "representative", "help"]):
        return "agent", "Okay — transferring you to an agent. Please hold."
    if any(k in t for k in ["back", "main", "menu"]):
        return "main", MENU_STRUCTURE["main"]["prompt"]
    # numeric PNR detection (6-digit)
    digits = "".join(ch for ch in t if ch.isdigit())
    if len(digits) >= 6:
        pnr = digits[:6]
        # mock a response
        return "pnr_lookup", f"PNR {pnr} is CONFIRMED. Train AI101 from Mumbai to Delhi."
    # fallback: try OpenAI if available
    return None, None

# ---------- Optional OpenAI fallback ----------
def openai_fallback(prompt: str):
    if not (OPENAI_API_KEY and openai):
        return None
    try:
        # simple ChatCompletion call - adjust model as available to you
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # replace with available model or "gpt-3.5-turbo"
            messages=[{"role": "system", "content": "You are a helpful IVR assistant for Indian Railways."},
                      {"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.2,
        )
        text = resp.choices[0].message.content.strip()
        return text
    except Exception as e:
        print("OpenAI fallback error:", e)
        return None

# ---------- Endpoints ----------
@app.get("/")
def root():
    return {"status": "IVR Conversational Simulator Running", "active_calls": len(active_calls), "total_calls": len(call_history)}

@app.post("/ivr/start")
def start_call(call: CallStart):
    call_id = f"CALL_{random.randint(100000, 999999)}"
    session = {
        "call_id": call_id,
        "caller_number": call.caller_number,
        "start_time": datetime.utcnow().isoformat(),
        "current_menu": "main",
        "menu_path": ["main"],
        "inputs": [],
    }
    active_calls[call_id] = session
    return {"call_id": call_id, "status": "connected", "prompt": MENU_STRUCTURE["main"]["prompt"]}

@app.post("/ivr/dtmf")
def handle_dtmf(inp: DTMFInput):
    call_id = inp.call_id
    if call_id not in active_calls:
        raise HTTPException(status_code=404, detail="Call not found")
    session = active_calls[call_id]
    digit = inp.digit
    session["inputs"].append(digit)
    # Very simple mapping: digit => intent
    mapping = {
        "1": "booking",
        "2": "train_status",
        "3": "refund",
        "9": "agent",
        "0": "main"
    }
    intent = mapping.get(digit, None)
    if not intent:
        return {"status": "invalid", "message": "Invalid option. Try again.", "current_menu": session["current_menu"]}
    # handle intent
    if intent == "agent":
        session["end_time"] = datetime.utcnow().isoformat()
        call_history.append(session.copy())
        del active_calls[call_id]
        return {"status": "transferred", "message": "Transferring to agent..."}
    if intent == "booking":
        session["current_menu"] = "booking"
        session["menu_path"].append("booking")
        return {"status": "ok", "message": "You selected Booking. Do you want Sleeper or AC?", "current_menu": "booking"}
    if intent == "train_status":
        session["current_menu"] = "train_status"
        session["menu_path"].append("train_status")
        return {"status": "ok", "message": "Please enter your 6-digit PNR.", "current_menu": "train_status"}
    if intent == "refund":
        session["current_menu"] = "refund"
        session["menu_path"].append("refund")
        return {"status": "ok", "message": "Refund menu. Say 'cancelled' or 'tatkal'.", "current_menu": "refund"}
    if intent == "main":
        session["current_menu"] = "main"
        session["menu_path"].append("main")
        return {"status": "ok", "message": MENU_STRUCTURE["main"]["prompt"], "current_menu": "main"}

@app.post("/ivr/conversation")
def conversation(req: ConversationRequest):
    """
    Accepts natural text from frontend (speech-to-text) and returns
    a structured response: { intent, message, next_menu }
    Uses simple keyword rules, with optional OpenAI fallback.
    """
    text = (req.input_text or "").strip()
    call_id = req.call_id
    # Optionally associate to session
    if call_id and call_id in active_calls:
        session = active_calls[call_id]
    else:
        session = None

    # First try rule-based detection
    intent, message = detect_intent(text)
    if intent:
        # apply session transitions for some intents
        if session:
            if intent in ["booking", "train_status", "refund", "main"]:
                session["current_menu"] = intent if intent != "pnr_lookup" else session.get("current_menu", "main")
                session["menu_path"].append(intent)
        return {"intent": intent, "message": message, "call_id": call_id}

    # Fallback: try OpenAI to craft a response if available
    ai_reply = openai_fallback(text)
    if ai_reply:
        return {"intent": "ai_fallback", "message": ai_reply, "call_id": call_id}

    # Final fallback: default help
    help_msg = "Sorry, I didn't get that. You can say 'book ticket', 'train status', or 'refund'."
    return {"intent": "unknown", "message": help_msg, "call_id": call_id}

@app.post("/ivr/end")
def end_call(call_id: str):
    if call_id not in active_calls:
        return {"status": "not_found"}
    session = active_calls[call_id]
    session["end_time"] = datetime.utcnow().isoformat()
    call_history.append(session.copy())
    del active_calls[call_id]
    return {"status": "ended"}
