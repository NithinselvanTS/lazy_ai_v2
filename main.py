import os
import json
import base64
import datetime
from typing import Optional
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from openai import OpenAI
from github import Github
from duckduckgo_search import DDGS

app = FastAPI(title="Productive Laziness AI")

# ---------------------------------------------------------
# Keys & Client Setup (Reads Render Environment Variables)
# ---------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY.strip() if GROQ_API_KEY else "unconfigured_key"
)

gh = Github(GITHUB_TOKEN.strip()) if GITHUB_TOKEN else None

# ---------------------------------------------------------
# Streak Storage
# ---------------------------------------------------------
STREAK_FILE = "streak.json"

def get_streak_stats() -> dict:
    today_str = datetime.date.today().isoformat()
    default_stats = {"last_active": today_str, "streak_days": 1, "interactions": 0}

    if not os.path.exists(STREAK_FILE):
        with open(STREAK_FILE, "w") as f:
            json.dump(default_stats, f)
        return default_stats

    try:
        with open(STREAK_FILE, "r") as f:
            data = json.load(f)
    except Exception:
        data = default_stats

    last_date = datetime.date.fromisoformat(data.get("last_active", today_str))
    today = datetime.date.today()
    diff = (today - last_date).days

    if diff == 1:
        data["streak_days"] += 1
        data["last_active"] = today_str
    elif diff > 1:
        data["streak_days"] = 1
        data["last_active"] = today_str

    data["interactions"] = data.get("interactions", 0) + 1

    with open(STREAK_FILE, "w") as f:
        json.dump(data, f)

    return data

# ---------------------------------------------------------
# Request Models & Tool Definitions
# ---------------------------------------------------------
class ChatReq(BaseModel):
    prompt: Optional[str] = None
    message: Optional[str] = None

class NarrateReq(BaseModel):
    text: str

tools_spec = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for up-to-date facts, current events, documentation, or links.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The exact web search keywords."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "narrate_text",
            "description": "Convert text or quotes into speech audio using ElevenLabs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The exact text or quote to speak."
                    }
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_github_pr",
            "description": "Create a GitHub issue or task tracker entry.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_name": {
                        "type": "string",
                        "description": "The target repository in format 'username/reponame'."
                    },
                    "title": {
                        "type": "string",
                        "description": "Title of the issue."
                    },
                    "body": {
                        "type": "string",
                        "description": "Body details of the issue."
                    }
                },
                "required": ["repo_name", "title", "body"]
            }
        }
    }
]

def perform_web_search(query: str) -> str:
    try:
        results = DDGS().text(query, max_results=4)
        if not results:
            return "No web results found for that query."
        
        formatted = []
        for r in results:
            formatted.append(f"• **[{r.get('title')}]({r.get('href')})**\n{r.get('body')}")
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Search lookup error: {str(e)}"

# ---------------------------------------------------------
# Web Routes & Handlers
# ---------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def get_ui():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return HTMLResponse("<h1>index.html not found.</h1>")

@app.post("/tools/narrate")
def narrate_response(req: NarrateReq):
    voice_id = "JBFqnCBsd6RMkjVDRZzb"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY.strip()
    }
    payload = {
        "text": req.text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
    }

    resp = requests.post(url, json=payload, headers=headers)
    if resp.status_code != 200:
        return f"⚠️ **ElevenLabs Notice:** API Error ({resp.status_code}): {resp.text}"

    audio_b64 = base64.b64encode(resp.content).decode("utf-8")
    return (
        f'🎙️ **ElevenLabs Voice Output:**<br>"{req.text}"<br><br>'
        f'<audio controls autoplay style="width: 100%; margin-top: 8px;">'
        f'<source src="data:audio/mpeg;base64,{audio_b64}" type="audio/mpeg">'
        f'Your browser does not support audio element.'
        f'</audio>'
    )

@app.post("/api/chat")
@app.post("/chat")
def handle_chat(req: ChatReq):
    user_input = req.prompt or req.message or ""

    streak_data = get_streak_stats()
    streak_banner = (
        f"🔥 **Streak: {streak_data['streak_days']} Day(s)** | "
        f"Interactions: {streak_data['interactions']}\n\n"
    )

    system_prompt = (
        "You are Productive Laziness AI, an autonomous developer coprocessor. "
        "Keep responses direct, sharp, and helpful. Use tools when requested."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=tools_spec,
            tool_choice="auto"
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            call = msg.tool_calls[0]
            
            # Tool 1: Web Search
            if call.function.name == "web_search":
                args = json.loads(call.function.arguments)
                search_query = args.get("query", "")
                raw_search_results = perform_web_search(search_query)

                follow_up = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are Productive Laziness AI. Summarize and synthesize the "
                                "provided search results concisely to answer the user's inquiry."
                            )
                        },
                        {
                            "role": "user",
                            "content": f