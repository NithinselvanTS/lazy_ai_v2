import os
import json
import re
import base64
import datetime
from typing import Optional, List
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from openai import OpenAI
from github import Github
from duckduckgo_search import DDGS

app = FastAPI(title="Productive Laziness AI")

# ---------------------------------------------------------
# Keys & Client Configuration
# ---------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "").strip()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY.strip() if GROQ_API_KEY else "unconfigured_key"
)

gh = Github(GITHUB_TOKEN.strip()) if GITHUB_TOKEN else None

# ---------------------------------------------------------
# Persistent Streak Tracker
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
# Live Web Search (DuckDuckGo Fallback)
# ---------------------------------------------------------
def perform_web_search(query: str) -> str:
    try:
        ddgs = DDGS()
        clean_q = re.sub(r"\b(202[4-9]|today)\b", "", query, flags=re.IGNORECASE).strip()
        search_term = clean_q if clean_q else query

        results = ddgs.text(search_term, max_results=4)
        if not results:
            results = ddgs.news(search_term, max_results=4)

        if not results:
            return ""

        formatted = []
        for r in results:
            title = r.get("title") or "Article"
            link = r.get("href") or r.get("url") or "#"
            body = r.get("body") or r.get("excerpt") or ""
            formatted.append(f"• **[{title}]({link})**\n{body}")
        return "\n\n".join(formatted)
    except Exception:
        return ""

# ---------------------------------------------------------
# Schemas & Tool Definitions
# ---------------------------------------------------------
class ChatReq(BaseModel):
    prompt: Optional[str] = None
    message: Optional[str] = None
    history: Optional[List[dict]] = []

class NarrateReq(BaseModel):
    text: str

tools_spec = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the live web for facts, documentation, releases, or news.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The exact search query."
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
            "description": "Convert text into spoken audio via ElevenLabs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to narrate."
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
                        "description": "Target repository formatted as 'owner/repo'."
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

# ---------------------------------------------------------
# Web Routes
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
        f'Your browser does not support audio playback.'
        f'</audio>'
    )

@app.post("/api/chat")
@app.post("/chat")
def handle_chat(req: ChatReq):
    raw_input = (req.prompt or req.message or "").strip()

    streak_data = get_streak_stats()
    streak_banner = (
        f"🔥 **Streak: {streak_data['streak_days']} Day(s)** | "
        f"Interactions: {streak_data['interactions']}\n\n"
    )

    clean_lower = raw_input.lower().strip()
    words = clean_lower.split()
    is_tool_cmd = any(words[0].startswith(x) for x in ["search", "lookup", "narrate", "create"]) if words else False

    # Direct search prefix handling
    if words and words[0] in ["search", "lookup"]:
        clean_query = re.sub(r"^(search the web for|search for|search|lookup)\s+", "", raw_input, flags=re.IGNORECASE).strip()
        search_res = perform_web_search(clean_query)

        followup_user = (
            f"User query: {raw_input}\n\nLive Search Results:\n{search_res}"
            if search_res
            else f"User query: {raw_input}\n\n(No search snippets returned. Supply all runtimes, ratings, and streaming availability directly from knowledge)."
        )

        try:
            summary = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": "You are Productive Laziness AI. Answer concisely and synthesize results directly."},
                    {"role": "user", "content": followup_user}
                ]
            )
            ans = summary.choices[0].message.content or "Search complete."
        except Exception:
            ans = search_res or "Search completed."

        out = f"{streak_banner}🔍 **Web Search Results for:** *{clean_query}*\n\n{ans}"
        return {"reply": out, "response": out, "message": out, "text": out}

    # Autonomous Expansion for 1-3 Word Queries
    if len(words) <= 3 and not is_tool_cmd:
        dossier_system = (
            "You are Productive Laziness AI, an autonomous coprocessor engineered for extreme user efficiency. "
            "The user gave a short keyword. NEVER output comic origins, creator biographies, or conversational questions.\n\n"
            "MANDATORY FORMAT:\n"
            "1. Output a Markdown table listing all major theatrical feature films/releases with exact columns:\n"
            "   | # | Title | Release Year | Runtime | IMDb Rating | Rotten Tomatoes | Where to Watch (Streaming) |\n"
            "2. Under the table, provide: Recommended Watch Order.\n"
            "3. End with a 2-sentence 'Lazy Verdict' on which entries are essential and which to skip.\n"
            "(If the term is technical like 'fastapi', provide an architecture breakdown and code boilerplate instead)."
        )

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": dossier_system},
                {"role": "user", "content": f"Generate the Productive Laziness Dossier for: '{raw_input}'"}
            ]
        )
        final_text = response.choices[0].message.content or "Completed."
        out = f"{streak_banner}{final_text}"
        return {"reply": out, "response": out, "message": out, "text": out}

    # Standard conversational multi-turn processing with tools
    standard_system = (
        "You are Productive Laziness AI, an autonomous developer & research coprocessor. "
        "Provide direct, high-density answers without conversational filler. Use tools when requested."
    )

    messages = [{"role": "system", "content": standard_system}]
    if req.history:
        for turn in req.history[-6:]:
            if turn.get("role") in ["user", "assistant"] and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": raw_input})

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=tools_spec,
            tool_choice="auto",
            extra_body={"reasoning_format": "hidden"}
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            call = msg.tool_calls[0]

            if call.function.name == "web_search":
                args = json.loads(call.function.arguments)
                s_query = args.get("query", raw_input)
                s_res = perform_web_search(s_query)

                context_block = f"Live Search Results:\n{s_res}" if s_res else "(Rely on internal knowledge base)."
                synth = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {"role": "system", "content": standard_system},
                        {"role": "user", "content": f"{raw_input}\n\n{context_block}"}
                    ]
                )
                final_text = synth.choices[0].message.content or "Search complete."
                out = f"{streak_banner}🔍 **Web Search Results for:** *{s_query}*\n\n{final_text}"
                return {"reply": out, "response": out, "message": out, "text": out}

            elif call.function.name == "narrate_text":
                args = json.loads(call.function.arguments)
                tts_result = narrate_response(NarrateReq(text=args.get("text", "")))
                out = streak_banner + tts_result
                return {"reply": out, "response": out, "message": out, "text": out}

            elif call.function.name == "create_github_pr":
                args = json.loads(call.function.arguments)
                try:
                    if not gh:
                        raise ValueError("GitHub token is not configured.")
                    repo = gh.get_repo(args.get("repo_name"))
                    issue = repo.create_issue(title=args.get("title"), body=args.get("body"))
                    pr_result = f"🚀 **GitHub Task Created:** [{issue.title}]({issue.html_url})"
                except Exception as gh_err:
                    pr_result = f"⚠️ GitHub Error: {str(gh_err)}"
                out = streak_banner + pr_result
                return {"reply": out, "response": out, "message": out, "text": out}

        bot_text = msg.content or "Processing complete."
        out = streak_banner + bot_text
        return {"reply": out, "response": out, "message": out, "text": out}

    except Exception as exc:
        err_msg = f"{streak_banner}⚠️ An error occurred: {str(exc)}"
        return {"reply": err_msg, "response": err_msg, "message": err_msg, "text": err_msg}