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
# Layer 1: Deterministic Enrichment (OMDb)
# ---------------------------------------------------------
def fetch_omdb_movie(title: str) -> dict:
    if not OMDB_API_KEY:
        return {}
    try:
        url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
        resp = requests.get(url, timeout=4).json()
        if resp.get("Response") == "True":
            return {
                "title": resp.get("Title"),
                "year": resp.get("Year"),
                "runtime": resp.get("Runtime"),
                "imdb": resp.get("imdbRating"),
                "director": resp.get("Director"),
                "plot": resp.get("Plot")
            }
    except Exception:
        pass
    return {}

# ---------------------------------------------------------
# Layer 2: Live Web Search with Resilient Fallback
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
# System Prompt & Core Directives
# ---------------------------------------------------------
SYSTEM_PROMPT = (
    "You are Productive Laziness AI, an autonomous developer & research coprocessor engineered for extreme user laziness. "
    "Your user wants high-density, finalized answers with zero conversational fluff.\n\n"
    "CRITICAL RULES:\n"
    "1. NEVER answer with conversational questions, pleasantries, or prompts like 'What would you like to know?' or 'Let me know if you need more details'.\n"
    "2. When prompted with any character, franchise, film, or director (e.g., 'batman', 'avengers', 'nolan', 'spider-man'):\n"
    "   - Do NOT write general history, creator bios, or comic book origins.\n"
    "   - IMMEDIATELY provide a clean Markdown Table of all major theatrical/live-action films with columns:\n"
    "     | # | Film Title | Release Year | Runtime | IMDb Rating | Rotten Tomatoes | Where to Watch (Streaming) |\n"
    "   - Under the table, provide: Recommended Watch Order (Chronological vs Release) and a 2-sentence 'Lazy Verdict' on viewing priorities.\n"
    "3. When prompted with technical tools or keywords (e.g., 'fastapi', 'auth', 'postgres'):\n"
    "   - Provide immediate code diffs or Mermaid.js architecture diagrams without introductory fluff."
)

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
            "description": "Search the live web for facts, documentation, releases, streaming details, or news.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The exact search terms."
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
# HTTP Routes
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

    # Autonomous Expansion for 1-3 Word Prompts: Enforce screen media dossier
    lower_words = raw_input.lower().split()
    is_tool_cmd = any(lower_words[0].startswith(x) for x in ["search", "lookup", "narrate", "create"]) if lower_words else False

    if len(lower_words) <= 3 and not is_tool_cmd:
        user_prompt = (
            f"Generate the definitive Productive Laziness Media & Franchise Dossier for: '{raw_input}'.\n\n"
            f"MANDATORY FORMAT:\n"
            f"1. Do NOT write comic origins, character trivia, or introductions.\n"
            f"2. Output a clean Markdown Table of all major theatrical/live-action films with columns:\n"
            f"   | # | Film Title | Release Year | Runtime | IMDb Rating | Rotten Tomatoes | Where to Watch (Streaming) |\n"
            f"3. Recommended Watch Order.\n"
            f"4. A 2-sentence 'Lazy Verdict' on which are essential and which to skip.\n"
            f"If '{raw_input}' is a software term, provide the architecture diagram, code boilerplate, and key concepts."
        )
    else:
        user_prompt = raw_input

    # Optional deterministic OMDb lookup
    movie_info = fetch_omdb_movie(raw_input)
    movie_addon = ""
    if movie_info:
        movie_addon = (
            f"\n[Verified Metadata: Title: {movie_info.get('title')}, "
            f"Year: {movie_info.get('year')}, Runtime: {movie_info.get('runtime')}, "
            f"IMDb: {movie_info.get('imdb')}/10]"
        )

    # Conversation history injection
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if req.history:
        for turn in req.history[-6:]:
            if turn.get("role") in ["user", "assistant"] and turn.get("content"):
                messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": f"{user_prompt}{movie_addon}"})

    # Direct search prefix handling
    if lower_words and lower_words[0] in ["search", "lookup"]:
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
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": followup_user}
                ]
            )
            ans = summary.choices[0].message.content or "Search complete."
        except Exception:
            ans = search_res or "Search completed."

        out = f"{streak_banner}🔍 **Web Search Results for:** *{clean_query}*\n\n{ans}"
        return {"reply": out, "response": out, "message": out, "text": out}

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

                context_block = (
                    f"Live Search Results:\n{s_res}"
                    if s_res
                    else "(No search snippets found. Provide complete table, runtimes, ratings, and streaming sources from knowledge)."
                )

                synth = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"{user_prompt}\n\n{context_block}"}
                    ]
                )
                final_text = synth.choices[0].message.content or "No response could be compiled."
                out = f"{streak_banner}{final_text}"
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
        err_str = str(exc)
        if "failed_generation" in err_str:
            match = re.search(r'"query":\s*"([^"]+)"', err_str)
            recovered_query = match.group(1) if match else raw_input
            s_res = perform_web_search(recovered_query)

            try:
                fallback_comp = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"{user_prompt}\n\nLive Search Results:\n{s_res}"}
                    ]
                )
                ans_text = fallback_comp.choices[0].message.content or s_res
            except Exception:
                ans_text = s_res or "Completed."

            out = f"{streak_banner}{ans_text}"
            return {"reply": out, "response": out, "message": out, "text": out}

        err_msg = f"{streak_banner}⚠️ An error occurred: {err_str}"
        return {"reply": err_msg, "response": err_msg, "message": err_msg, "text": err_msg}