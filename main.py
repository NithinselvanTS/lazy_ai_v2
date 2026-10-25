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
            else f"User query: {raw_input}\n\n(No search snippets returned. Supply all relevant information directly from knowledge)."
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

    # Autonomous 360° Productive Laziness Dossier (1-3 word queries)
    if len(words) <= 3 and not is_tool_cmd:
        dossier_system = (
            "You are Productive Laziness AI, an autonomous coprocessor engineered for extreme user efficiency. "
            "The user provided a short keyword. Do NOT ask clarifying questions or present options. Deliver an exhaustive, all-in-one dossier.\n\n"
            "MANDATORY STRUCTURE FOR CHARACTERS / HEROES / FRANCHISES:\n"
            "1. **Executive Profile & Identity**: Real name, affiliations, creator origins, and what they do.\n"
            "2. **Powers, Armor & Specialties**: Signature tech, suits, intellect, combat style, and unique abilities.\n"
            "3. **Screen Appearances & Streaming Guide**: A clean Markdown Table listing major cinematic titles with columns:\n"
            "   | # | Title | Release Year | Runtime | IMDb Rating | Rotten Tomatoes | Where to Watch (Streaming) |\n"
            "4. **The Lazy Verdict**: 2-sentence summary of the essential watch/read path and what can be skipped.\n\n"
            "IF THE PROMPT IS TECHNICAL (e.g., 'fastapi', 'postgres', 'docker'):\n"
            "- Provide an architectural overview, Mermaid diagram, key strengths/trade-offs, and code boilerplate."
        )

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": dossier_system},
                {"role": "user", "content": f"Generate the complete 360° Productive Laziness Dossier for: '{raw_input}'"}
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