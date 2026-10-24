import os
import json
from datetime import date, timedelta

STREAK_FILE = "streak_data.json"

def record_activity():
    today = date.today()
    record = {
        "current_streak": 0,
        "last_login": None,
        "longest_streak": 0,
        "total_interactions": 0
    }

    if os.path.exists(STREAK_FILE):
        try:
            with open(STREAK_FILE, "r") as f:
                record = json.load(f)
        except Exception:
            pass

    last_str = record.get("last_login")
    if last_str:
        last_date = date.fromisoformat(last_str)
        if last_date == today:
            pass
        elif last_date == today - timedelta(days=1):
            record["current_streak"] += 1
            record["last_login"] = str(today)
        else:
            record["current_streak"] = 1
            record["last_login"] = str(today)
    else:
        record["current_streak"] = 1
        record["last_login"] = str(today)

    record["total_interactions"] = record.get("total_interactions", 0) + 1

    if record["current_streak"] > record.get("longest_streak", 0):
        record["longest_streak"] = record["current_streak"]

    with open(STREAK_FILE, "w") as f:
        json.dump(record, f, indent=2)

    return record