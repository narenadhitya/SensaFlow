"""SensaFlow -> Telegram fault alerts.

After `python pipeline.py --predict <file>` scores new data, a digest of the alert episodes at or above
TELEGRAM_MIN_SEVERITY (default HIGH) is sent to your Telegram chat. Setup steps: README -> Telegram alerts.

    python notify.py --get-chat-id            # after you have sent any message to your bot
    python notify.py --test                   # send a test message
    python notify.py --preview outputs/custom # print the digest for an output folder without sending
    python notify.py --send outputs/custom    # send the digest for an output folder now

Settings (environment or the git-ignored .env):
    TELEGRAM_BOT_TOKEN       from @BotFather (required)
    TELEGRAM_CHAT_ID         your chat / group id (required; see --get-chat-id)
    TELEGRAM_MIN_SEVERITY    HIGH (default) or MEDIUM
    TELEGRAM_MAX_ALERTS      episodes listed per message (default 5)
    TELEGRAM_ALL_CLEAR       1 = also send a short message when a run has no alerts (default off)
    SENSAFLOW_DASHBOARD_URL  optional link added to each message, e.g. http://localhost:8501
"""
import argparse
import html
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
API = "https://api.telegram.org/bot{token}/{method}"


# ------------------------------------------------------------------------------------------- settings

def setting(name: str, default: str | None = None) -> str | None:
    """Environment first, then the project's .env file."""
    if os.environ.get(name):
        return os.environ[name].strip()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == name and value.strip():
                return value.strip().strip('"').strip("'")
    return default


def configured() -> bool:
    return bool(setting("TELEGRAM_BOT_TOKEN") and setting("TELEGRAM_CHAT_ID"))


class TelegramError(RuntimeError):
    pass


def api(method: str, payload: dict | None = None) -> dict:
    token = setting("TELEGRAM_BOT_TOKEN")
    if not token:
        raise TelegramError("TELEGRAM_BOT_TOKEN is not set (add it to .env)")
    req = urllib.request.Request(API.format(token=token, method=method),
                                 data=json.dumps(payload or {}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        try:
            detail = json.load(e).get("description", "")
        except Exception:
            detail = ""
        hint = {401: "token rejected - copy it again from @BotFather",
                400: "check TELEGRAM_CHAT_ID, and send /start to your bot first",
                403: "the bot can't message this chat - send /start to it, or add it to the group",
                404: "token format is wrong - copy it again from @BotFather"}.get(e.code, "")
        raise TelegramError(f"Telegram API {e.code}: {detail} {('(' + hint + ')') if hint else ''}".strip()) from None
    except urllib.error.URLError as e:
        raise TelegramError(f"cannot reach api.telegram.org: {e.reason}") from None
    if not data.get("ok"):
        raise TelegramError(data.get("description", "unknown Telegram error"))
    return data["result"]


def send_message(text: str) -> None:
    if not setting("TELEGRAM_BOT_TOKEN"):
        raise TelegramError("TELEGRAM_BOT_TOKEN is not set (add it to .env)")
    chat_id = setting("TELEGRAM_CHAT_ID")
    if not chat_id:
        raise TelegramError("TELEGRAM_CHAT_ID is not set (run: python notify.py --get-chat-id)")
    api("sendMessage", {"chat_id": chat_id, "text": text[:4096], "parse_mode": "HTML",
                        "disable_web_page_preview": True})


# ------------------------------------------------------------------------------------------- digest

def _analyst():
    sys.path.insert(0, str(ROOT / "dashboard"))
    import assistant   # same episode grouping and ranking the dashboard's analyst uses
    return assistant


def _short(drivers: str) -> str:
    """'strain = 12,252 Nm·min (above normal 362–9,004)' -> 'strain = 12,252 Nm·min ▲ above normal'."""
    return re.sub(r"\((above|below|within) normal [^)]*\)",
                  lambda m: {"above": "▲ above normal", "below": "▼ below normal", "within": "in range"}[m.group(1)],
                  drivers)


def digest(out_dir: Path) -> str | None:
    """Telegram HTML message for the alerts in an output folder, or None when there is nothing to report."""
    out_dir = Path(out_dir)
    assistant = _analyst()
    run = assistant.load_run(out_dir)
    eps = assistant.run_episodes(run)
    info = run["info"]
    min_sev = (setting("TELEGRAM_MIN_SEVERITY", "HIGH") or "HIGH").upper()
    keep = [e for e in eps if e["severity"] == "HIGH" or min_sev == "MEDIUM"]
    name = html.escape(str(info.get("input", out_dir.name)))
    model = html.escape(str(info.get("model", "?")))
    link = setting("SENSAFLOW_DASHBOARD_URL")
    footer = f'\n<a href="{html.escape(link)}">Open the SensaFlow dashboard</a>' if link else ""

    if not keep:
        if setting("TELEGRAM_ALL_CLEAR") == "1":
            return f"✅ <b>SensaFlow</b> · all clear\n{name} · {info.get('rows', '?')} rows · no {min_sev} alerts{footer}"
        return None

    n_high = sum(e["severity"] == "HIGH" for e in keep)
    title = (f"🚨 <b>SensaFlow: {len(keep)} fault alert{'s' if len(keep) != 1 else ''}</b>"
             + (f" ({n_high} HIGH)" if min_sev == "MEDIUM" else ""))
    lines = [title, f"{name} · {model} · {info.get('rows', '?')} rows", ""]
    limit = int(setting("TELEGRAM_MAX_ALERTS", "5") or 5)
    profile = info.get("profile", "ai4i")   # SKAB: longest events first; AI4I: most confident first
    for e in sorted(keep, key=lambda e: assistant._rank(e, profile))[:limit]:
        when = e["start"] if e["rows"] == 1 else f"{e['start']} → {e['end']} ({e['rows']} rows)"
        icon = "🔴" if e["severity"] == "HIGH" else "🟠"
        lines.append(f"{icon} <b>{html.escape(e['type'])}</b> · p={e['peak']:.2f}\n"
                     f"   {html.escape(when)}")
        if e["drivers"]:
            lines.append(f"   <i>{html.escape(_short(e['drivers']))}</i>")
    if len(keep) > limit:
        lines.append(f"\n…and {len(keep) - limit} more alert episode{'s' if len(keep) - limit != 1 else ''}.")
    return "\n".join(lines) + footer


def notify_run(out_dir: Path) -> str:
    """Send the digest for a finished run. Never raises: returns a one-line status for the pipeline log."""
    if not configured():
        return "telegram: not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID) - skipped"
    try:
        text = digest(out_dir)
        if text is None:
            return "telegram: no alerts at the notification severity - nothing sent"
        send_message(text)
        return "telegram: alert digest sent"
    except Exception as e:   # notifications must never break scoring
        return f"telegram: NOT sent - {e}"


# ------------------------------------------------------------------------------------------- CLI

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="SensaFlow Telegram alerts")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--get-chat-id", action="store_true", help="list chats that recently messaged your bot")
    g.add_argument("--test", action="store_true", help="send a test message")
    g.add_argument("--preview", metavar="OUTPUT_DIR", help="print the digest for an output folder (no sending)")
    g.add_argument("--send", metavar="OUTPUT_DIR", help="send the digest for an output folder")
    args = ap.parse_args()
    try:
        if args.get_chat_id:
            updates = api("getUpdates")
            chats = {}
            for u in updates:
                msg = u.get("message") or u.get("channel_post") or u.get("my_chat_member") or {}
                chat = msg.get("chat")
                if chat:
                    chats[chat["id"]] = chat.get("title") or chat.get("username") or chat.get("first_name") or ""
            if not chats:
                print("No messages found. Open your bot in Telegram, press Start (or send any message), then rerun.")
            for cid, label in chats.items():
                print(f"TELEGRAM_CHAT_ID={cid}    # {label}")
        elif args.test:
            send_message("✅ <b>SensaFlow</b> is connected. Fault alerts will arrive in this chat.")
            print("Test message sent - check Telegram.")
        elif args.preview:
            text = digest(Path(args.preview))
            print(text or "(no alerts at the notification severity - nothing would be sent)")
        else:
            print(notify_run(Path(args.send)))
    except TelegramError as e:
        sys.exit(f"Telegram error: {e}")


if __name__ == "__main__":
    main()
