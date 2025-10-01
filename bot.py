# bot.py
import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()


from telegram import Bot
from telegram.constants import ParseMode
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from engine import Engine, load_config

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")  # your user id or channel/group id

assert BOT_TOKEN, "Set BOT_TOKEN in env"
assert CHAT_ID, "Set CHAT_ID in env"

cfg = load_config()
engine = Engine(cfg)
bot = Bot(BOT_TOKEN)

async def notify_start():
    try:
        await bot.send_message(chat_id=CHAT_ID, text="✅ Signal bot online. Scanning…")
    except Exception as e:
        print("Startup notify error:", e)


DISCLAIMER = (
    "<i>Educational alerts only. Not financial advice. Do your own research.\n"
    "Trading involves risk; past performance is not indicative of future results.</i>"
)

def fmt(sig: dict) -> str:
    t = datetime.fromisoformat(sig["time"]).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"<b>{sig['symbol']}</b>  <code>{sig['timeframe']}</code>  <b>{sig['side']}</b>\n"
        f"Time: {t}\n"
        f"Entry: <b>{sig['entry']:.2f}</b>\n"
        f"SL: <b>{sig['sl']:.2f}</b>\n"
        f"TP1: <b>{sig['tp1']:.2f}</b>  |  TP2: <b>{sig['tp2']:.2f}</b>\n"
        f"Strategy: <code>{sig['meta']['strategy']}</code>\n\n"
        f"{DISCLAIMER}"
    )

async def tick():
    sigs = engine.run_once()
    for s in sigs:
        try:
            await bot.send_message(chat_id=CHAT_ID, text=fmt(s), parse_mode=ParseMode.HTML, disable_web_page_preview=True)
        except Exception as e:
            print("Send error:", e)

async def main():
    await notify_start()
    scheduler = AsyncIOScheduler(timezone="UTC")
    # scan every minute (per timeframe close the dedupe avoids duplicates)
    scheduler.add_job(tick, IntervalTrigger(seconds=60))
    scheduler.start()

    print("Bot started. Press Ctrl+C to stop.")
    while True:
        await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())