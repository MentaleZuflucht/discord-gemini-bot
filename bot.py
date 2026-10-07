"""Discord slash-command bot. It never reads message content and never logs or stores prompts."""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime

import discord
from discord import app_commands
from dotenv import load_dotenv

from gemini_client import GeminiClient, GeminiError, chunk_discord_message, parse_api_keys
from logging_config import setup_logging

logger = logging.getLogger("bot")

MAX_PROMPT_LENGTH = 2000
EMBED_DESCRIPTION_LIMIT = 4096
EMBED_COLOR = 0x7C6AF7
GEMINI_ICON_URL = (
    "https://www.gstatic.com/lamda/images/gemini_sparkle_4g_512_lt_f94943af3be039176192d.png"
)


class PromptBot(discord.Client):
    def __init__(self, gemini: GeminiClient, guild_id: int | None) -> None:
        super().__init__(
            intents=discord.Intents.none(),
            chunk_guilds_at_startup=False,
            member_cache_flags=discord.MemberCacheFlags.none(),
        )
        self.gemini = gemini
        self.guild_id = guild_id
        self.tree = app_commands.CommandTree(self)
        self.tree.error(self.on_app_command_error)

    async def setup_hook(self) -> None:
        prompt.name = "prompt"
        self.tree.add_command(prompt)
        if self.guild_id is None:
            await self.tree.sync()
            logger.info("Synced global slash commands.")
            return
        guild = discord.Object(id=self.guild_id)
        self.tree.copy_global_to(guild=guild)
        await self.tree.sync(guild=guild)
        logger.info("Synced slash commands to guild %s.", self.guild_id)

    async def close(self) -> None:
        await self.gemini.aclose()
        await super().close()

    async def on_ready(self) -> None:
        logger.info("Logged in as %s.", self.user)

    async def on_app_command_error(
        self,
        interaction: discord.Interaction,
        error: app_commands.AppCommandError,
    ) -> None:
        command = interaction.command.qualified_name if interaction.command else "unknown"
        logger.error("Slash command /%s failed.", command, exc_info=error)
        await _send(interaction, "Something went wrong while handling that command.")


bot_client: PromptBot | None = None


@app_commands.command(description="Ask the bot something. It replies once and forgets the message.")
@app_commands.describe(message="The message to answer.")
async def prompt(
    interaction: discord.Interaction,
    message: app_commands.Range[str, 1, MAX_PROMPT_LENGTH],
) -> None:
    cleaned = message.strip()
    # Only metadata here: the prompt text itself must never reach the logs.
    logger.debug("/prompt from guild %s, %s characters.", interaction.guild_id, len(cleaned))
    if not cleaned:
        await interaction.response.send_message("Send a message with the command.", ephemeral=True)
        return

    await interaction.response.defer()
    client = bot_client
    if client is None:
        await interaction.followup.send("The bot is not ready yet.")
        return

    started = time.perf_counter()
    try:
        reply = await client.gemini.generate(cleaned)
    except GeminiError as exc:
        logger.warning("No reply for /prompt: %s", exc)
        await interaction.followup.send(str(exc))
        return
    except Exception:
        logger.exception("Gemini request failed.")
        await interaction.followup.send("The model could not be reached. Try again in a moment.")
        return

    elapsed = time.perf_counter() - started
    parts = compose_embed_descriptions(reply.text, cleaned)
    footer = format_reply_footer(reply.model, elapsed, datetime.now())
    logger.debug("Reply from %s in %.1fs, sent as %s embed(s).", reply.model, elapsed, len(parts))
    last = len(parts) - 1
    for index, part in enumerate(parts):
        embed = discord.Embed(description=part, color=EMBED_COLOR)
        if index == last:
            embed.set_footer(text=footer, icon_url=GEMINI_ICON_URL)
        await interaction.followup.send(embed=embed)


def format_reply_footer(model: str, elapsed_seconds: float, when: datetime) -> str:
    return f"{model} • {elapsed_seconds:.1f}s • {when.strftime('%d/%m/%Y %H:%M')}"


def fade_prompt(prompt: str) -> str:
    """Discord subtext: smaller and dimmer, the closest thing to faded text."""
    lines = prompt.splitlines() or [prompt]
    return "\n".join(f"-# {line}" for line in lines)


def compose_embed_descriptions(
    reply: str,
    prompt: str,
    limit: int = EMBED_DESCRIPTION_LIMIT,
) -> list[str]:
    faded = fade_prompt(prompt)
    tail = f"\n\n{faded}"
    if len(tail) >= limit:
        faded = faded[: limit - 2].rstrip() + "…"
        tail = f"\n\n{faded}"
    reply_limit = max(1, limit - len(tail))
    parts = chunk_discord_message(reply, limit=reply_limit)
    parts[-1] = f"{parts[-1]}{tail}"
    return parts


async def _send(interaction: discord.Interaction, content: str) -> None:
    if interaction.response.is_done():
        await interaction.followup.send(content)
    else:
        await interaction.response.send_message(content)


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        logger.critical("Missing %s. Copy .env.example to .env and fill it in.", name)
        raise SystemExit(1)
    return value


def main() -> None:
    global bot_client
    load_dotenv()
    setup_logging()
    token = _required_env("DISCORD_TOKEN")
    api_keys = parse_api_keys(
        os.environ.get("GEMINI_API_KEY", ""),
        os.environ.get("GEMINI_API_KEYS", ""),
    )
    if not api_keys:
        logger.critical("Missing GEMINI_API_KEY or GEMINI_API_KEYS. Copy .env.example to .env and fill it in.")
        raise SystemExit(1)
    logger.info("Loaded %s Gemini API key(s).", len(api_keys))
    raw_guild = os.environ.get("DISCORD_GUILD_ID", "").strip()
    guild_id = int(raw_guild) if raw_guild else None

    bot_client = PromptBot(GeminiClient(api_keys), guild_id)
    try:
        # log_handler=None keeps discord.py from adding its own handler; its logs go through ours.
        bot_client.run(token, log_handler=None)
    except Exception:
        logger.critical("The bot stopped because of an unhandled error.", exc_info=True)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
