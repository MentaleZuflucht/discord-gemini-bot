"""Offline tests. Gemini and Discord are faked, so no keys or network are needed.

Model names come from PREFERRED_FREE_MODELS instead of being hard-coded, so
updating the free-model list does not break these tests.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import discord
import httpx
import pytest

import bot
from bot import PromptBot, compose_embed_descriptions
from gemini_client import (
    PREFERRED_FREE_MODELS,
    QUOTA_MESSAGE,
    GeminiClient,
    GeminiError,
    GeminiReply,
    chunk_discord_message,
    is_text_chat_model,
    parse_api_keys,
    rank_models,
)

BEST, SECOND = PREFERRED_FREE_MODELS[0], PREFERRED_FREE_MODELS[1]


def model_payload(name: str) -> dict:
    return {"name": f"models/{name}", "supportedGenerationMethods": ["generateContent"]}


def listed(*names: str) -> dict:
    return {"models": [model_payload(name) for name in names]}


def reply(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]}


def quota_error() -> httpx.Response:
    error = {"message": "Resource exhausted", "status": "RESOURCE_EXHAUSTED"}
    return httpx.Response(429, json={"error": error})


def fake_gemini(handle_generate, list_body: object = None):
    """Fake API that lists every free model (or returns `list_body`) and answers
    generate calls with `handle_generate(key, model)`.

    Returns the (key, model) generate calls it saw, and the transport to plug into httpx.
    """
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        key = request.headers["x-goog-api-key"]
        if request.url.path.endswith("/models"):
            body = listed(*PREFERRED_FREE_MODELS) if list_body is None else list_body
            return httpx.Response(200, json=body)
        model = request.url.path.rsplit("/", 1)[-1].removesuffix(":generateContent")
        calls.append((key, model))
        return handle_generate(key, model)

    return calls, httpx.MockTransport(handler)


def run_generate(transport: httpx.MockTransport, keys: list[str], prompt: str = "hi"):
    async def go():
        async with httpx.AsyncClient(transport=transport) as http:
            return await GeminiClient(keys, client=http).generate(prompt)

    return asyncio.run(go())


# --- Free-model list -------------------------------------------------------


def test_preferred_models_are_unique_gemini_chat_models():
    assert len(set(PREFERRED_FREE_MODELS)) == len(PREFERRED_FREE_MODELS)
    for name in PREFERRED_FREE_MODELS:
        assert is_text_chat_model(model_payload(name)), name


def test_rank_models_keeps_preferred_order_and_drops_unknown_models():
    payloads = listed("some-paid-model", SECOND, f"{BEST}-image", BEST)["models"]
    assert rank_models(payloads) == (BEST, SECOND)


def test_rank_models_falls_back_to_free_list_when_none_are_listed():
    assert rank_models(listed("gemini-paid-only")["models"]) == PREFERRED_FREE_MODELS


@pytest.mark.parametrize(
    "name", ["gemini-x-image", "gemini-x-tts", "gemini-embedding-001", "gemma-3"]
)
def test_non_chat_models_are_filtered(name):
    assert not is_text_chat_model(model_payload(name))


# --- Gemini client ---------------------------------------------------------


def test_generate_uses_the_best_model():
    calls, transport = fake_gemini(lambda key, model: httpx.Response(200, json=reply("pong")))
    result = run_generate(transport, ["key-a"])
    assert (result.text, result.model) == ("pong", BEST)
    assert calls == [("key-a", BEST)]


def test_generate_skips_thought_parts():
    parts = [{"text": "thinking...", "thought": True}, {"text": "answer"}]
    body = {"candidates": [{"content": {"parts": parts}}]}
    _, transport = fake_gemini(lambda key, model: httpx.Response(200, json=body))
    assert run_generate(transport, ["key-a"]).text == "answer"


def test_generate_falls_back_to_next_model_on_quota():
    def handle(key, model):
        return quota_error() if model == BEST else httpx.Response(200, json=reply("from second"))

    _, transport = fake_gemini(handle)
    assert run_generate(transport, ["key-a"]).model == SECOND


def test_generate_moves_to_next_key_when_a_key_is_rejected():
    def handle(key, model):
        if key == "bad-key":
            return httpx.Response(400, json={"error": {"message": "API key not valid."}})
        return httpx.Response(200, json=reply("ok"))

    calls, transport = fake_gemini(handle)
    assert run_generate(transport, ["bad-key", "good-key"]).text == "ok"
    assert calls[-1] == ("good-key", BEST)


def test_generate_reports_quota_when_every_model_is_exhausted():
    _, transport = fake_gemini(lambda key, model: quota_error())
    with pytest.raises(GeminiError, match=QUOTA_MESSAGE):
        run_generate(transport, ["key-a", "key-b"])


def test_generate_falls_back_to_next_model_on_empty_reply():
    # The model spent its whole token budget thinking and wrote no answer.
    thinking_only = {
        "candidates": [
            {"finishReason": "MAX_TOKENS", "content": {"parts": [{"text": "...", "thought": True}]}}
        ]
    }

    def handle(key, model):
        body = thinking_only if model == BEST else reply("from second")
        return httpx.Response(200, json=body)

    _, transport = fake_gemini(handle)
    assert run_generate(transport, ["key-a"]).model == SECOND


def test_generate_reports_empty_reply_when_every_model_is_empty():
    _, transport = fake_gemini(lambda key, model: httpx.Response(200, json={"candidates": []}))
    with pytest.raises(GeminiError, match="no reply"):
        run_generate(transport, ["key-a"])


def test_generate_uses_built_in_order_when_model_list_is_malformed():
    calls, transport = fake_gemini(
        lambda key, model: httpx.Response(200, json=reply("ok")), list_body=[]
    )
    assert run_generate(transport, ["key-a"]).model == BEST
    assert calls == [("key-a", BEST)]


def test_generate_handles_error_body_wrapped_in_a_list():
    def handle(key, model):
        if model == BEST:
            return httpx.Response(500, json=[{"error": {"message": "Internal error"}}])
        return httpx.Response(200, json=reply("ok"))

    _, transport = fake_gemini(handle)
    assert run_generate(transport, ["key-a"]).model == SECOND


def test_generate_reports_a_blocked_prompt():
    body = {"promptFeedback": {"blockReason": "SAFETY"}}
    calls, transport = fake_gemini(lambda key, model: httpx.Response(200, json=body))
    with pytest.raises(GeminiError, match="declined"):
        run_generate(transport, ["key-a"])
    assert len(calls) == 1  # another model would refuse too, so none is tried


# --- Helpers ---------------------------------------------------------------


def test_parse_api_keys_splits_strips_and_dedupes():
    assert parse_api_keys('"a", b\nc,a', "d") == ("a", "b", "c", "d")


def test_chunk_discord_message_respects_limit():
    text = " ".join(["word"] * 1000)
    parts = chunk_discord_message(text, limit=200)
    assert all(len(part) <= 200 for part in parts)
    assert " ".join(parts) == text


def test_embeds_fit_discord_limit_and_end_with_the_prompt():
    parts = compose_embed_descriptions("x " * 5000, "what is love", limit=4096)
    assert len(parts) > 1
    assert all(len(part) <= 4096 for part in parts)
    assert parts[-1].endswith("-# what is love")


# --- Discord ---------------------------------------------------------------


class FakeInteraction:
    guild_id = 1

    def __init__(self):
        self.command = SimpleNamespace(qualified_name="prompt")
        self.deferred = False
        self.sent: list = []  # message text or embeds, in order
        self.response = SimpleNamespace(
            is_done=lambda: self.deferred, defer=self._defer, send_message=self._send
        )
        self.followup = SimpleNamespace(send=self._followup)

    async def _defer(self):
        self.deferred = True

    async def _send(self, content, **kwargs):
        self.sent.append(content)

    async def _followup(self, content=None, *, embed=None):
        self.sent.append(embed if embed is not None else content)


def run_prompt(monkeypatch, generate, message: str = "hello") -> FakeInteraction:
    gemini = SimpleNamespace(generate=generate)
    monkeypatch.setattr(bot, "bot_client", SimpleNamespace(gemini=gemini))
    interaction = FakeInteraction()
    asyncio.run(bot.prompt.callback(interaction, message))
    return interaction


def test_prompt_sends_long_reply_as_embeds_with_footer_on_the_last(monkeypatch):
    async def generate(prompt):
        return GeminiReply("x " * 5000, BEST)

    interaction = run_prompt(monkeypatch, generate)
    embeds = interaction.sent
    assert interaction.deferred
    assert len(embeds) > 1
    assert all(len(embed.description) <= bot.EMBED_DESCRIPTION_LIMIT for embed in embeds)
    assert BEST in embeds[-1].footer.text
    assert embeds[0].footer.text is None


def test_prompt_sends_gemini_errors_as_text(monkeypatch):
    async def generate(prompt):
        raise GeminiError(QUOTA_MESSAGE)

    assert run_prompt(monkeypatch, generate).sent == [QUOTA_MESSAGE]


def test_prompt_rejects_a_blank_message(monkeypatch):
    async def generate(prompt):
        raise AssertionError("Gemini should not be called")

    interaction = run_prompt(monkeypatch, generate, message="   ")
    assert not interaction.deferred
    assert interaction.sent == ["Send a message with the command."]


def test_expired_interaction_is_not_answered():
    expired = discord.NotFound(
        SimpleNamespace(status=404, reason="Not Found"),
        {"code": 10062, "message": "Unknown interaction"},
    )
    error = SimpleNamespace(original=expired)  # stands in for CommandInvokeError
    interaction = FakeInteraction()
    asyncio.run(PromptBot.on_app_command_error(None, interaction, error))
    assert interaction.sent == []


def test_other_errors_get_a_reply():
    interaction = FakeInteraction()
    asyncio.run(PromptBot.on_app_command_error(None, interaction, RuntimeError("boom")))
    assert interaction.sent == ["Something went wrong while handling that command."]
