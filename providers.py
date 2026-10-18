"""Провайдеры ИИ: Claude (официальный SDK) и Mistral AI (REST).

Общий интерфейс `generate(system_prompt, history)` позволяет переключать
провайдеров одной переменной в .env, не трогая остальной код.
"""
from __future__ import annotations

import abc
import asyncio
import logging

import anthropic
import httpx2 as httpx
from anthropic import AsyncAnthropic

from config import Config

log = logging.getLogger("userbot")

# Официальная документация Mistral API: https://mistral.ai/api/
MISTRAL_CHAT_URL = "https://api.mistral.ai/v1/chat/completions"


class AIError(Exception):
    """Генерация ответа не удалась."""


class AIProvider(abc.ABC):
    name: str = "ai"

    @abc.abstractmethod
    async def generate(self, system_prompt: str, history: list[dict[str, str]]) -> str:
        """По системной роли и истории сообщений вернуть текст ответа.

        history — список {"role": "user"|"assistant", "content": str},
        где "user" — собеседник владельца, "assistant" — владелец/ИИ.
        """

    async def close(self) -> None:
        pass


class ClaudeProvider(AIProvider):
    name = "claude"

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._client = AsyncAnthropic(api_key=cfg.anthropic_api_key)

    async def generate(self, system_prompt: str, history: list[dict[str, str]]) -> str:
        kwargs: dict = {
            "model": self._cfg.claude_model,
            "max_tokens": self._cfg.max_tokens,
            # Адаптивное мышление: модель сама решает, когда и сколько думать.
            # Глубина управляется через effort (low — быстрые и дешёвые ответы).
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self._cfg.claude_effort},
            "system": system_prompt,
            "messages": history,
        }

        try:
            if self._cfg.use_fallbacks:
                # Серверный fallback: при отказе модели по политике
                # безопасности запрос автоматически повторится на запасной модели.
                response = await self._client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                    **kwargs,
                )
            else:
                response = await self._client.messages.create(**kwargs)
        except anthropic.APIStatusError as exc:
            raise AIError(f"Ошибка API Anthropic ({exc.status_code}): {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise AIError("Нет связи с API Anthropic. Проверьте интернет.") from exc

        if response.stop_reason == "refusal":
            log.warning("Claude отклонил запрос (refusal), request_id=%s", response._request_id)
            raise AIError("Модель отклонила этот запрос по правилам безопасности.")

        text = "".join(block.text for block in response.content if block.type == "text").strip()
        if not text:
            raise AIError("Модель вернула пустой ответ.")
        return text

    async def close(self) -> None:
        await self._client.close()


class MistralProvider(AIProvider):
    name = "mistral"

    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg
        self._client = httpx.AsyncClient(
            timeout=60.0,
            headers={"Authorization": f"Bearer {cfg.mistral_api_key}"},
        )

    async def generate(self, system_prompt: str, history: list[dict[str, str]]) -> str:
        payload = {
            "model": self._cfg.mistral_model,
            "messages": [{"role": "system", "content": system_prompt}, *history],
            "max_tokens": self._cfg.max_tokens,
            "temperature": self._cfg.mistral_temperature,
        }

        try:
            response = await self._client.post(MISTRAL_CHAT_URL, json=payload)
        except Exception as exc:  # транспортные ошибки httpx2
            raise AIError(f"Нет связи с API Mistral: {exc}") from exc

        if response.status_code != 200:
            raise AIError(f"Mistral API вернул {response.status_code}: {response.text[:300]}")

        try:
            data = response.json()
            text = data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, ValueError) as exc:
            raise AIError("Неожиданный формат ответа Mistral API.") from exc

        if not text:
            raise AIError("Mistral вернул пустой ответ.")
        return text

    async def close(self) -> None:
        await self._client.aclose()


class EchoProvider(AIProvider):
    """Заглушка для проверки без API-ключей (ECHO_MODE=1)."""

    name = "echo"

    async def generate(self, system_prompt: str, history: list[dict[str, str]]) -> str:
        await asyncio.sleep(0.05)
        last_user = next(
            (m["content"] for m in reversed(history) if m["role"] == "user"), ""
        )
        return f"[echo] {last_user}"


def build_provider(cfg: Config, *, echo: bool = False) -> AIProvider:
    if echo:
        return EchoProvider()
    if cfg.provider == "claude":
        return ClaudeProvider(cfg)
    if cfg.provider == "mistral":
        return MistralProvider(cfg)
    raise ValueError(f"Неизвестный провайдер: {cfg.provider}")
