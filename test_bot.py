"""Смоук-тесты логики без сети и API-ключей.

Запуск: python -m pytest -q
"""
from __future__ import annotations

import asyncio

import pytest

from config import Config, ConfigError
from main import OwnerOverride
from memory import MemoryStore
from providers import EchoProvider, build_provider


def _cfg(**overrides) -> Config:
    base = dict(
        api_id=1, api_hash="h", phone="+1", session_file="s.session",
        allowed_chats=["@a"],
        provider="claude", anthropic_api_key="k",
        claude_model="claude-opus-5-5", claude_effort="low", use_fallbacks=True,
        mistral_api_key="m", mistral_model="mistral-small-latest", mistral_temperature=0.7,
        system_prompt="role", history_limit=20, max_tokens=4096, min_reply_delay=0.0,
    )
    base.update(overrides)
    return Config(**base)


# --- OwnerOverride (перехват диалога владельцем) ---

def test_owner_override_mute_unmute():
    o = OwnerOverride()
    assert not o.is_muted(5)
    o.mute(5)
    assert o.is_muted(5)
    o.unmute(5)
    assert not o.is_muted(5)


def test_owner_override_isolated_per_chat():
    o = OwnerOverride()
    o.mute(1)
    assert o.is_muted(1)
    assert not o.is_muted(2)


# --- MemoryStore ---

def test_memory_append_and_trim(tmp_path):
    async def run():
        store = MemoryStore(limit=4, path=tmp_path / "h.json")
        for i in range(6):
            await store.append(1, "user", f"m{i}")
        return [t["content"] for t in await store.get_history(1)]

    assert asyncio.run(run()) == ["m2", "m3", "m4", "m5"]


def test_memory_history_starts_with_user(tmp_path):
    async def run():
        store = MemoryStore(limit=10, path=tmp_path / "h.json")
        await store.append(1, "assistant", "leading")
        await store.append(1, "user", "hello")
        return await store.get_history(1)

    assert asyncio.run(run())[0]["role"] == "user"


def test_memory_persists(tmp_path):
    path = tmp_path / "h.json"

    async def write():
        s = MemoryStore(limit=10, path=path)
        await s.append(7, "user", "запомни")
        await s.append(7, "assistant", "ок")

    async def read():
        return await MemoryStore(limit=10, path=path).get_history(7)

    asyncio.run(write())
    assert [t["content"] for t in asyncio.run(read())] == ["запомни", "ок"]


# --- providers ---

def test_build_provider_names():
    assert build_provider(_cfg(provider="claude")).name == "claude"
    assert build_provider(_cfg(provider="mistral")).name == "mistral"
    assert build_provider(_cfg(), echo=True).name == "echo"


def test_build_provider_unknown():
    with pytest.raises(ValueError):
        build_provider(_cfg(provider="gpt"))


def test_echo_provider():
    out = asyncio.run(EchoProvider().generate("role", [{"role": "user", "content": "тест"}]))
    assert "тест" in out


# --- config parsing ---

def test_parse_chats_helper():
    from config import _parse_chats
    assert _parse_chats("@a, -100123 , ,@b") == ["@a", "-100123", "@b"]


def test_config_requires_allowed_chats(monkeypatch, tmp_path):
    import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "BASE_DIR", tmp_path)
    (tmp_path / "prompt.txt").write_text("role", encoding="utf-8")
    monkeypatch.setenv("API_ID", "1")
    monkeypatch.setenv("API_HASH", "h")
    monkeypatch.setenv("PHONE", "+1")
    monkeypatch.setenv("ALLOWED_CHATS", "")
    monkeypatch.setenv("PROVIDER", "claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    with pytest.raises(ConfigError, match="ALLOWED_CHATS"):
        cfg_mod.load_config()


def test_config_requires_api_id(monkeypatch, tmp_path):
    import config as cfg_mod
    monkeypatch.setattr(cfg_mod, "BASE_DIR", tmp_path)
    (tmp_path / "prompt.txt").write_text("role", encoding="utf-8")
    monkeypatch.setenv("API_ID", "not-a-number")
    with pytest.raises(ConfigError):
        cfg_mod.load_config()
