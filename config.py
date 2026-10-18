"""Загрузка и валидация настроек юзербота из .env."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


class ConfigError(Exception):
    """Ошибка конфигурации — понятный текст для пользователя."""


def _env(name: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise ConfigError(
            f"Не заполнена обязательная переменная {name} в файле .env "
            f"(см. шаблон .env.example)."
        )
    return value or ""


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"Переменная {name} должна быть целым числом, получено: {raw!r}") from None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _parse_chats(raw: str) -> list[str]:
    """Список чатов из .env: через запятую, @username или числовой id."""
    chats: list[str] = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            chats.append(item)
    return chats


@dataclass(frozen=True)
class Config:
    api_id: int
    api_hash: str
    phone: str
    session_file: str
    allowed_chats: list[str] = field(default_factory=list)

    provider: str = "claude"  # "claude" | "mistral"
    anthropic_api_key: str = ""
    claude_model: str = "claude-opus-5-5"
    claude_effort: str = "low"
    use_fallbacks: bool = True
    mistral_api_key: str = ""
    mistral_model: str = "mistral-small-latest"
    mistral_temperature: float = 0.7

    system_prompt: str = ""
    history_limit: int = 20
    max_tokens: int = 4096

    # Предохранители юзербота
    min_reply_delay: float = 3.0      # «человеческая» пауза перед ответом, сек
    cooldown_after_owner: int = 0     # 0 = ИИ молчит в чате до следующего входящего

    @property
    def model_name(self) -> str:
        return self.claude_model if self.provider == "claude" else self.mistral_model


def load_config() -> Config:
    provider = _env("PROVIDER", "claude").strip().lower()
    if provider not in ("claude", "mistral"):
        raise ConfigError(f"PROVIDER должен быть 'claude' или 'mistral', получено: {provider!r}")

    api_id_raw = _env("API_ID", required=True)
    try:
        api_id = int(api_id_raw)
    except ValueError:
        raise ConfigError(f"API_ID должен быть числом (получить: https://my.telegram.org), получено: {api_id_raw!r}") from None

    prompt_file = BASE_DIR / _env("SYSTEM_PROMPT_FILE", "prompt.txt")
    if not prompt_file.exists():
        raise ConfigError(f"Файл с ролью не найден: {prompt_file}")
    system_prompt = prompt_file.read_text(encoding="utf-8").strip()
    if not system_prompt:
        raise ConfigError(f"Файл {prompt_file.name} пуст — опишите в нём роль.")

    allowed_chats = _parse_chats(_env("ALLOWED_CHATS", ""))
    if not allowed_chats:
        raise ConfigError(
            "ALLOWED_CHATS пуст. Перечислите через запятую @username или id чатов, "
            "в которых ИИ может отвечать от вашего имени. Без белого списка бот не запустится."
        )

    try:
        temperature = float(_env("MISTRAL_TEMPERATURE", "0.7"))
    except ValueError:
        raise ConfigError("MISTRAL_TEMPERATURE должна быть числом, например 0.7") from None

    cfg = Config(
        api_id=api_id,
        api_hash=_env("API_HASH", required=True),
        phone=_env("PHONE", required=True),
        session_file=_env("SESSION_FILE", "userbot.session"),
        allowed_chats=allowed_chats,
        provider=provider,
        anthropic_api_key=_env("ANTHROPIC_API_KEY"),
        claude_model=_env("CLAUDE_MODEL", "claude-opus-5-5"),
        claude_effort=_env("CLAUDE_EFFORT", "low").strip().lower(),
        use_fallbacks=_env_bool("USE_FALLBACKS", True),
        mistral_api_key=_env("MISTRAL_API_KEY"),
        mistral_model=_env("MISTRAL_MODEL", "mistral-small-latest"),
        mistral_temperature=temperature,
        system_prompt=system_prompt,
        history_limit=_env_int("HISTORY_LIMIT", 20),
        max_tokens=_env_int("MAX_TOKENS", 4096),
        min_reply_delay=float(_env("MIN_REPLY_DELAY", "3") or 3),
    )

    if cfg.provider == "claude" and not cfg.anthropic_api_key:
        raise ConfigError("Для PROVIDER=claude укажите ANTHROPIC_API_KEY в .env.")
    if cfg.provider == "mistral" and not cfg.mistral_api_key:
        raise ConfigError("Для PROVIDER=mistral укажите MISTRAL_API_KEY в .env.")
    if cfg.history_limit < 2:
        raise ConfigError("HISTORY_LIMIT должен быть не меньше 2.")

    return cfg
