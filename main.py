"""Юзербот-автоответчик: ИИ отвечает от ВАШЕГО аккаунта в выбранных чатах.

Работает через Telethon (MTProto). При первом запуске попросит код из
Telegram (и облачный пароль 2FA, если включён) — дальше сессия хранится
в файле *.session.

Запуск:  python main.py
Остановка: Ctrl+C

ВАЖНО: автоматизация личного аккаунта нарушает правила Telegram и может
привести к ограничениям. Белый список чатов (ALLOWED_CHATS) обязателен —
без него бот не запустится. Ответы идут с паузой и только туда, куда вы
разрешили.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from telethon import TelegramClient, events

from config import BASE_DIR, ConfigError, load_config
from memory import MemoryStore
from providers import AIError, AIProvider, build_provider

log = logging.getLogger("userbot")


class OwnerOverride:
    """Помнит, в каких чатах владелец ответил сам — там ИИ замолкает
    до следующего входящего сообщения от собеседника."""

    def __init__(self) -> None:
        self._muted: set[int] = set()

    def mute(self, chat_id: int) -> None:
        self._muted.add(chat_id)

    def unmute(self, chat_id: int) -> None:
        self._muted.discard(chat_id)

    def is_muted(self, chat_id: int) -> bool:
        return chat_id in self._muted


async def resolve_allowed_ids(client: TelegramClient, allowed: list[str]) -> list[int]:
    """Превращает @username / id из .env в числовые id для фильтра Telethon."""
    ids: list[int] = []
    for item in allowed:
        try:
            if item.lstrip("-").isdigit():
                entity = await client.get_entity(int(item))
            else:
                entity = await client.get_entity(item.lstrip("@"))
            ids.append(entity.id)
            log.info("Разрешён чат %s → id=%s", item, entity.id)
        except Exception as exc:  # noqa: BLE001 — один плохой чат не валит запуск
            log.error("Не удалось найти чат %r (%s) — пропускаю.", item, exc)
    return ids


async def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
    )
    logging.getLogger("telethon").setLevel(logging.WARNING)

    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"\n[ошибка настройки] {exc}\n")
        return

    echo_mode = os.getenv("ECHO_MODE", "").strip().lower() in ("1", "true", "yes", "on")
    provider: AIProvider = build_provider(cfg, echo=echo_mode)
    memory = MemoryStore(limit=cfg.history_limit, path=BASE_DIR / "data" / "history.json")
    owner = OwnerOverride()

    session_path = BASE_DIR / cfg.session_file
    client = TelegramClient(str(session_path), cfg.api_id, cfg.api_hash)

    await client.start(phone=cfg.phone)
    me = await client.get_me()
    log.info(
        "Вход выполнен: %s %s (@%s), id=%s",
        me.first_name or "", me.last_name or "", me.username or "—", me.id,
    )

    allowed_ids = await resolve_allowed_ids(client, cfg.allowed_chats)
    if not allowed_ids:
        print("\n[ошибка] Ни один чат из ALLOWED_CHATS не найден. Проверьте имена/id.\n")
        await client.disconnect()
        await provider.close()
        return

    # Фильтр Telethon: только входящие сообщения в разрешённых чатах.
    # Владельца (me.id) исключаем явно — на свои сообщения не отвечаем.
    chat_locks: dict[int, asyncio.Lock] = {}

    def lock_for(chat_id: int) -> asyncio.Lock:
        return chat_locks.setdefault(chat_id, asyncio.Lock())

    @client.on(events.NewMessage(chats=allowed_ids, incoming=True))
    async def on_incoming(event: events.NewMessage.Event) -> None:
        msg = event.message
        chat_id = msg.chat_id

        # Никогда не отвечаем себе.
        if msg.sender_id == me.id:
            return
        # Собеседник написал — снимаем глушение, ИИ снова может отвечать.
        owner.unmute(chat_id)

        text = (msg.raw_text or "").strip()
        if not text:
            return  # стикер/фото/голосовое — пропускаем

        async with lock_for(chat_id):
            await memory.append(chat_id, "user", text)
            history = await memory.get_history(chat_id)

            try:
                answer = await provider.generate(cfg.system_prompt, history)
            except AIError as exc:
                log.error("Ошибка генерации в чате %s: %s", chat_id, exc)
                return
            except Exception:  # noqa: BLE001
                log.exception("Неожиданная ошибка генерации")
                return

            # «Человеческая» пауза перед ответом.
            if cfg.min_reply_delay > 0:
                await asyncio.sleep(cfg.min_reply_delay)

            # Если за время генерации/паузы владелец ответил сам — молчим.
            if owner.is_muted(chat_id):
                log.info("Чат %s: владелец ответил сам — пропускаю.", chat_id)
                return

            try:
                sent = await client.send_message(chat_id, answer, reply_to=msg.id)
            except Exception as exc:  # noqa: BLE001
                log.error("Не удалось отправить ответ в чат %s: %s", chat_id, exc)
                return

            await memory.append(chat_id, "assistant", answer)
            log.info("Чат %s: ответ отправлен (%d симв.)", chat_id, len(answer))

    # Исходящие сообщения владельца: глушим ИИ в этом чате.
    @client.on(events.NewMessage(chats=allowed_ids, outgoing=True))
    async def on_outgoing(event: events.NewMessage.Event) -> None:
        chat_id = event.message.chat_id
        owner.mute(chat_id)
        log.info("Чат %s: владелец пишет сам — ИИ замолкает до нового входящего.", chat_id)

    print("\nЮзербот запущен. Ctrl+C — остановить.\n")
    try:
        await client.run_until_disconnected()
    finally:
        await provider.close()
        await client.disconnect()


def main() -> None:
    try:
        asyncio.run(run())
    except (KeyboardInterrupt, SystemExit):
        print("Остановлено.")


if __name__ == "__main__":
    main()
