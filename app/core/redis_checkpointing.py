from __future__ import annotations

import os
import threading

from langgraph.checkpoint.redis import RedisSaver


class RedisClient:
    """
    RedisClient — singleton/factory для Redis-backed компонентов проекта.

    Сейчас основная задача класса:
        - создать RedisSaver для LangGraph checkpointing;
        - не плодить несколько RedisSaver внутри одного worker-процесса;
        - централизовать REDIS_URL;
        - один раз выполнить setup().

    Важно:
        Это не сам AgentState.
        AgentState остаётся TypedDict-схемой состояния.
        RedisSaver — durable backend, который сохраняет checkpoints AgentState.
    """

    _checkpointer: RedisSaver | None = None
    _lock = threading.Lock()

    @classmethod
    def get_redis_url(cls) -> str:
        """
        Возвращает Redis URL для LangGraph checkpoint backend.

        Локально:
            redis://localhost:6379/0

        В docker-compose:
            redis://redis:6379/0

        В проде:
            REDIS_URL=redis://user:password@host:6379/0
        """

        return os.getenv("REDIS_URL", "redis://localhost:6379/0")

    @classmethod
    def build_checkpointer(cls) -> RedisSaver:
        """
        Возвращает singleton RedisSaver.

        Первый вызов:
            - создаёт RedisSaver;
            - вызывает setup();
            - сохраняет объект в cls._checkpointer.

        Последующие вызовы:
            - возвращают уже созданный RedisSaver.

        Это нужно, чтобы build_graph() не создавал новое Redis-подключение
        при каждом вызове внутри одного worker-процесса.
        """

        if cls._checkpointer is not None:
            return cls._checkpointer

        with cls._lock:
            if cls._checkpointer is not None:
                return cls._checkpointer

            redis_url = cls.get_redis_url()

            checkpointer = RedisSaver.from_conn_string(redis_url)

            # setup() создаёт/обновляет служебные структуры RedisSaver.
            # Для локального MVP можно безопасно вызывать при старте worker-а.
            if hasattr(checkpointer, "setup"):
                checkpointer.setup()

            cls._checkpointer = checkpointer

            return cls._checkpointer

    @classmethod
    def reset_checkpointer(cls) -> None:
        """
        Сбрасывает singleton checkpointer.

        Нужен в основном для тестов, когда нужно пересоздать RedisSaver
        с другим REDIS_URL или изолированным Redis DB.
        """

        with cls._lock:
            cls._checkpointer = None