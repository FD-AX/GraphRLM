from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


DEFAULT_RLM_CONTEXT_WINDOW = 8192
DEFAULT_RLM_CHUNK_TEXT_RATIO = 0.75
DEFAULT_RLM_CHUNK_OVERLAP_RATIO = 0.0
DEFAULT_RLM_MIN_CHUNK_TOKENS = 512


def _get_env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default

    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"Env variable {name} must be int, got {raw_value!r}") from exc


def _get_env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default

    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Env variable {name} must be float, got {raw_value!r}") from exc


@dataclass(frozen=True)
class ChunkerSettings:
    """
    ChunkerSettings — настройки chunking-слоя.

    Базовая идея:
        max_chunk_tokens вычисляется из контекстного окна модели.

    Например:
        RLM_CONTEXT_WINDOW=8192
        RLM_CHUNK_TEXT_RATIO=0.75
        max_chunk_tokens = 6144

    Это оставляет часть контекстного окна под prompt, schema, state и output.
    """

    context_window: int = DEFAULT_RLM_CONTEXT_WINDOW
    text_budget_ratio: float = DEFAULT_RLM_CHUNK_TEXT_RATIO
    overlap_ratio: float = DEFAULT_RLM_CHUNK_OVERLAP_RATIO
    min_chunk_tokens: int = DEFAULT_RLM_MIN_CHUNK_TOKENS

    @property
    def max_chunk_tokens(self) -> int:
        calculated = int(self.context_window * self.text_budget_ratio)
        return max(self.min_chunk_tokens, calculated)

    @property
    def chunking_size(self) -> int:
        """
        Backward-compatible alias.
        Старое имя можно временно оставить, чтобы не ломать остальной код.
        """

        return self.max_chunk_tokens

    @property
    def overlap_tokens(self) -> int:
        return max(0, int(self.max_chunk_tokens * self.overlap_ratio))

    @classmethod
    def from_env(cls) -> "ChunkerSettings":
        return cls(
            context_window=_get_env_int(
                "RLM_CONTEXT_WINDOW",
                DEFAULT_RLM_CONTEXT_WINDOW,
            ),
            text_budget_ratio=_get_env_float(
                "RLM_CHUNK_TEXT_RATIO",
                DEFAULT_RLM_CHUNK_TEXT_RATIO,
            ),
            overlap_ratio=_get_env_float(
                "RLM_CHUNK_OVERLAP_RATIO",
                DEFAULT_RLM_CHUNK_OVERLAP_RATIO,
            ),
            min_chunk_tokens=_get_env_int(
                "RLM_MIN_CHUNK_TOKENS",
                DEFAULT_RLM_MIN_CHUNK_TOKENS,
            ),
        )

    def estimate_chunk_count(self, total_tokens: int) -> int:
        if total_tokens <= 0:
            return 0

        return max(1, math.ceil(total_tokens / self.max_chunk_tokens))


class TextChunk(BaseModel):
    """
    TextChunk — один фрагмент исходного документа.

    Это минимальная единица обработки extractor-а.
    """

    chunk_id: str = Field(description="Стабильный идентификатор чанка.")
    text: str = Field(description="Текст чанка.")
    index: int = Field(ge=0, description="Порядковый номер чанка.")

    token_count: int = Field(
        ge=0,
        description="Примерное количество токенов в чанке.",
    )

    start_char: int | None = Field(
        default=None,
        description="Начальная позиция чанка в исходном документе.",
    )

    end_char: int | None = Field(
        default=None,
        description="Конечная позиция чанка в исходном документе.",
    )

    source_document_id: str | None = Field(
        default=None,
        description="ID исходного документа.",
    )


class TokenCounter(Protocol):
    def count(self, text: str) -> int:
        ...


class ApproxTokenCounter:
    """
    ApproxTokenCounter — простой приблизительный счётчик токенов.

    Для русского текста берём консервативную оценку:
        примерно 1 токен на 3 символа.
    """

    def count(self, text: str) -> int:
        if not text:
            return 0

        return max(1, len(text) // 3)


class ParagraphChunker:
    """
    ParagraphChunker — chunker по абзацам.

    Он знает:
        - context-window-aware max_chunk_tokens;
        - token_counter;
        - текст.
    """

    def __init__(
        self,
        settings: ChunkerSettings | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.settings = settings or ChunkerSettings.from_env()
        self.token_counter = token_counter or ApproxTokenCounter()

    def stable_chunk_id(
        self,
        text: str,
        index: int,
        source_document_id: str | None = None,
    ) -> str:
        base = f"{source_document_id or 'document'}:{index}:{text}"
        digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:12]

        return f"chunk_{index}_{digest}"

    def split(
        self,
        text: str,
        source_document_id: str | None = None,
    ) -> list[TextChunk]:
        paragraphs = self._extract_paragraphs_with_offsets(text)

        chunks: list[TextChunk] = []
        current_parts: list[tuple[str, int, int]] = []
        current_tokens = 0

        for paragraph_text, paragraph_start, paragraph_end in paragraphs:
            paragraph_tokens = self.token_counter.count(paragraph_text)

            if paragraph_tokens > self.settings.max_chunk_tokens:
                if current_parts:
                    chunks.append(
                        self._build_chunk(
                            parts=current_parts,
                            index=len(chunks),
                            source_document_id=source_document_id,
                        )
                    )
                    current_parts = []
                    current_tokens = 0

                for part_text, part_start, part_end in self._split_large_paragraph(
                    paragraph_text=paragraph_text,
                    paragraph_start=paragraph_start,
                ):
                    chunks.append(
                        self._build_chunk(
                            parts=[(part_text, part_start, part_end)],
                            index=len(chunks),
                            source_document_id=source_document_id,
                        )
                    )

                continue

            would_exceed = (
                bool(current_parts)
                and current_tokens + paragraph_tokens > self.settings.max_chunk_tokens
            )

            if would_exceed:
                chunks.append(
                    self._build_chunk(
                        parts=current_parts,
                        index=len(chunks),
                        source_document_id=source_document_id,
                    )
                )
                current_parts = [(paragraph_text, paragraph_start, paragraph_end)]
                current_tokens = paragraph_tokens
            else:
                current_parts.append((paragraph_text, paragraph_start, paragraph_end))
                current_tokens += paragraph_tokens

        if current_parts:
            chunks.append(
                self._build_chunk(
                    parts=current_parts,
                    index=len(chunks),
                    source_document_id=source_document_id,
                )
            )

        return chunks

    def _build_chunk(
        self,
        parts: list[tuple[str, int, int]],
        index: int,
        source_document_id: str | None,
    ) -> TextChunk:
        chunk_body = "\n\n".join(part[0] for part in parts)
        start_char = parts[0][1] if parts else None
        end_char = parts[-1][2] if parts else None

        return TextChunk(
            chunk_id=self.stable_chunk_id(
                text=chunk_body,
                index=index,
                source_document_id=source_document_id,
            ),
            text=chunk_body,
            index=index,
            token_count=self.token_counter.count(chunk_body),
            start_char=start_char,
            end_char=end_char,
            source_document_id=source_document_id,
        )

    def _extract_paragraphs_with_offsets(self, text: str) -> list[tuple[str, int, int]]:
        """
        Извлекает непустые абзацы вместе с offsets в исходном тексте.

        Важно:
            Режем именно по блокам, разделённым пустыми строками,
            а не по каждой строке отдельно.
        """

        result: list[tuple[str, int, int]] = []

        for match in re.finditer(
            r"\S(?:.*?\S)?(?=\n\s*\n|\s*\Z)",
            text,
            flags=re.DOTALL,
        ):
            paragraph = match.group(0).strip()
            if not paragraph:
                continue

            leading_spaces = len(match.group(0)) - len(match.group(0).lstrip())
            start = match.start() + leading_spaces
            end = start + len(paragraph)
            result.append((paragraph, start, end))

        return result

    def _split_large_paragraph(
        self,
        paragraph_text: str,
        paragraph_start: int,
    ) -> list[tuple[str, int, int]]:
        """
        Fallback для абзацев, которые больше max_chunk_tokens.

        Сначала пытаемся резать по предложениям.
        Если отдельное предложение всё равно слишком большое — режем по символам
        приблизительно под token budget.
        """

        sentence_matches = list(
            re.finditer(r"[^.!?…]+[.!?…]*", paragraph_text, flags=re.DOTALL)
        )
        result: list[tuple[str, int, int]] = []
        current_sentences: list[tuple[str, int, int]] = []
        current_tokens = 0

        for match in sentence_matches:
            sentence = match.group(0).strip()
            if not sentence:
                continue

            sentence_start = (
                paragraph_start
                + match.start()
                + len(match.group(0))
                - len(match.group(0).lstrip())
            )
            sentence_end = sentence_start + len(sentence)
            sentence_tokens = self.token_counter.count(sentence)

            if sentence_tokens > self.settings.max_chunk_tokens:
                if current_sentences:
                    result.append(self._merge_parts(current_sentences))
                    current_sentences = []
                    current_tokens = 0

                result.extend(
                    self._split_by_approx_chars(
                        text=sentence,
                        absolute_start=sentence_start,
                    )
                )
                continue

            would_exceed = (
                bool(current_sentences)
                and current_tokens + sentence_tokens > self.settings.max_chunk_tokens
            )

            if would_exceed:
                result.append(self._merge_parts(current_sentences))
                current_sentences = [(sentence, sentence_start, sentence_end)]
                current_tokens = sentence_tokens
            else:
                current_sentences.append((sentence, sentence_start, sentence_end))
                current_tokens += sentence_tokens

        if current_sentences:
            result.append(self._merge_parts(current_sentences))

        return result

    def _merge_parts(self, parts: list[tuple[str, int, int]]) -> tuple[str, int, int]:
        text = " ".join(part[0] for part in parts)
        return text, parts[0][1], parts[-1][2]

    def _split_by_approx_chars(
        self,
        text: str,
        absolute_start: int,
    ) -> list[tuple[str, int, int]]:
        approx_chars = max(1, self.settings.max_chunk_tokens * 3)
        result: list[tuple[str, int, int]] = []

        local_start = 0
        while local_start < len(text):
            local_end = min(len(text), local_start + approx_chars)
            part = text[local_start:local_end].strip()

            if part:
                raw_part = text[local_start:local_end]
                leading_spaces = len(raw_part) - len(raw_part.lstrip())
                start = absolute_start + local_start + leading_spaces
                end = start + len(part)
                result.append((part, start, end))

            local_start = local_end

        return result


def chunk_text(
    text: str,
    source_document_id: str | None = None,
    max_chunk_tokens: int | None = None,
    settings: ChunkerSettings | None = None,
    token_counter: TokenCounter | None = None,
) -> list[TextChunk]:
    """
    Публичная wrapper-функция над ParagraphChunker.

    Приоритет настроек:
        1. Явный max_chunk_tokens.
        2. Явный settings.
        3. ChunkerSettings.from_env().

    Старый API с max_chunk_tokens сохранён.
    Новый API умеет читать:
        RLM_CONTEXT_WINDOW
        RLM_CHUNK_TEXT_RATIO
        RLM_CHUNK_OVERLAP_RATIO
        RLM_MIN_CHUNK_TOKENS
    """

    if max_chunk_tokens is not None:
        effective_settings = ChunkerSettings(
            context_window=max_chunk_tokens,
            text_budget_ratio=1.0,
            min_chunk_tokens=max_chunk_tokens,
        )
    else:
        effective_settings = settings or ChunkerSettings.from_env()

    chunker = ParagraphChunker(
        settings=effective_settings,
        token_counter=token_counter,
    )

    return chunker.split(
        text=text,
        source_document_id=source_document_id,
    )
