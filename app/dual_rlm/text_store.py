from __future__ import annotations

import hashlib
import re

from app.dual_rlm.models import SourceChunk, TextEvidenceSpan


class ImmutableTextStore:
    def __init__(self, chunks: list[SourceChunk]) -> None:
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self._ordered = sorted(chunks, key=lambda chunk: (chunk.start_char, chunk.chunk_id))

    def search_text(self, query: str, top_k: int = 5) -> list[SourceChunk]:
        query_terms = _terms(query)
        scored = []
        for chunk in self._ordered:
            chunk_terms = _terms(chunk.text)
            overlap = len(query_terms & chunk_terms)
            if overlap == 0:
                continue
            score = overlap / max(len(query_terms), 1)
            scored.append((score, chunk.start_char, chunk.chunk_id, chunk))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [chunk for _, _, _, chunk in scored[:top_k]]

    def read_chunk(self, chunk_id: str) -> SourceChunk | None:
        return self.chunks.get(chunk_id)

    def read_window(self, chunk_id: str, before: int = 1, after: int = 1) -> list[SourceChunk]:
        positions = {chunk.chunk_id: index for index, chunk in enumerate(self._ordered)}
        if chunk_id not in positions:
            return []
        index = positions[chunk_id]
        start = max(0, index - before)
        end = min(len(self._ordered), index + after + 1)
        return self._ordered[start:end]

    def find_occurrences(self, surface_forms: list[str]) -> list[TextEvidenceSpan]:
        spans = []
        for chunk in self._ordered:
            for surface in surface_forms:
                if not surface:
                    continue
                for match in re.finditer(re.escape(surface), chunk.text, flags=re.IGNORECASE):
                    start = chunk.start_char + match.start()
                    end = chunk.start_char + match.end()
                    spans.append(
                        TextEvidenceSpan(
                            evidence_span_id=_span_id(chunk.document_id, chunk.chunk_id, start, end),
                            document_id=chunk.document_id,
                            chunk_id=chunk.chunk_id,
                            start_char=start,
                            end_char=end,
                            text=chunk.text[match.start() : match.end()],
                        )
                    )
        return spans

    def evidence_for_chunk(self, chunk: SourceChunk) -> TextEvidenceSpan:
        end_char = chunk.end_char
        if end_char is None:
            end_char = chunk.start_char + len(chunk.text)
        return TextEvidenceSpan(
            evidence_span_id=_span_id(chunk.document_id, chunk.chunk_id, chunk.start_char, end_char),
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            start_char=chunk.start_char,
            end_char=end_char,
            text=chunk.text,
        )


def _terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-zА-Яа-яЁё0-9]+", text)
        if len(token) > 2
    }


def _span_id(document_id: str, chunk_id: str, start: int, end: int) -> str:
    digest = hashlib.sha256(f"{document_id}:{chunk_id}:{start}:{end}".encode("utf-8")).hexdigest()[:12]
    return f"text_span_{digest}"
