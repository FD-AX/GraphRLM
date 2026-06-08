from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

from app.core.chunking import chunk_text


ROOT_DIR = Path(__file__).resolve().parents[1]
WORD_NAMESPACE = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def _docx_text(path: Path) -> str:
    with ZipFile(path) as archive:
        document_xml = archive.read("word/document.xml")

    root = ET.fromstring(document_xml)
    paragraphs: list[str] = []

    for paragraph in root.findall(".//w:body/w:p", WORD_NAMESPACE):
        text = "".join(
            node.text or ""
            for node in paragraph.findall(".//w:t", WORD_NAMESPACE)
        ).strip()
        if text:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def test_chunk_text_splits_docx_fixture_into_stable_chunks() -> None:
    docx_files = list(ROOT_DIR.glob("*.docx"))
    assert len(docx_files) == 1

    docx_path = docx_files[0]
    text = _docx_text(docx_path)

    chunks = chunk_text(
        text=text,
        source_document_id=docx_path.stem,
        max_chunk_tokens=1000,
    )

    assert len(chunks) == 15
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert {chunk.source_document_id for chunk in chunks} == {docx_path.stem}
    assert [chunk.token_count for chunk in chunks[:5]] == [907, 961, 318, 891, 868]

    previous_end = 0
    for chunk in chunks:
        assert chunk.start_char is not None
        assert chunk.end_char is not None
        assert chunk.start_char >= previous_end
        assert chunk.end_char > chunk.start_char
        assert text[chunk.start_char : chunk.end_char] == chunk.text
        assert chunk.token_count <= 1005
        previous_end = chunk.end_char
