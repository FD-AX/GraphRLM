from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile


WORD_NAMESPACE = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def read_docx_text(path: Path) -> str:
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
