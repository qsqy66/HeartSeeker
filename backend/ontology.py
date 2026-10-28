"""Loader and lookup helpers for a small local medical ontology.

The bundled subset is deliberately tiny: it maps common clinical concepts to
their Chinese/English synonyms, ICD-10-CM codes and related concepts.  It is a
deterministic local aid for query expansion and concept lookup, not a clinical
knowledge base.

A deployment can replace the subset with its own JSON file by setting
``PAPER_READER_ONTOLOGY=/path/to/ontology.json`` or calling
:func:`reload_ontology`.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "medical_ontology.json"
ENV_PATH = "PAPER_READER_ONTOLOGY"
CJK_RE = re.compile(r"[㐀-鿿]")
ASCII_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")


class OntologyError(ValueError):
    """Raised when an ontology file is missing or malformed."""


@dataclass
class Concept:
    id: str
    term: str
    aliases: list[str] = field(default_factory=list)
    icd10: list[str] = field(default_factory=list)
    category: str = ""
    parent: Optional[str] = None
    related: list[str] = field(default_factory=list)
    note: str = ""

    def names(self) -> list[str]:
        return [self.term, *self.aliases]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "term": self.term, "aliases": list(self.aliases),
            "icd10": list(self.icd10), "category": self.category,
            "parent": self.parent, "related": list(self.related), "note": self.note,
        }


def _string_list(value: Any, field_name: str, where: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise OntologyError(f"{where} 的 {field_name} 必须是字符串列表")
    return [item.strip() for item in value if item.strip()]


@dataclass
class Ontology:
    concepts: list[Concept]
    version: str = ""
    description: str = ""
    source_path: Optional[str] = None

    def __post_init__(self) -> None:
        self.by_id: dict[str, Concept] = {concept.id: concept for concept in self.concepts}

    def _match_score(self, concept: Concept, query: str) -> Optional[tuple[float, str]]:
        lowered = query.lower().strip()
        if not lowered:
            return None
        if lowered == concept.id.lower():
            return 1.0, concept.id
        if lowered == concept.term.lower():
            return 1.0, concept.term
        if lowered in {code.lower() for code in concept.icd10}:
            return 1.0, "ICD-10 " + lowered.upper()
        for alias in concept.aliases:
            if lowered == alias.lower():
                return 1.0, alias

        query_tokens = {token.lower() for token in ASCII_WORD_RE.findall(query)}
        has_cjk = bool(CJK_RE.search(query))
        best: Optional[tuple[float, str]] = None
        for name in concept.names():
            name_lower = name.lower()
            if CJK_RE.search(name):
                if has_cjk and name in query:
                    score = min(1.0, 0.55 + 0.05 * len(name))
                    if best is None or score > best[0]:
                        best = (score, name)
            else:
                if ASCII_WORD_RE.fullmatch(name):
                    if name_lower in query_tokens:
                        score = 0.8
                        if best is None or score > best[0]:
                            best = (score, name)
                elif has_cjk and name_lower in lowered:
                    score = 0.6
                    if best is None or score > best[0]:
                        best = (score, name)
        return best

    def lookup(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for concept in self.concepts:
            found = self._match_score(concept, query)
            if found is None:
                continue
            score, matched = found
            matches.append({**concept.as_dict(), "matched": matched,
                            "match_score": round(score, 3)})
        matches.sort(key=lambda item: (-item["match_score"], item["id"]))
        return matches[:max(1, limit)]

    def related(self, concept_id: str) -> list[dict[str, Any]]:
        concept = self.by_id.get(concept_id) or self.by_id.get(concept_id.upper())
        if concept is None:
            return []
        results: list[dict[str, Any]] = []
        for related_id in concept.related:
            target = self.by_id.get(related_id)
            if target is not None:
                results.append(target.as_dict())
        return results

    def children(self, concept_id: str) -> list[dict[str, Any]]:
        return [concept.as_dict() for concept in self.concepts
                if concept.parent == concept_id]

    def expand_terms(self, text: str, limit: int = 5) -> list[str]:
        """Return aliases of concepts mentioned in ``text`` (for retrieval)."""
        extras: list[str] = []
        for concept in self.concepts:
            if self._match_score(concept, text) is None:
                continue
            for name in concept.names():
                if name not in extras:
                    extras.append(name)
            if len(extras) >= limit:
                break
        return extras[:limit]

    def stats(self) -> dict[str, Any]:
        categories: dict[str, int] = {}
        for concept in self.concepts:
            categories[concept.category or "未分类"] = categories.get(
                concept.category or "未分类", 0) + 1
        return {"version": self.version, "concepts": len(self.concepts),
                "categories": categories, "source_path": self.source_path}


def load_ontology(path: Optional[Path | str] = None) -> Ontology:
    source = Path(path) if path else Path(os.environ.get(ENV_PATH) or DEFAULT_PATH)
    if not source.is_file():
        raise OntologyError(f"找不到本地医学本体文件：{source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OntologyError(f"读取医学本体失败：{error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("concepts"), list):
        raise OntologyError("医学本体必须是包含 concepts 列表的 JSON 对象")
    concepts: list[Concept] = []
    seen: set[str] = set()
    for index, raw in enumerate(payload["concepts"]):
        where = f"第 {index + 1} 个概念"
        if not isinstance(raw, dict):
            raise OntologyError(f"{where} 必须是对象")
        concept_id = str(raw.get("id") or "").strip()
        term = str(raw.get("term") or "").strip()
        if not concept_id or not term:
            raise OntologyError(f"{where} 缺少 id 或 term")
        if concept_id in seen:
            raise OntologyError(f"概念 id 重复：{concept_id}")
        seen.add(concept_id)
        parent = raw.get("parent")
        concepts.append(Concept(
            id=concept_id,
            term=term,
            aliases=_string_list(raw.get("aliases"), "aliases", where),
            icd10=_string_list(raw.get("icd10"), "icd10", where),
            category=str(raw.get("category") or "").strip(),
            parent=str(parent).strip() if parent else None,
            related=_string_list(raw.get("related"), "related", where),
            note=str(raw.get("note") or "").strip(),
        ))
    return Ontology(concepts=concepts,
                    version=str(payload.get("version") or ""),
                    description=str(payload.get("description") or ""),
                    source_path=str(source))


_ONTOLOGY: Optional[Ontology] = None


def get_ontology() -> Ontology:
    global _ONTOLOGY
    if _ONTOLOGY is None:
        _ONTOLOGY = load_ontology()
    return _ONTOLOGY


def reload_ontology(path: Optional[Path | str] = None) -> Ontology:
    global _ONTOLOGY
    _ONTOLOGY = load_ontology(path)
    return _ONTOLOGY


def lookup(query: str, limit: int = 5) -> list[dict[str, Any]]:
    return get_ontology().lookup(query, limit=limit)


def stats() -> dict[str, Any]:
    return get_ontology().stats()


def expand_terms(text: str, limit: int = 5) -> Iterable[str]:
    return get_ontology().expand_terms(text, limit=limit)
