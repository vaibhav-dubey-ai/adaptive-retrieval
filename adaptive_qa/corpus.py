"""Corpus construction for retrieval.

The local corpus is the union of all 10 paragraphs (2 gold + 8 distractors)
of every sampled question, deduplicated across questions. A paragraph that is
a distractor for question X may be the gold evidence for question Y - that is
what makes retrieval non-trivial and realistic.

Leakage rule (enforced by tests): the gold-paragraph map
(``gold_doc_ids_by_question``) is written only to the *evaluation manifest*.
The retriever and decision engine consume only the plain document list.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .config import resolve_path
from .types import CorpusDoc


def _doc_hash(title: str, text: str) -> str:
    h = hashlib.sha1()
    h.update(title.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()[:12]


def _paragraph_text(sentences: Sequence[str]) -> str:
    return " ".join(s.strip() for s in sentences if s and s.strip())


def build_corpus(records: Iterable[Dict[str, Any]]) -> Tuple[List[CorpusDoc], Dict[str, Any]]:
    """Build the shared document corpus (plus evaluation-only metadata).

    Returns ``(docs, evaluation_meta)`` where ``evaluation_meta`` contains the
    per-question list of gold doc ids. ``evaluation_meta`` must be persisted
    only to the corpus *evaluation manifest*, never handed to retrieval-time
    components.
    """
    docs: List[CorpusDoc] = []
    seen: Dict[str, int] = {}
    gold_doc_ids_by_question: Dict[str, List[str]] = {}

    for rec in records:
        qid = rec["question_id"]
        paragraphs = rec["paragraphs"]
        gold_title_set = set(rec.get("gold_titles", []))

        found_gold: List[str] = []
        for para in paragraphs:
            title = para["title"]
            sentences = [s for s in para["sentences"] if s and s.strip()]
            text = _paragraph_text(sentences)
            if not text:
                continue
            key = _doc_hash(title, text)
            if key not in seen:
                seen[key] = len(docs)
                docs.append(
                    CorpusDoc(
                        doc_id=f"doc_{key}",
                        title=title,
                        text=text,
                        sentences=tuple(sentences),
                        source_questions=(),
                    )
                )
            if title in gold_title_set:
                found_gold.append(docs[seen[key]].doc_id)

        if found_gold:
            gold_doc_ids_by_question[qid] = sorted(set(found_gold))

    evaluation_meta = {
        "gold_doc_ids_by_question": gold_doc_ids_by_question,
        "n_documents": len(docs),
    }
    return docs, evaluation_meta


def save_corpus(docs: List[CorpusDoc], jsonl_path: str) -> None:
    p = resolve_path(jsonl_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(
                json.dumps(
                    {
                        "doc_id": doc.doc_id,
                        "title": doc.title,
                        "text": doc.text,
                        "sentences": list(doc.sentences),
                        "source_questions": list(doc.source_questions),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def save_eval_manifest(evaluation_meta: Dict[str, Any], manifest_path: str) -> None:
    """Persist the evaluation-only gold map to a separate manifest file."""
    p = resolve_path(manifest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(evaluation_meta)
    payload["_note"] = (
        "EVALUATION ONLY: gold_doc_ids_by_question names reference evidence "
        "per question. Retrieval components and the decision engine must never "
        "read this file."
    )
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_corpus(jsonl_path: str) -> List[CorpusDoc]:
    p = resolve_path(jsonl_path)
    docs: List[CorpusDoc] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docs.append(
                CorpusDoc(
                    doc_id=obj["doc_id"],
                    title=obj["title"],
                    text=obj["text"],
                    sentences=tuple(obj.get("sentences", [])),
                    source_questions=tuple(obj.get("source_questions", [])),
                )
            )
    return docs
