"""Deterministic surface-answer-span extraction.

Both answering methods (closed-book and grounded) ultimately need to turn a
candidate sentence into a short answer string. This module implements a small,
explicitly rule-based extractor - the deterministic analog of the span the
model chooses to emit. It never invents text: if no rule matches, extraction
fails and the calling backend must abstain.
"""

from __future__ import annotations

import re
from typing import List, Optional

_YEAR_RE = re.compile(r"\b(1[4-9]\d{2}|20[0-4]\d)\b")
_MONTH_YEAR_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+(1[4-9]\d{2}|20[0-4]\d)\b",
    re.IGNORECASE,
)
_CAPITALIZED_SPAN_RE = re.compile(
    r"\b([A-Z][a-z']+(?:\s+(?:of|the|de|II|III|IV|V|\d))*(?:\s+[A-Z][A-Za-z']+)+)\b"
)
_SIMPLE_CAPITALIZED_RE = re.compile(r"\b[A-Z][A-Za-z']+(?:\s+[A-Z][A-Za-z']+)*\b")
_QUESTION_HINT_RE = re.compile(
    r"(?:what group|what genre|what movement|what era|which genre|movement as|styled|"
    r"known as|called|description)"
)
_LOCATED_RE = re.compile(
    r"\b(?:born|founded|formed|established|located|situated|based|resides)\b"
    r"[^,\.0-9]{0,60}?\b(?:in|at|near)\s+([A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*)*)"
    r"(?=[,\.]|\s+(?:and|the)\b|\s*$)",
)


def question_type(question: str) -> str:
    q = question.strip().lower()
    if re.match(r"^(are both|is both|do both|were both|were either|are either|are all)", q):
        return "comparison"
    if re.match(r"^who\b", q):
        return "who"
    if re.match(r"^when\b", q) or "what year" in q:
        return "when"
    if re.match(r"^where\b", q) or q.startswith("in what"):
        return "where"
    if re.match(r"^(what|which)\b", q):
        return "what"
    if re.match(r"^how (many|much|long|far)", q):
        return "quantity"
    return "other"


def extract_answer_span(question: str, sentence: str) -> Optional[str]:
    """Extract a short answer span from ``sentence`` conditioned on the question.

    Deterministic and purely lexical. Returns None when extraction fails.
    """
    qtype = question_type(question)
    caps = _CAPITALIZED_SPAN_RE.findall(sentence) + _SIMPLE_CAPITALIZED_RE.findall(sentence)
    q_lower = question.lower()

    if qtype == "comparison":
        if sentence.strip().lower().startswith(("yes", "no")):
            return sentence.strip().lower().split()[0]
        # Not a yes/no sentence. Aggregation-style two-entity questions such
        # as "X and Y have which mutual occupations?" ARE evaluable against
        # a biography sentence, so fall through to the generic rules - but
        # only when the question lacks explicit compare markers (both/
        # either/or), which correctly stay None (measured failure cases).
        if not re.search(r"\b(?:both|either)\b|\s+or\s+", question.lower()):
            qtype = "other"

    if qtype == "when":
        m = _MONTH_YEAR_RE.search(sentence)
        if m:
            return m.group(0)
        years = _YEAR_RE.findall(sentence)
        if years:
            return years[0]
        return None

    if qtype == "who":
        m = re.search(
            r"\b(?:by|from)\s+([A-Z][A-Za-z']+(?:\s+[A-Z][A-Za-z']+)*)", sentence
        )
        if m:
            return m.group(1).strip()
        # prefer a person-like span not echoed by the question
        for span in caps:
            if span.lower() not in q_lower and not span.isupper():
                return span.strip()
        return caps[0].strip() if caps else None

    if qtype == "where":
        m = _LOCATED_RE.search(sentence)
        if m:
            return m.group(1).strip()
        m = re.search(r"\bin\s+([A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*)*)", sentence)
        if m:
            return m.group(1).strip()
        carried = [s for s in caps if s.lower() not in q_lower]
        if carried:
            return carried[-1].strip()
        return None

    if qtype == "quantity":
        nums = re.findall(r"\b\d[\d,\.]*\b", sentence)
        if nums:
            return nums[0]
        return None

    # 'what'/'other': prefer a span not repeated from the question, and prefer
    # to echo the semantic head right after a copula ("was a <Genre>", "is a Y")
    # True compare questions keep the None contract of the graceful-failure
    # test: without a yes/no sentence and with no aggregation fall-through,
    # extraction is not trustworthy.
    if qtype == "comparison":
        return None
    # Rule (b) copular-subject: HotpotQA easy gold sentences overwhelmingly
    # define the answer at sentence head ("The Maloof family is a prominent
    # ...", "When the Good Guys Win is the ninth studio album ...") - the
    # answer is the SENTENCE SUBJECT there.
    m_subject = re.match(
        r"^(?:The\s+|A\s+|An\s+)?([A-Z][A-Za-z'\-\s]+?)\s+(?:is|was|are)\s+(?:a\s|an\s|the\s|one\s)",
        sentence,
    )
    if m_subject:
        subj = m_subject.group(1).strip()
        if subj.lower() not in q_lower:
            return subj
    # Rule (c) occupation questions: the answer is the lowercase professional
    # phrase following the copula ("is an American musician, singer and ...")
    q_toks = set(q_lower.replace("?", " ").split())
    if q_toks & {"occupation", "occupations", "profession", "jobs", "job", "famous", "known"}:
        m_occ = re.search(
            r"\bis\s+(?:an?\s+)?(?:[A-Z][a-z][a-z]+\s+|American\s+)?(([a-z][a-z\-']+(?:[ ,\-']+[a-z][a-z\-']*){1,6}))",
            sentence,
        )
        if m_occ:
            return m_occ.group(1).strip().strip(",")
    if re.search(r"\b(?:is|was|are|were)\b.{0,40}\b(?:by|in)\b", sentence, re.IGNORECASE) and caps:
        not_in_q = [s for s in caps if s.lower() not in q_lower and not s.isupper()]
        return (not_in_q[-1] if not_in_q else caps[-1]).strip()
    m = re.search(r"\byears?\b|\bdynasty\b|\bstation\b|\btruth\b", sentence, re.IGNORECASE)
    if _QUESTION_HINT_RE.search(sentence.lower()) and caps:
        not_in_q = [s for s in caps if s.lower() not in q_lower]
        return (not_in_q[-1] if not_in_q else caps[-1]).strip()
    if re.search(r"\b(is|was|corresponds|belongs)\b", sentence, re.IGNORECASE) and caps:
        not_in_q = [s for s in caps if s.lower() not in q_lower]
        if len(caps) > 1:
            span = not_in_q[-1] if not_in_q else caps[-1]
            return span.strip()
        return caps[0].strip()
    if m and caps:
        return caps[-1].strip()
    not_in_q = [s for s in caps if s.lower() not in q_lower and not s.isupper()]
    if not_in_q:
        return not_in_q[-1].strip()
    return caps[-1].strip() if caps else None


_WH_WORDS = {
    "which", "what", "who", "whom", "whose", "where", "when", "why", "how",
    "does", "did", "do", "are", "is", "was", "were", "were", "in", "both",
    "either", "all", "a", "an", "or", "and", "the", "of",
}


def extract_option_candidates(question: str) -> List[str]:
    """Extract answer options from an either/or (or both-and) question.

    Handles the two common HotpotQA shapes:
    - "..., <Option A> or <Option B>?"
    - "both <Option A> and <Option B>" (option form only; yes/no never uses it)

    Returns up to two candidates (an empty list when the shape is absent);
    callers must decide how to rank them against their own knowledge.
    """
    q = question.strip().rstrip("?").strip()
    opts: List[str] = []

    if "," in q:
        tail = q.rsplit(",", 1)[1]
        tail = re.sub(r"\s*\bor\b\s*", "|", tail)
        for piece in tail.split("|"):
            piece = piece.strip(" .")
            first = piece.split(" ", 1)[0].lower()
            if piece and first not in _WH_WORDS and piece[0].isupper():
                opts.append(piece)
    if len(opts) < 2:
        m = re.search(r"\bboth\s+(.+?)\s+and\s+(.+?)(?:\s+have|\s+are|\s+is|\s+of\b|\s+from\b|\?|$)", q)
        if m:
            a, b = m.group(1).strip(), m.group(2).strip()
            cand_a = _most_capitalized_part(a)
            cand_b = _most_capitalized_part(b)
            opts = [c for c in (cand_a, cand_b) if c]
    # de-dup, case-insensitive, max 2
    out: List[str] = []
    for o in opts:
        if o and not any(o.lower() == x.lower() for x in out):
            out.append(o)
    return out[:2]


def _most_capitalized_part(text: str) -> Optional[str]:
    spans = _CAPITALIZED_SPAN_RE.findall(text) + _SIMPLE_CAPITALIZED_RE.findall(text)
    if not spans:
        return None
    # the option usually ends the clause; take the last capitalized span
    return spans[-1]
