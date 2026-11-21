"""Dense (semantic) retrieval using a real pretrained sentence-embedding model.

Default model: ``sentence-transformers/all-MiniLM-L6-v2`` (~80 MB, CPU-friendly,
384-dimensional embeddings).

Design notes
------------
* **Real inference only.** If ``sentence-transformers`` cannot be imported or
  the model cannot be loaded/downloaded, ``EmbeddingModelError`` is raised with
  an actionable message. There is deliberately NO silent fallback to BM25:
  substituting lexical retrieval would invalidate any BM25-vs-dense comparison.
* **Cosine similarity.** Document vectors and query vectors are L2-normalised,
  so cosine similarity is a plain dot product and scores lie in ``[-1, 1]``
  (in practice roughly ``[0, 1]`` for this model family).
* **Persistent cache** (``data/embeddings/<model>/<namespace>.npz``) keyed by
  ``sha256(model | include_title | chunk_text)``. Unchanged chunks are never
  re-encoded; a re-chunked or edited document produces different chunk texts,
  hence different keys, hence fresh vectors. Switching models writes a
  different file, so vectors from one model are never mixed with another's.
* **Determinism.** Ranking ties break on ``doc_id`` ascending, matching the
  BM25 retriever's contract.
* Chunks of *inactive* documents never reach this class: ``DocumentStore``
  only hands over chunks of active documents.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import Retriever
from ..common import PROJECT_ROOT
from ..common import CorpusDoc, RetrievalHit

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "embeddings"

Embedder = Callable[[Sequence[str]], np.ndarray]


class EmbeddingModelError(RuntimeError):
    """Raised when the pretrained embedding model is unavailable or unusable."""


def cache_key(model_name: str, text: str, include_title: bool = True) -> str:
    """Stable content key: identical text + model + config => identical vector."""
    payload = f"{model_name}\x00{int(include_title)}\x00{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation in float32 (zero rows are left as zeros)."""
    arr = np.asarray(matrix, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (arr / norms).astype(np.float32)


def cosine_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector and every row of ``doc_matrix``.

    Inputs are normalised internally, so callers may pass raw vectors.
    """
    q = l2_normalize(np.asarray(query_vec, dtype=np.float32).reshape(1, -1))
    m = l2_normalize(doc_matrix)
    if m.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return (m @ q.T).ravel().astype(np.float32)


def _model_slug(model_name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "__" for ch in model_name)


class EmbeddingCache:
    """Persistent float32 vector cache: one ``.npz`` per (model, namespace)."""

    def __init__(
        self,
        root: Optional[Path] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        namespace: str = "default",
    ) -> None:
        if root is None:
            self.root = DEFAULT_CACHE_DIR
        else:
            self.root = Path(root)
            if not self.root.is_absolute():  # config paths are project-relative
                self.root = PROJECT_ROOT / self.root
        self.model_name = model_name
        self.namespace = namespace
        self.path = self.root / _model_slug(model_name) / f"{namespace}.npz"
        self._vectors: Dict[str, np.ndarray] = {}
        self.hits = 0   # cache hits during this process (for diagnostics)
        self.misses = 0
        self._load()

    # ------------------------------------------------------------------ io
    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with np.load(self.path, allow_pickle=False) as data:
                keys = [str(k) for k in data["keys"].tolist()]
                vectors = np.asarray(data["vectors"], dtype=np.float32)
        except Exception:
            # A truncated/corrupt cache must never break retrieval: ignore it
            # and re-encode (the next flush rewrites a valid file).
            self._vectors = {}
            return
        self._vectors = {k: vectors[i] for i, k in enumerate(keys) if i < len(vectors)}

    def _flush(self) -> None:
        if not self._vectors:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(self._vectors.keys()), dtype="<U64")
        vectors = np.vstack([self._vectors[k] for k in self._vectors]).astype(np.float32)
        tmp = self.path.with_suffix(".npz.tmp")
        with tmp.open("wb") as fh:
            np.savez(fh, keys=keys, vectors=vectors)
        tmp.replace(self.path)

    # --------------------------------------------------------------- access
    def get(self, key: str) -> Optional[np.ndarray]:
        vec = self._vectors.get(key)
        if vec is None:
            self.misses += 1
        else:
            self.hits += 1
        return vec

    def put_many(self, mapping: Dict[str, np.ndarray]) -> None:
        if not mapping:
            return
        self._vectors.update({k: np.asarray(v, dtype=np.float32) for k, v in mapping.items()})
        self._flush()

    def __len__(self) -> int:
        return len(self._vectors)

    def stats(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "entries": len(self._vectors),
            "hits": self.hits,
            "misses": self.misses,
            "model_name": self.model_name,
        }


class DenseRetriever(Retriever):
    """Semantic retrieval over dense sentence embeddings.

    ``embedder`` may be injected (tests, alternative backends) as a callable
    taking a sequence of texts and returning a ``(n, dim)`` matrix. When it is
    omitted, a real ``sentence_transformers.SentenceTransformer`` is loaded
    lazily on first use.
    """

    def __init__(
        self,
        docs: Sequence[CorpusDoc],
        model_name: str = DEFAULT_MODEL_NAME,
        batch_size: int = 32,
        cache_dir: Optional[Path] = None,
        namespace: str = "default",
        include_title: bool = True,
        device: Optional[str] = None,
        use_cache: bool = True,
        embedder: Optional[Embedder] = None,
    ) -> None:
        self.docs: List[CorpusDoc] = list(docs)
        self.model_name = model_name
        self.batch_size = max(1, int(batch_size))
        self.include_title = bool(include_title)
        self.device = device
        self.namespace = namespace
        self._embedder = embedder
        self.cache = (
            EmbeddingCache(cache_dir, model_name=model_name, namespace=namespace)
            if use_cache
            else None
        )
        self.encode_ms = 0.0        # time spent encoding (cold-start cost)
        self.model_load_ms = 0.0    # one-off SentenceTransformer load/import cost
        self.cache_reused = 0       # vectors reused from the persistent cache
        self._matrix: Optional[np.ndarray] = None
        self._keys: List[str] = []
        self._build()

    # -------------------------------------------------------------- encoding
    def _texts(self) -> List[str]:
        if self.include_title:
            return [f"{d.title}\n{d.text}" for d in self.docs]
        return [d.text for d in self.docs]

    def _resolve_embedder(self) -> Embedder:
        if self._embedder is not None:
            return self._embedder
        import time

        t0 = time.perf_counter()
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise EmbeddingModelError(
                "Dense retrieval needs the 'sentence-transformers' package. "
                "Install it with:  pip install sentence-transformers"
            ) from exc
        try:
            model = SentenceTransformer(self.model_name, device=self.device)
            self.model_load_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:  # download / network / corrupt checkpoint
            raise EmbeddingModelError(
                f"Could not load sentence-embedding model {self.model_name!r}: {exc}. "
                "Check your network connection (the weights are downloaded once and "
                "then cached locally by Hugging Face) or set retrieval.dense.model_name "
                "to a model that is already available on this machine."
            ) from exc
        self._embedder = model
        return self._embedder

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        embedder = self._resolve_embedder()
        encode_fn = getattr(embedder, "encode", None)
        if encode_fn is None:  # injected plain callable(texts) -> matrix
            encode_fn = embedder
        out: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            vectors = encode_fn(batch)
            arr = np.asarray(vectors, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            out.append(arr)
        if not out:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(out).astype(np.float32)

    # ----------------------------------------------------------------- build
    def _build(self) -> None:
        import time

        texts = self._texts()
        keys = [cache_key(self.model_name, t, self.include_title) for t in texts]
        self._keys = keys
        if not texts:
            self._matrix = np.zeros((0, 0), dtype=np.float32)
            return

        vectors: List[Optional[np.ndarray]] = [None] * len(texts)
        pending: Dict[str, str] = {}   # key -> text (duplicates encoded once)
        if self.cache is not None:
            for i, key in enumerate(keys):
                vec = self.cache.get(key)
                if vec is not None:
                    vectors[i] = vec
                    self.cache_reused += 1
                elif key not in pending:
                    pending[key] = texts[i]
        else:
            pending = {k: t for k, t in dict(zip(keys, texts)).items()}

        if pending:
            t0 = time.perf_counter()
            encoded = self._encode(list(pending.values()))
            self.encode_ms = (time.perf_counter() - t0) * 1000.0
            fresh = {key: encoded[i] for i, key in enumerate(pending)}
            if self.cache is not None:
                self.cache.put_many(fresh)
            for i, key in enumerate(keys):
                if vectors[i] is None:
                    vectors[i] = fresh[key]
        else:
            self.encode_ms = 0.0
        self._matrix = l2_normalize(np.vstack(vectors))

    # -------------------------------------------------------------- retrieval
    @property
    def dim(self) -> int:
        return 0 if self._matrix is None or self._matrix.size == 0 else int(self._matrix.shape[1])

    @property
    def n_docs(self) -> int:
        return len(self.docs)

    def retrieve(self, query: str, top_k: int) -> List[RetrievalHit]:
        if self.n_docs == 0 or not query.strip():
            return []
        query_vec = self._encode([query])[0]
        scores = cosine_similarity(query_vec, self._matrix)
        order = sorted(
            range(len(self.docs)),
            key=lambda i: (-float(scores[i]), self.docs[i].doc_id),
        )
        hits: List[RetrievalHit] = []
        for i in order[: max(0, int(top_k))]:
            doc = self.docs[i]
            hits.append(
                RetrievalHit(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    text=doc.text,
                    score=float(scores[i]),
                    method="dense",
                )
            )
        return hits

    def diagnostics(self) -> Dict[str, Any]:
        """Provenance used to verify that REAL model inference happened."""
        return {
            "retriever": "dense",
            "model_name": self.model_name,
            "embedding_dim": self.dim,
            "n_documents": self.n_docs,
            "encode_ms": round(self.encode_ms, 3),
            "model_load_ms": round(self.model_load_ms, 3),
            "vectors_reused_from_cache": self.cache_reused,
            "cache": self.cache.stats() if self.cache is not None else None,
        }
