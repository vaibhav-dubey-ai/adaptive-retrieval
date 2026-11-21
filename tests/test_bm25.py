from adaptive_qa.hotpot.data import CorpusDoc
from adaptive_qa.retrieval.bm25 import BM25Retriever, tokenize


def _docs():
    return [
        CorpusDoc("m_api_keys", "API Keys", "API keys help manage your payments and billing.", ()),
        CorpusDoc("m_apple_pie", "Apple Pie", "A classic apple pie recipe uses apples cinnamon and sugar. The pie bakes 40 minutes.", ()),
        CorpusDoc("m_keyboard", "Keyboard", "The keyboard input device connects to the computer.", ()),
        CorpusDoc("m_pomodoro", "Pomodoro Method", "A study method using 25 minute timers and breaks.", ()),
        CorpusDoc("m_pie_advice", "Pie Advice", "Every pie crust recipe benefits from cold butter and rest.", ()),
    ]


def test_tokenizer_lowercases_and_stops():
    toks = tokenize("The Apple PIE recipe, bakes 40 minutes!")
    assert toks == ["apple", "pie", "recipe", "bakes", "40", "minutes"]


def test_bm25_ranks_best_doc_first():
    ret = BM25Retriever(_docs())
    hits = ret.retrieve("apple pie recipe", top_k=2)
    assert hits[0].doc_id == "m_apple_pie"
    assert len(hits) == 2


def test_bm25_title_tokens_contribute():
    ret = BM25Retriever(_docs())
    hits = ret.retrieve("pomodoro", top_k=4)
    assert hits[0].doc_id == "m_pomodoro"


def test_bm25_deterministic_and_tie_broken():
    docs = [
        CorpusDoc("b", "", "the same sentence words here.", ()),
        CorpusDoc("a", "", "The same sentence words here.", ()),
    ]
    ret = BM25Retriever(docs)
    h1 = ret.retrieve("same sentence", top_k=2)
    h2 = ret.retrieve("same sentence", top_k=2)
    assert [x.doc_id for x in h1] == [x.doc_id for x in h2]
    # identical scores for identical docs -> tie broken by doc_id
    assert h1[0].doc_id == "a"


def test_top_k_limits_outputs():
    docs = [
        CorpusDoc("a", "", "a dog runs fast.", ()),
        CorpusDoc("b", "", "a dog walks slowly.", ()),
        CorpusDoc("c", "", "a cat sleeps.", ()),
    ]
    ret = BM25Retriever(docs)
    assert len(ret.retrieve("dog", top_k=1)) == 1
    assert len(ret.retrieve("dog", top_k=2)) == 2
