"""Tests for the golden-query evaluator (needs the all-MiniLM-L6-v2 encoder, no LLM)."""
import os
import sys
from types import SimpleNamespace

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from sentence_transformers import SentenceTransformer
from services.malay2sql_service import GoldenQuery, normalize_sql

GOLDEN = [
    ("SELECT count(*) FROM head WHERE age  >  56", "department_management"),
    ("SELECT name FROM singer ORDER BY age DESC", "concert_singer"),
    ("SELECT T1.name FROM airports AS T1 JOIN flights AS T2 ON T1.id = T2.src", "flight_1"),
]


@pytest.fixture(scope="module")
def encoder_holder():
    return SimpleNamespace(encoder=SentenceTransformer("all-MiniLM-L6-v2"))


@pytest.fixture
def golden(tmp_path, encoder_holder):
    path = tmp_path / "golden.sql"
    path.write_text("\n".join(f"{q}\t{db}" for q, db in GOLDEN) + "\n", encoding="utf-8")
    return GoldenQuery(encoder_holder, str(path))


def test_normalize_sql():
    assert normalize_sql("  SELECT a\n FROM  t ;") == "SELECT a FROM t"


def test_each_golden_query_scores_exactly_one(golden):
    for sql, db_id in GOLDEN:
        match = golden.best_match(sql)
        assert match.score == 1.0
        assert match.db_id == db_id


def test_formatting_does_not_change_the_score(golden):
    assert golden.calculate_cosine_similarity("select count(*) from head where age > 56;") == 1.0


def test_other_queries_score_below_one_and_pick_the_closest(golden):
    match = golden.best_match("SELECT name FROM singer ORDER BY age")
    assert 0.5 < match.score < 1.0
    assert match.golden_sql == "SELECT name FROM singer ORDER BY age DESC"
    assert golden.best_match("DELETE FROM users").score < match.score


def test_score_is_deterministic(golden):
    q = "SELECT id, name, email FROM users WHERE age > 25"
    assert golden.calculate_cosine_similarity(q) == golden.calculate_cosine_similarity(q)


def test_rerank_puts_the_golden_query_first(golden):
    ranked = golden.rerank_queries(["SELECT 1", GOLDEN[1][0], "SELECT name FROM singer"])
    assert ranked[0] == (GOLDEN[1][0], 1.0)
    assert [s for _, s in ranked] == sorted((s for _, s in ranked), reverse=True)


def test_embeddings_are_cached(tmp_path, encoder_holder, golden):
    cached = [f for f in os.listdir(tmp_path) if f.startswith(".golden_") and f.endswith(".npy")]
    assert len(cached) == 1
    again = GoldenQuery(encoder_holder, str(tmp_path / "golden.sql"))
    assert (again.golden_embeddings == golden.golden_embeddings).all()


def test_full_golden_file_every_line_counts(encoder_holder):
    # The old evaluator embedded the whole file as one string; the encoder
    # truncates at 256 tokens, so only the first ~12 queries mattered.
    path = os.path.join(ROOT, "data", "golden_query.sql")
    gq = GoldenQuery(encoder_holder, path)
    assert len(gq.golden_queries) == 8659
    for index in (0, 4000, 8658):
        assert gq.best_match(gq.golden_queries[index]).score == 1.0
