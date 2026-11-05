"""Run the Malay2SQL pipeline over a small Malay test set and score each
generated query with the golden-query evaluator.

    python scripts/evaluate.py --runs 3 --out results.json

Metrics per run:
  golden_score  similarity to the closest golden query (1.0 = identical)
  executes      the SQL runs without error on an in-memory SQLite `users` table
Translation samples (do_sample=True), so each question is run --runs times.
"""
import argparse
import asyncio
import json
import os
import sqlite3
import statistics
import sys
import time
from datetime import datetime, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Settings
from services.malay2sql_service import Malay2SQLService

USERS_SCHEMA = {
    "table_name": "users",
    "columns": {
        "id": {"data_type": "Int64", "description": "Unique identifier for each user"},
        "name": {"data_type": "String", "description": "Full name of the user"},
        "email": {"data_type": "String", "description": "Email address of the user"},
        "age": {"data_type": "Int64", "description": "Age of the user in years"},
    },
}

# Same questions as tests/test_malay2sql_service.py
TEST_SET = [
    "Tunjukkan semua pengguna",
    "Cari pengguna yang berumur lebih dari 25 tahun",
    "Senaraikan nama dan emel untuk semua pengguna",
    "Berapa ramai pengguna yang ada dalam sistem?",
]

USER_ID = "evaluation"


def executes(sql: str) -> tuple:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE TABLE users (id INTEGER, name TEXT, email TEXT, age INTEGER)")
        conn.executemany("INSERT INTO users VALUES (?, ?, ?, ?)",
                         [(1, "Aminah", "aminah@example.com", 31),
                          (2, "Ravi", "ravi@example.com", 22),
                          (3, "Mei Ling", "meiling@example.com", 27)])
        rows = conn.execute(sql).fetchall()
        return True, rows, None
    except Exception as e:
        return False, None, str(e)
    finally:
        conn.close()


async def main(runs: int, pause: float, out: str):
    cfg = Settings().llm_config()
    service = Malay2SQLService(cfg, cache_client=None)
    service.initialize_schema(USERS_SCHEMA, user_id=USER_ID)
    report = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": cfg.provider,
        "model": cfg.model,
        "base_url": cfg.base_url,
        "feedback_examples_loaded": len(service.feedback_history),
        "golden_queries": len(service.golden_query.golden_queries),
        "test_set_size": len(TEST_SET),
        "runs_per_question": runs,
        "results": [],
    }
    for question in TEST_SET:
        for run in range(runs):
            started = time.time()
            try:
                result = await service.process_query(question, user_id=USER_ID)
            except Exception as e:
                report["results"].append({"question": question, "run": run, "error": str(e)})
                print(f"ERROR {question!r}: {e}", flush=True)
                if "429" in str(e) or "quota" in str(e).lower():
                    report["stopped_early"] = f"quota: {e}"
                    return finish(report, out)
                continue
            ok, rows, err = executes(result.sql_query)
            rescore = service.golden_query.calculate_cosine_similarity(result.sql_query)
            entry = {
                "question": question,
                "run": run,
                "translation": result.english_translation,
                "sql": result.sql_query,
                "golden_score": result.golden_score,
                "rescore_equal": rescore == result.golden_score,
                "closest_golden_query": result.closest_golden_query,
                "executes": ok,
                "rows": rows,
                "sqlite_error": err,
                "seconds": round(time.time() - started, 2),
            }
            report["results"].append(entry)
            print(json.dumps(entry, ensure_ascii=False), flush=True)
            time.sleep(pause)  # stay under the free-tier requests-per-minute limit
    return finish(report, out)


def finish(report, out):
    scored = [r for r in report["results"] if "golden_score" in r]
    if scored:
        scores = [r["golden_score"] for r in scored]
        report["summary"] = {
            "scored": len(scored),
            "errors": len(report["results"]) - len(scored),
            "golden_score_mean": round(statistics.mean(scores), 4),
            "golden_score_min": round(min(scores), 4),
            "golden_score_max": round(max(scores), 4),
            "executes_rate": round(sum(r["executes"] for r in scored) / len(scored), 4),
            "deterministic_rescore": all(r["rescore_equal"] for r in scored),
            "per_question_mean": {
                q: round(statistics.mean(r["golden_score"] for r in scored if r["question"] == q), 4)
                for q in dict.fromkeys(r["question"] for r in scored)
            },
        }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(report.get("summary", {}), indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--pause", type=float, default=7.0, help="seconds between LLM calls")
    parser.add_argument("--out", default="evaluation_results.json")
    args = parser.parse_args()
    asyncio.run(main(args.runs, args.pause, args.out))
