from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import time

import pandas as pd
import psutil

from evaluation_utils import hit_rate, recall, rr, precision
from ranking_utils import rrf
from strategies.BM25.item_name_search_strategy import ItemNameSearchStrategy
from strategies.DenseSearch.dense_search_on_description import DenseSearch


PROJECT_DIR = Path(__file__).parent
ITEM_DATA_DIR = PROJECT_DIR / "data"
QUERY_DATA_PATH = ITEM_DATA_DIR / "user_queries_with_history_and_clicks.csv"
REPORT_DIR = PROJECT_DIR / "evaluation_results"
DEFAULT_QUERY_LIMIT = 100
DEFAULT_K = 100


def parse_item_ids(value: str | list[str] | None) -> list[str]:
    if isinstance(value, list):
        return [str(item_id) for item_id in value]
    if value is None or pd.isna(value):
        return []
    return [str(item_id) for item_id in json.loads(value)]


def get_bm25_search_results(strategy: ItemNameSearchStrategy, query: str, k: int) -> list[str]:
    return strategy.bm25_search_on_item_name(query, k)


def get_dense_search_results(strategy: DenseSearch, query: str, k: int) -> list[str]:
    return strategy.dense_search_on_item_desc(query, k)


def retrieve_rankings(query: str, keyword_search_strategy: ItemNameSearchStrategy, dense_search_strategy: DenseSearch,
    executor: ThreadPoolExecutor, k: int) -> dict[str, list[str]]:
    keyword_future = executor.submit(
        get_bm25_search_results, keyword_search_strategy, query, k
    )
    dense_future = executor.submit(
        get_dense_search_results, dense_search_strategy, query, k
    )
    keyword_results = keyword_future.result()
    dense_results = dense_future.result()

    return {
        "BM25": keyword_results,
        "Dense": dense_results,
        "Fused": rrf.reciprocal_rank_fusion(
            [keyword_results, dense_results], top_k=k
        ),
    }


def score_ranking(ranked_ids: list[str], relevant_ids: list[str], k: int) -> dict[str, float | None]:
    if not relevant_ids:
        return {
            "rr_at_k": None,
            "hit_rate_at_k": None,
            "recall_at_k": None,
            "precision_at_k": None,
        }

    return {
        "rr_at_k": rr.reciprocal_rank_at_k(ranked_ids, relevant_ids, k),
        "hit_rate_at_k": hit_rate.hit_rate_at_k(ranked_ids, relevant_ids, k),
        "recall_at_k": recall.recall_at_k(ranked_ids, relevant_ids, k),
        "precision_at_k": precision.precision_at_k(ranked_ids, relevant_ids, k)
    }


def evaluate_queries(queries: pd.DataFrame, query_limit: int, k: int) -> tuple[pd.DataFrame, int]:
    selected_queries = queries.head(query_limit)
    keyword_search_strategy = ItemNameSearchStrategy()
    dense_search_strategy = DenseSearch()
    detail_rows = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        for query_number, query in enumerate(
            selected_queries.itertuples(index=False), start=1
        ):
            rankings = retrieve_rankings(
                query.query_string,
                keyword_search_strategy,
                dense_search_strategy,
                executor,
                k,
            )
            relevance_sets = {
                "related_items": parse_item_ids(query.response_item_ids),
                "user_clicks": parse_item_ids(query.clicked_item_ids),
            }

            for relevance_source, relevant_ids in relevance_sets.items():
                for retriever, ranked_ids in rankings.items():
                    detail_rows.append(
                        {
                            "query_number": query_number,
                            "query": query.query_string,
                            "relevance_source": relevance_source,
                            "retriever": retriever,
                            "relevant_item_count": len(set(relevant_ids)),
                            **score_ranking(ranked_ids, relevant_ids, k),
                        }
                    )

    return pd.DataFrame(detail_rows), len(selected_queries)


def build_summary(details: pd.DataFrame, query_count: int, k: int, run_id: str, runtime: dict[str, float]) -> pd.DataFrame:
    summary = (
        details.groupby(["relevance_source", "retriever"], sort=False)
        .agg(
            queries_with_labels=(
                "relevant_item_count",
                lambda counts: int((counts > 0).sum()),
            ),
            mrr_at_k=("rr_at_k", "mean"),
            hit_rate_at_k=("hit_rate_at_k", "mean"),
            recall_at_k=("recall_at_k", "mean"),
            precision_at_k=("precision_at_k", "mean"),
        )
        .reset_index()
    )
    summary.insert(0, "run_id", run_id)
    summary.insert(3, "queries_evaluated", query_count)
    summary["label_coverage"] = summary["queries_with_labels"] / query_count
    summary.insert(4, "k", k)
    for metric_name, metric_value in runtime.items():
        summary[metric_name] = metric_value
    return summary


def save_reports(
    details: pd.DataFrame, summary: pd.DataFrame, run_id: str
) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    details_path = REPORT_DIR / f"query_metrics_{run_id}.csv"
    summary_path = REPORT_DIR / f"summary_{run_id}.csv"
    details.to_csv(details_path, index=False)
    summary.to_csv(summary_path, index=False)
    return details_path, summary_path


def print_report(summary: pd.DataFrame, runtime: dict[str, float]) -> None:
    display_columns = [
        "relevance_source",
        "retriever",
        "queries_evaluated",
        "queries_with_labels",
        "label_coverage",
        "k",
        "mrr_at_k",
        "hit_rate_at_k",
        "recall_at_k",
        "precision_at_k"
    ]
    print("\nEvaluation summary")
    print(
        summary[display_columns].to_string(
            index=False,
            formatters={
                "label_coverage": "{:.1%}".format,
                "mrr_at_k": "{:.4f}".format,
                "hit_rate_at_k": "{:.4f}".format,
                "recall_at_k": "{:.4f}".format,
                "precision_at_k":  "{:.4f}".format
            },
        )
    )
    print("\nRun performance")
    print(pd.DataFrame([runtime]).to_string(index=False, float_format="{:.3f}".format))


def main() -> None:
    query_count = DEFAULT_QUERY_LIMIT
    k = DEFAULT_K
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    process = psutil.Process(os.getpid())
    process.cpu_percent()
    memory_before = process.memory_info().rss
    start_time = time.perf_counter()

    queries = pd.read_csv(QUERY_DATA_PATH)
    details, evaluated_query_count = evaluate_queries(queries, query_count, k)
    elapsed_seconds = time.perf_counter() - start_time
    cpu_percent = process.cpu_percent()
    memory_after = process.memory_info().rss

    runtime = {
        "elapsed_seconds": elapsed_seconds,
        "cpu_percent": cpu_percent,
        "rss_delta_mb": (memory_after - memory_before) / (1024 * 1024),
        "rss_after_mb": memory_after / (1024 * 1024),
    }
    details.insert(0, "run_id", run_id)
    details.insert(4, "k", k)
    summary = build_summary(
        details, evaluated_query_count, k, run_id, runtime
    )
    #details_path, summary_path = save_reports(details, summary, run_id)
    print_report(summary, runtime)
    #print(f"\nPer-query results: {details_path.relative_to(PROJECT_DIR)}")
    #print(f"Summary results:   {summary_path.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()