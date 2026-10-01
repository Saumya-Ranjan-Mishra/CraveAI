from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
import re
from pathlib import Path
import time

import numpy as np
import pandas as pd
import psutil
import xgboost as xgb

from evaluation_utils import hit_rate, recall, rr, precision
from ranking_utils import rrf
from strategies.BM25.item_name_search_strategy import ItemNameSearchStrategy
from strategies.DenseSearch.dense_search_on_description import DenseSearch

PROJECT_DIR = Path(__file__).parent
ITEM_DATA_DIR = PROJECT_DIR / "data"
QUERY_DATA_PATH = ITEM_DATA_DIR / "user_queries_with_history_and_clicks.csv"
REPORT_DIR = PROJECT_DIR / "evaluation_results"
MODEL_PATH = PROJECT_DIR / "indexes" / "xgboost_reranker_model" / "crave_ai_xgb_reranker.json"
DEFAULT_QUERY_LIMIT = 100
DEFAULT_K = 10

items_df = pd.read_csv(ITEM_DATA_DIR / "items.csv").fillna("")
items_by_id = items_df.set_index("item_id")
item_lookup = items_df.set_index("item_id").to_dict(orient="index")

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


def compute_token_overlap(query: str, text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    q_tokens = set(re.findall(r'\w+', query.lower()))
    t_tokens = set(re.findall(r'\w+', text.lower()))
    if not q_tokens:
        return 0.0
    return len(q_tokens.intersection(t_tokens)) / len(q_tokens)


def rerank_with_xgboost(
    query: str,
    candidate_ids: list[str],
    xgb_model: xgb.Booster,
    dense_strategy: DenseSearch,
) -> list[str]:
    if not candidate_ids:
        return []

    if not hasattr(dense_strategy, "_cached_embeddings"):
        dense_strategy._cached_embeddings = np.load(dense_strategy.item_desc_embeddings_path)
        dense_strategy._id_to_idx = {
            iid: idx for idx, iid in enumerate(pd.read_csv(dense_strategy.item_list_path)["item_id"])
        }
    doc_embeddings = dense_strategy._cached_embeddings
    id_to_idx = dense_strategy._id_to_idx
    query_vector = dense_strategy.embedding_model_obj.get_sentence_emdeddings([query])[0]

    features_list = []
    for rank, item_id in enumerate(candidate_ids):
        item_info = item_lookup.get(item_id, {})
        idx = id_to_idx.get(item_id)
        dense_score = float(doc_embeddings[idx] @ query_vector) if idx is not None else 0.0

        name_overlap = compute_token_overlap(query, str(item_info.get("item_name", "")))
        cat_overlap = compute_token_overlap(query, str(item_info.get("menu_category", "")))
        desc_overlap = compute_token_overlap(query, str(item_info.get("item_description", "")))

        try:
            rating = float(item_info.get("item_rating", 4.0))
        except (ValueError, TypeError):
            rating = 4.0

        features_list.append({
            "dense_score": dense_score,
            "name_token_overlap": name_overlap,
            "category_token_overlap": cat_overlap,
            "desc_token_overlap": desc_overlap,
            "candidate_rank": rank + 1,
            "item_rating": rating,
        })

    feature_cols = [
        "dense_score",
        "name_token_overlap",
        "category_token_overlap",
        "desc_token_overlap",
        "candidate_rank",
        "item_rating",
    ]
    df_feat = pd.DataFrame(features_list)[feature_cols]
    dmat = xgb.DMatrix(df_feat)
    preds = xgb_model.predict(dmat)

    # Sort candidates by descending XGBoost predicted score
    reranked_pairs = sorted(zip(candidate_ids, preds), key=lambda x: x[1], reverse=True)
    return [item_id for item_id, _ in reranked_pairs]


def retrieve_rankings(query: str, keyword_search_strategy: ItemNameSearchStrategy, dense_search_strategy: DenseSearch,
    xgb_model: xgb.Booster | None, executor: ThreadPoolExecutor, k: int) -> dict[str, list[str]]:
    candidate_pool_size = max(k, 50)
    keyword_future = executor.submit(
        get_bm25_search_results, keyword_search_strategy, query, candidate_pool_size
    )
    dense_future = executor.submit(
        get_dense_search_results, dense_search_strategy, query, candidate_pool_size
    )
    keyword_results = keyword_future.result()
    dense_results = dense_future.result()

    # RRF fusion
    fused_pairs = rrf.reciprocal_rank_fusion(
        [keyword_results, dense_results], top_k=candidate_pool_size
    )
    fused_ids = [item_id for item_id, _ in fused_pairs]

    rankings = {
        "BM25": keyword_results[:k],
        "Dense": dense_results[:k],
        "Fused": fused_ids[:k],
    }

    if xgb_model is not None:
        xgb_reranked = rerank_with_xgboost(query, fused_ids, xgb_model, dense_search_strategy)
        rankings["XGB_Reranked"] = xgb_reranked[:k]

    return rankings


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

    xgb_model = None
    if MODEL_PATH.exists():
        xgb_model = xgb.Booster()
        xgb_model.load_model(str(MODEL_PATH))
        print(f"Loaded trained XGBoost model from {MODEL_PATH}")
    else:
        print(f"XGBoost model file not found at {MODEL_PATH}, skipping reranking.")

    detail_rows = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        for query_number, query in enumerate(
            selected_queries.itertuples(index=False), start=1
        ):
            rankings = retrieve_rankings(
                query.query_string,
                keyword_search_strategy,
                dense_search_strategy,
                xgb_model,
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