import pandas as pd
import json
import os
from ranking_utils import rrf, xgb_reranking
from strategies.BM25.item_name_search_strategy import ItemNameSearchStrategy
from strategies.DenseSearch.dense_search_on_description import DenseSearch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
ITEM_DATA_DIR = Path(os.environ.get("CRAVEAI_DATA_DIR", PROJECT_DIR / "data"))
MAX_TOP_K = int(os.environ.get("CRAVEAI_MAX_TOP_K", "100"))
items_by_id = pd.read_csv(ITEM_DATA_DIR / "items.csv").set_index("item_id")

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

def get_search_results(query: str, k : int) -> pd.DataFrame:
    keyword_search_strategy = ItemNameSearchStrategy()
    dense_search_strategy = DenseSearch()

    with ThreadPoolExecutor(max_workers=2) as executor:
        rankings = retrieve_rankings(query, keyword_search_strategy, dense_search_strategy, executor, k)
        fused_items = rankings["Fused"][:k]
        fused_ids = [item_id for item_id, _ in fused_items]
        fused_scores = [score for _, score in fused_items]

        detail_rows = items_by_id.loc[fused_ids].copy()
        detail_rows["rrf_score"] = fused_scores

    return detail_rows


class FoodSearchService:

    def __init__(self) -> None:
        self.items = pd.read_csv(ITEM_DATA_DIR / "items.csv").fillna("")
        self.items_by_id = self.items.set_index("item_id")
        self.keyword_search_strategy = ItemNameSearchStrategy()
        self.dense_search_strategy = DenseSearch()
        self.reranker = xgb_reranking.load_reranker_model()
        self.executor = ThreadPoolExecutor(max_workers=2)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if top_k < 1 or top_k > MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")

        candidate_k = min(max(top_k * 5, 50), len(self.items))
        rankings = retrieve_rankings(
            normalized_query,
            self.keyword_search_strategy,
            self.dense_search_strategy,
            self.executor,
            candidate_k,
        )
        fused_pairs = rankings["Fused"]
        if not fused_pairs:
            return []

        fused_ids = [item_id for item_id, _ in fused_pairs]
        fused_scores = [float(score) for _, score in fused_pairs]
        candidates = self.items_by_id.loc[fused_ids].copy().reset_index()
        candidates["rrf_score"] = fused_scores
        dense_scores = self.dense_search_strategy.score_item_ids(normalized_query, fused_ids)
        candidates["dense_score"] = [dense_scores.get(item_id, 0.0) for item_id in fused_ids]
        dense_ranks = {
            item_id: rank for rank, item_id in enumerate(rankings["Dense"], start=1)
        }
        bm25_ranks = {
            item_id: rank for rank, item_id in enumerate(rankings["BM25"], start=1)
        }
        candidates["candidate_rank"] = [
            dense_ranks.get(item_id, candidate_k + bm25_ranks.get(item_id, candidate_k))
            for item_id in fused_ids
        ]

        final_results = xgb_reranking.rerank_search_results(
            normalized_query,
            candidates,
            model=self.reranker,
        ).head(top_k)
        final_results = final_results.astype(object).where(pd.notna(final_results), None)
        return final_results.to_dict(orient="records")

    def close(self) -> None:
        self.executor.shutdown(wait=True)

if __name__ == "__main__":
    query = "craving for itilian food"
    results = get_search_results(query, 50)
    reranked_results = xgb_reranking.rerank_search_results(query, results)
    print(reranked_results.columns.tolist())
