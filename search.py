from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

import pandas as pd
import xgboost as xgb

from ranking_utils import rrf, xgb_reranking
from strategies.BM25.item_name_search_strategy import ItemNameSearchStrategy
from strategies.DenseSearch.dense_search_on_description import DenseSearch


PROJECT_DIR = Path(__file__).parent
ITEM_DATA_DIR = Path(os.environ.get("CRAVEAI_DATA_DIR", PROJECT_DIR / "data"))
MAX_TOP_K = int(os.environ.get("CRAVEAI_MAX_TOP_K", "100"))


class FoodSearchService:
    def __init__(self) -> None:
        self.items = pd.read_csv(ITEM_DATA_DIR / "items.csv").fillna("")
        self.items_by_id = self.items.set_index("item_id")
        self.keyword_search = ItemNameSearchStrategy()
        self.dense_search = DenseSearch()
        self.reranker: xgb.Booster = xgb_reranking.load_reranker_model()
        self.executor = ThreadPoolExecutor(max_workers=2)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        if top_k < 1 or top_k > MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")

        candidate_k = min(max(top_k * 5, 50), len(self.items))
        bm25_future = self.executor.submit(self.keyword_search.bm25_search_on_item_name, normalized_query, candidate_k)
        dense_future = self.executor.submit(self.dense_search.dense_search_on_item_desc, normalized_query, candidate_k)
        bm25_ids = bm25_future.result()
        dense_ids = dense_future.result()
        fused_pairs = rrf.reciprocal_rank_fusion([bm25_ids, dense_ids], top_k=candidate_k)

        if not fused_pairs:
            return []

        candidate_ids = [item_id for item_id, _ in fused_pairs]
        rrf_scores = [float(score) for _, score in fused_pairs]
        candidates = self.items_by_id.loc[candidate_ids].copy().reset_index()
        candidates["rrf_score"] = rrf_scores
        dense_scores = self.dense_search.score_item_ids(normalized_query, candidate_ids)
        candidates["dense_score"] = [dense_scores.get(item_id, 0.0) for item_id in candidate_ids]
        dense_ranks = {item_id: rank for rank, item_id in enumerate(dense_ids, start=1)}
        bm25_ranks = {item_id: rank for rank, item_id in enumerate(bm25_ids, start=1)}
        candidates["candidate_rank"] = [
            dense_ranks.get(item_id, candidate_k + bm25_ranks.get(item_id, candidate_k))
            for item_id in candidate_ids
        ]

        final_results = xgb_reranking.rerank_search_results(normalized_query, candidates, model=self.reranker).head(top_k)
        final_results = final_results.astype(object).where(pd.notna(final_results), None)
        return final_results.to_dict(orient="records")

    def close(self) -> None:
        self.executor.shutdown(wait=True)