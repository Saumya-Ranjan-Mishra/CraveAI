import pandas as pd
import json
from ranking_utils import rrf, xgb_reranking
from strategies.BM25.item_name_search_strategy import ItemNameSearchStrategy
from strategies.DenseSearch.dense_search_on_description import DenseSearch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
ITEM_DATA_DIR = PROJECT_DIR / "data"
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

if __name__ == "__main__":
    query = "craving for itilian food"
    results = get_search_results(query, 50)
    reranked_results = xgb_reranking.rerank_search_results(query, results)
    print(reranked_results.columns.tolist())
