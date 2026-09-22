from pathlib import Path
from strategies.BM25.item_name_search_strategy import ItemNameSearchStrategy
from strategies.DenseSearch.dense_search_on_description import DenseSearch
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
import psutil, os
from collections import defaultdict

ITEM_NAME_INDEX_DIR = Path(__file__).parent / "indexes" / "item_names"
ITEM_DATA_DIR = Path(__file__).parent / "data"
items_data = pd.read_csv(ITEM_DATA_DIR / "items.csv")

def reciprocal_rank_fusion(result_lists: list[list[str]], rank_constant: int = 60,  top_k: int = 10) -> list[str]:
    scores: dict[str, float] = {}

    for result_list in result_lists:
        for rank, item_id in enumerate(result_list, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + (1.0 / (rank_constant + rank))

    return [
        item_id
        for item_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True,)[:top_k]
    ]

#def get_ndcg_score(item_ids: list[str]):
   

if __name__ == "__main__":

  query = "tender kathal biryany with basmati rice"
  keywordSearchStrategy = ItemNameSearchStrategy()
  denseSearchStrategy = DenseSearch()

  process = psutil.Process(os.getpid())
  process.cpu_percent()  # first call primes the measurement, discard it
  mem_before = process.memory_info().rss
  start_time = time.perf_counter()

  with ThreadPoolExecutor(max_workers=2) as executor:
    future_keyword_results = executor.submit(keywordSearchStrategy.bm25_search_on_item_name, query, 50)
    future_dense_results = executor.submit(denseSearchStrategy.dense_search_on_item_desc, query, 50)

  keyword_results = future_keyword_results.result()
  dense_results = future_dense_results.result()

  results = list(keyword_results + dense_results)
  reranked_ids = reciprocal_rank_fusion([keyword_results, dense_results], top_k=100)

  id_to_info_mapping = items_data.set_index("item_id")
  records = id_to_info_mapping.loc[reranked_ids]

  elapsed_time = time.perf_counter() - start_time
  cpu_percent = process.cpu_percent()
  mem_after = process.memory_info().rss

  print(f"time took for search, {elapsed_time:.3f}s")
  print(f"cpu usage: {cpu_percent:.1f}%")
  print(f"memory used: {(mem_after - mem_before) / (1024 * 1024):.2f} MB (rss delta)")
  print(f"peak rss: {mem_after / (1024 * 1024):.2f} MB")
  
  pd.set_option("display.max_rows", None)
  pd.set_option("display.max_columns", None)
  pd.set_option("display.max_colwidth", None)
  pd.set_option("display.width", None)

  print(records)