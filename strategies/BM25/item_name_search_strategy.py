from pathlib import Path
import pandas as pd
import bm25s
from typing import List

DATA_DIR = Path(__file__).parent.parent.parent / "data"
INDEX_DIR = Path(__file__).parent.parent.parent / "indexes"
items_corpus_path = DATA_DIR / "items.csv"
restaurant_corpus_path = DATA_DIR / "restaurants.csv"
items_data =  pd.read_csv(items_corpus_path)

class ItemNameSearchStrategy:

  def create_index_on_item_name(self):
    item_names_index_path = INDEX_DIR/ "item_names"
    items_ids =  items_data["item_id"]
    item_names = items_data["item_name"]
    tokens = bm25s.tokenize(item_names, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(tokens)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    retriever.save(str(item_names_index_path))
    (item_names_index_path / "items_name_id_mapping.txt").write_text("\n".join(items_ids))

  def load_item_name_indexes(self):
    item_names_index_path = INDEX_DIR/ "item_names"
    retriever = bm25s.BM25.load(str(item_names_index_path), load_corpus=False)

    item_id_mapping_path = item_names_index_path / "items_name_id_mapping.txt"
    item_ids = item_id_mapping_path.read_text().splitlines()

    return retriever, item_ids

  def bm25_search_on_item_name(self, query: str, k: int = 10) -> List[str]:

    if not (INDEX_DIR / "item_names").exists():
      self.create_index_on_item_name()

    retriever, item_ids = self.load_item_name_indexes()

    query_tokens = bm25s.tokenize(query, stopwords="en")
    indices, scores = retriever.retrieve(query_tokens, k=k)

    matched_ids = [item_ids[i] for i in indices[0]]
    return matched_ids
