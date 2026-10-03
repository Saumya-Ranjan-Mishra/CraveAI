from pathlib import Path
import os
import pandas as pd
import bm25s
from typing import List

PROJECT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = Path(os.environ.get("CRAVEAI_DATA_DIR", PROJECT_DIR / "data"))
INDEX_DIR = Path(os.environ.get("CRAVEAI_INDEX_DIR", PROJECT_DIR / "indexes"))
items_corpus_path = DATA_DIR / "items.csv"
restaurant_corpus_path = DATA_DIR / "restaurants.csv"
ITEM_ID_MAPPING_FILENAME = "items_name_id_mapping.txt"

class ItemNameSearchStrategy:
  def __init__(self):
    self.items_data = pd.read_csv(items_corpus_path).fillna("")
    item_names_index_path = INDEX_DIR / "item_names"
    mapping_path = item_names_index_path / ITEM_ID_MAPPING_FILENAME
    if not (item_names_index_path / "params.index.json").is_file() or not mapping_path.is_file():
      self.create_index_on_item_name()
    self.retriever, self.item_ids = self.load_item_name_indexes()

  def create_index_on_item_name(self):
    item_names_index_path = INDEX_DIR/ "item_names"
    items_ids = self.items_data["item_id"].astype(str)
    item_names = self.items_data["item_name"].astype(str)
    tokens = bm25s.tokenize(item_names, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(tokens)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    retriever.save(str(item_names_index_path))
    (item_names_index_path / ITEM_ID_MAPPING_FILENAME).write_text("\n".join(items_ids))

  def load_item_name_indexes(self):
    item_names_index_path = INDEX_DIR/ "item_names"
    retriever = bm25s.BM25.load(str(item_names_index_path), load_corpus=False)

    item_id_mapping_path = item_names_index_path / ITEM_ID_MAPPING_FILENAME
    item_ids = item_id_mapping_path.read_text().splitlines()

    return retriever, item_ids

  def bm25_search_on_item_name(self, query: str, k: int = 10) -> List[str]:
    query_tokens = bm25s.tokenize(query, stopwords="en")
    indices, _ = self.retriever.retrieve(query_tokens, k=k)

    matched_ids = [self.item_ids[i] for i in indices[0]]
    return matched_ids
