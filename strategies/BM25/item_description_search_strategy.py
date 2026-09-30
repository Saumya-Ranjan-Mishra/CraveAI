import pandas as pd
from pathlib import Path
import bm25s
from models import SearchResult
from typing import List

ITEM_DATA_DIR = Path(__file__).parent.parent.parent / "data" 
ITEM_DESC_INDEX_PATH = Path(__file__).parent.parent.parent / "indexes" / "item_description"
ITEM_ID_MAPPING_PATH = ITEM_DESC_INDEX_PATH / "item_id_description_mapping.txt"

items_data = pd.read_csv(ITEM_DATA_DIR / "items.csv")
class ItemDescriptionSearchStrategy:


  def create_index_for_item_description(self):
    item_ids = items_data["item_id"]
    item_desc = items_data["item_description"]

    tokens = bm25s.tokenize(item_desc, stopwords="en")
    retriever = bm25s.BM25()
    retriever.index(tokens)

    ITEM_DESC_INDEX_PATH.mkdir(parents=True, exist_ok=True)

    retriever.save(str(ITEM_DESC_INDEX_PATH))
    ITEM_ID_MAPPING_PATH.write_text("\n".join(item_ids))

  def load_item_desc_indexes(self):
    item_description_index_path = ITEM_DESC_INDEX_PATH
    retriever = bm25s.BM25.load(str(item_description_index_path), load_corpus=False)

    item_ids = ITEM_ID_MAPPING_PATH.read_text().splitlines()

    return retriever, item_ids

  def bm25_search_on_item_desc(self, query: str, k: int = 10) -> List[str]:

    params_path = ITEM_DESC_INDEX_PATH / "params.index.json"
    if not params_path.is_file() or not ITEM_ID_MAPPING_PATH.is_file():
      self.create_index_for_item_description()

    retriever, item_ids = self.load_item_desc_indexes()

    query_tokens = bm25s.tokenize(query, stopwords="en")
    indices = retriever.retrieve(query_tokens, k=k)

    matched_ids = [item_ids[i] for i in indices[0]]
    return matched_ids

