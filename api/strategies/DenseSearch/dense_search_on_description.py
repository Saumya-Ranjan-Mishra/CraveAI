import numpy as np
import os
import pandas as pd
from pathlib import Path
from typing import List
from api.models import embedding_model
from api.models.item_text import build_item_search_text

class DenseSearch:
  def __init__(self):
    project_dir = Path(__file__).parent.parent.parent
    data_dir = Path(os.environ.get("CRAVEAI_DATA_DIR", project_dir / "data"))
    index_dir = Path(os.environ.get("CRAVEAI_INDEX_DIR", project_dir / "indexes"))
    self.item_list_path = data_dir / "items.csv"
    self.item_desc_embeddings_path = index_dir / "item_desc_embeddings" / "embeddings.npy"
    self.embedding_model_obj = embedding_model.EmbeddingModel()
    self.items_data = pd.read_csv(self.item_list_path).fillna("")
    self.item_ids = self.items_data["item_id"].astype(str).tolist()
    self.item_id_to_index = {item_id: i for i, item_id in enumerate(self.item_ids)}

    if not self.item_desc_embeddings_path.is_file():
      self.create_vector_index()
    self.doc_embeddings = np.load(self.item_desc_embeddings_path)
    if self.doc_embeddings.shape[0] != len(self.item_ids):
      raise ValueError("Embedding index row count does not match items.csv; rebuild the dense index.")

  def create_vector_index(self):
    embedding_texts = build_item_search_text(self.items_data)
    print("Creating dense item embeddings")
    self.item_desc_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    doc_embeddings = self.embedding_model_obj.get_sentence_emdeddings(embedding_texts)
    np.save(self.item_desc_embeddings_path, doc_embeddings)

  def score_item_ids(self, query: str, item_ids: list[str]) -> dict[str, float]:
    query_vector = self.embedding_model_obj.get_sentence_emdeddings([query])[0]
    scores: dict[str, float] = {}
    for item_id in item_ids:
      index = self.item_id_to_index.get(item_id)
      if index is not None:
        scores[item_id] = float(self.doc_embeddings[index] @ query_vector)
    return scores

  def dense_search_on_item_desc(self, query: str, k: int = 10) -> List[str]:
    query_vector = self.embedding_model_obj.get_sentence_emdeddings([query])[0]
    scores = self.doc_embeddings @ query_vector

    top_k_results = np.argsort(-scores)[:k]

    return [self.item_ids[i] for i in top_k_results]