import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from models import embedding_model

class DenseSearch:
  def __init__(self):
    self.item_list_path = Path(__file__).parent.parent.parent / "data" / "items.csv"
    self.item_desc_embeddings_path = Path(__file__).parent.parent.parent / "indexes" / "item_desc_embeddings" / "embeddings.npy"
    self.embedding_model_obj = embedding_model.EmbeddingModel()

  def create_vector_index(self):
    if not self.item_desc_embeddings_path.exists():
      corpus = pd.read_csv(self.item_list_path)
      print("Creating vector embeddings for item descriptions")
      self.item_desc_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
      doc_embeddings = self.embedding_model_obj.get_sentence_emdeddings(corpus["item_description"].tolist())

      np.save(self.item_desc_embeddings_path, doc_embeddings)

  def dense_search_on_item_desc(self, query: str, k: int = 10) -> List[str]:
    if not self.item_desc_embeddings_path.exists():
      self.create_vector_index()

    item_ids = pd.read_csv(self.item_list_path)["item_id"].tolist()
    doc_embeddings = np.load(self.item_desc_embeddings_path)
    query_vector = self.embedding_model_obj.get_sentence_emdeddings([query])[0]
    scores = doc_embeddings @ query_vector

    top_k_results = np.argsort(-scores)[:k]

    return [item_ids[i] for i in top_k_results]