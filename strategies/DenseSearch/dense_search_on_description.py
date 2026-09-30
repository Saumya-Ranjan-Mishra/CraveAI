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

      embedding_texts = (
            "Name: " + corpus["item_name"].astype(str)
            + ". Type: " + corpus["item_type"].astype(str)
            + ". Cuisine: " + corpus["cuisine"].astype(str)
            + ". Description: " + corpus["item_description"].astype(str)
      ).tolist()

      # descriptions = corpus["item_description"].fillna("").astype(str).str.strip()
      # item_names = corpus["item_name"].fillna("").astype(str).str.strip()
      # embedding_texts = descriptions.where(descriptions.ne(""), item_names).tolist()

      print("Creating vector embeddings for item descriptions")
      self.item_desc_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
      doc_embeddings = self.embedding_model_obj.get_sentence_emdeddings(embedding_texts)

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