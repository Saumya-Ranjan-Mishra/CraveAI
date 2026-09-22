from tokenizers import Tokenizer
import onnxruntime as ort
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

class DenseSearch:
  def __init__(self):
    self.item_list_path = Path(__file__).parent.parent.parent / "data" / "items.csv"
    self.item_desc_embeddings_path = Path(__file__).parent.parent.parent / "indexes" / "item_desc_embeddings" / "embeddings.npy"

    self.tokenizer = Tokenizer.from_file(r"C:\Users\A1134913\Downloads\tokenizer.json")
    self.tokenizer.enable_padding()
    self.tokenizer.enable_truncation(max_length=512)

    self.embedding_inference_session = ort.InferenceSession(r"C:\Users\A1134913\Downloads\embedding_model.onnx", providers=["CPUExecutionProvider"])

    self.model_inputs = [inp.name for inp in self.embedding_inference_session.get_inputs()]
    self.model_outputs = [inp.name for inp in self.embedding_inference_session.get_outputs()]

  def get_sentence_emdeddings(self,texts: list[str], batch_size: int = 32) -> np.ndarray:

    all_embeddings = []

    for start in range(0, len(texts), batch_size):
      batch = texts[start:start + batch_size]
      encodings = self.tokenizer.encode_batch(batch)
      tokenizer_output = {
        "input_ids": np.asarray([encoding.ids for encoding in encodings], dtype=np.int64),
        "attention_mask": np.asarray(
          [encoding.attention_mask for encoding in encodings], dtype=np.int64
        ),
        "token_type_ids": np.asarray(
          [encoding.type_ids for encoding in encodings], dtype=np.int64
        ),
      }

      ort_inputs = {
        name: tokenizer_output[name]
        for name in self.model_inputs
        if name in tokenizer_output
      }

      model_output = self.embedding_inference_session.run(self.model_outputs, ort_inputs)
      token_embeddings = model_output[0]

      sentence_embeddings = self.mean_polling(token_embeddings, tokenizer_output["attention_mask"])
      print("sentense embedding shape", sentence_embeddings.shape)

      ## Normalize the final embedding so that we can calculate the cosine simlarity.
      norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
      normalized_embeddings = sentence_embeddings / np.clip(norms, a_min=1e-12, a_max=None)

      all_embeddings.append(normalized_embeddings)

    return np.concatenate(all_embeddings, axis=0)

  def mean_polling(self, token_embedding, attension_mask):
    mask_expanded = np.expand_dims(attension_mask, -1).astype(np.float32)
    sum_embeddings = np.sum(token_embedding * mask_expanded, axis=1) 
    sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None) 
    return sum_embeddings / sum_mask 

  def create_vector_index(self):
    if not self.item_desc_embeddings_path.exists():
      corpus = pd.read_csv(self.item_list_path)
      print("Creating vector embeddings for item descriptions")
      self.item_desc_embeddings_path.parent.mkdir(parents=True, exist_ok=True)
      doc_embeddings = self.get_sentence_emdeddings(corpus["item_description"].tolist())

      np.save(self.item_desc_embeddings_path, doc_embeddings)

  def dense_search_on_item_desc(self, query: str, k: int = 10) -> List[str]:
    if not self.item_desc_embeddings_path.exists():
      self.create_vector_index()

    item_ids = pd.read_csv(self.item_list_path)["item_id"].tolist()
    doc_embeddings = np.load(self.item_desc_embeddings_path)
    query_vector = self.get_sentence_emdeddings([query])[0]
    scores = doc_embeddings @ query_vector

    top_k_results = np.argsort(-scores)[:k]

    return [item_ids[i] for i in top_k_results]