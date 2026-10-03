from tokenizers import Tokenizer
import onnxruntime as ort
import numpy as np
import os
from pathlib import Path

class EmbeddingModel:
  def __init__(self):
    tokenizer_path = Path(os.environ.get("CRAVEAI_TOKENIZER_PATH"))
    model_path = Path(os.environ.get("CRAVEAI_ONNX_MODEL_PATH"))

    self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
    self.tokenizer.enable_padding()
    self.tokenizer.enable_truncation(max_length=512)

    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = int(os.environ.get("CRAVEAI_ORT_INTRA_OP_THREADS", "1"))
    session_options.inter_op_num_threads = int(os.environ.get("CRAVEAI_ORT_INTER_OP_THREADS", "1"))
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    self.embedding_inference_session = ort.InferenceSession(
      str(model_path),
      sess_options=session_options,
      providers=["CPUExecutionProvider"],
    )

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
      #print("sentense embedding shape", sentence_embeddings.shape)

      norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
      normalized_embeddings = sentence_embeddings / np.clip(norms, a_min=1e-12, a_max=None)

      all_embeddings.append(normalized_embeddings)

    return np.concatenate(all_embeddings, axis=0)

  def mean_polling(self, token_embedding, attension_mask):
    mask_expanded = np.expand_dims(attension_mask, -1).astype(np.float32)
    sum_embeddings = np.sum(token_embedding * mask_expanded, axis=1) 
    sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None) 
    return sum_embeddings / sum_mask 
