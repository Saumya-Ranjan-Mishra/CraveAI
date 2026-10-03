FROM python:3.11-slim

ARG VCS_URL
LABEL org.opencontainers.image.source=$VCS_URL

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1 \
  CRAVEAI_DATA_DIR=/mnt/search-assets/data \
  CRAVEAI_INDEX_DIR=/mnt/search-assets/indexes \
  CRAVEAI_TOKENIZER_PATH=/mnt/search-assets/models/tokenizer.json \
  CRAVEAI_ONNX_MODEL_PATH=/mnt/search-assets/models/embedding_model.onnx \
  CRAVEAI_RERANKER_PATH=/mnt/search-assets/indexes/xgboost_reranker_model/crave_ai_xgb_reranker.json

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
  && mkdir -p /app/data /app/models /app/ranking_utils /app/strategies/BM25 /app/strategies/DenseSearch
COPY api.py search.py ./
COPY data/extract_ranking_features.py ./data/extract_ranking_features.py
COPY ranking_utils/rrf.py ranking_utils/xgb_reranking.py ./ranking_utils/
COPY strategies/BM25/item_name_search_strategy.py ./strategies/BM25/
COPY strategies/DenseSearch/dense_search_on_description.py ./strategies/DenseSearch/
COPY models/embedding_model.py models/item_text.py ./models/

RUN useradd --create-home --uid 10001 app \
  && mkdir -p /app/indexes \
  && chown -R app:app /app
USER 10001:10001

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]