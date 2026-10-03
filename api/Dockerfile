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
  && mkdir -p /app/api/data /app/api/dtos /app/api/models /app/api/ranking_utils \
  /app/api/strategies/BM25 /app/api/strategies/DenseSearch
COPY __init__.py app.py search.py ./api/
COPY data/__init__.py data/extract_ranking_features.py ./api/data/
COPY dtos/__init__.py dtos/search_result.py ./api/dtos/
COPY models/__init__.py models/embedding_model.py models/item_text.py ./api/models/
COPY ranking_utils/__init__.py ranking_utils/rrf.py ranking_utils/xgb_reranking.py ./api/ranking_utils/
COPY strategies/__init__.py ./api/strategies/
COPY strategies/BM25/__init__.py strategies/BM25/item_name_search_strategy.py ./api/strategies/BM25/
COPY strategies/DenseSearch/__init__.py strategies/DenseSearch/dense_search_on_description.py ./api/strategies/DenseSearch/

RUN useradd --create-home --uid 10001 app \
  && mkdir -p /app/indexes \
  && chown -R app:app /app
USER 10001:10001

EXPOSE 8000
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]