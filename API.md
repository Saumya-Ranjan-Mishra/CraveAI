# Search API

The API returns the final XGBoost-reranked item records only. Each result contains the catalog fields plus `dense_score`, `rrf_score`, and `rerank_score`.

## Local run

Install dependencies and set paths to the ONNX model and tokenizer:

```powershell
python -m pip install -r api/requirements.txt
$env:CRAVEAI_TOKENIZER_PATH = "C:\path\to\tokenizer.json"
$env:CRAVEAI_ONNX_MODEL_PATH = "C:\path\to\embedding_model.onnx"
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

The API loads the BM25 index, dense embeddings, ONNX session, catalog, and XGBoost model once per worker during startup. First startup may create the BM25 and dense indexes. Search requests then reuse those resources.

## Request

`POST /search`

```json
{
  "query": "craving for Italian food",
  "top_k": 10
}
```

The response is a JSON array of final XGBoost-ranked item records. `top_k` is limited to 1 through 100. `/health` provides a process liveness/readiness response; model loading happens before the app is ready to serve requests.

## Kubernetes

Build the image with `docker build -t craveai-search:latest -f api/Dockerfile api`. The manifest at `k8s/craveai-search.yaml` mounts the data, indexes, and model assets from Blob Storage. Build/push the image to a registry accessible by the cluster, update the image reference, then apply the manifest.

The BM25 and versioned dense indexes are created under `/app/indexes` at startup. For multiple replicas, prebuild and distribute these indexes or use a writable shared/persistent index volume; otherwise each replica builds its own local indexes on first start.
