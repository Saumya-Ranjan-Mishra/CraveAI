import xgboost as xgb
import pandas as pd
from pathlib import Path
from data import extract_ranking_features
from pandas import DataFrame

xgb_trained_model_path = Path(__file__).parent.parent / "indexes" / "xgboost_reranker_model" / "crave_ai_xgb_reranker.json"

def rerank_search_results(query: str, candidates_df: DataFrame) -> DataFrame:
  model = xgb.Booster()
  model.load_model(xgb_trained_model_path)

  if "item_id" not in candidates_df.columns:
    candidates_df = candidates_df.reset_index()

  candidates_df["dense_score"] = candidates_df["rrf_score"]
  candidates_df["name_token_overlap"] = candidates_df.apply(
    lambda row: extract_ranking_features.compute_token_overlap(query, str(row["item_name"])),
    axis=1,
  )

  candidates_df["category_token_overlap"] = candidates_df.apply(
    lambda row: extract_ranking_features.compute_token_overlap(query, str(row.get("menu_category", ""))),
    axis=1,
  )

  candidates_df["desc_token_overlap"] = candidates_df.apply(
    lambda row: extract_ranking_features.compute_token_overlap(query, str(row.get("item_description", ""))),
    axis=1,
  )

  candidates_df["candidate_rank"] = range(1, len(candidates_df) + 1)

  candidates_df["item_rating"] = pd.to_numeric(candidates_df.get("item_rating", 4.0), errors="coerce").fillna(4.0)

  feature_cols = [
      'dense_score',
      'name_token_overlap',
      'category_token_overlap',
      'desc_token_overlap',
      'candidate_rank',
      'item_rating',
  ]

  x_inference = candidates_df[feature_cols]
  dmatrix_inference = xgb.DMatrix(x_inference)

  scores = model.predict(dmatrix_inference)
  candidates_df['rerank_score'] = scores

  available_cols = [
      'item_id', 'restaurant_id', 'item_name', 'item_description',
      'item_rating', 'menu_category',
      'menu_price', 'restaurant_name', 'restaurant_description',
      'restaurant_address', 'city', 'rerank_score'
  ]
  selected_cols = [col for col in available_cols if col in candidates_df.columns]
  reranked_results = candidates_df.sort_values(by='rerank_score', ascending=False)[selected_cols]
  print("from xgb...")
  print(reranked_results)
  return reranked_results


