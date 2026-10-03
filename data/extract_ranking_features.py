import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from models import embedding_model
from models.item_text import build_item_search_text

query_csv = Path(__file__).parent / "user_queries_with_history_and_clicks.csv"
items_csv = Path(__file__).parent / "items.csv"
output_path = Path(__file__).parent / "xgboost_ranking_train_data.csv"

def compute_token_overlap(query: str, text: str) -> float:
  """Calculates normalized token overlap ratio between query and item text."""
  if not isinstance(text, str) or not text.strip():
    return 0.0
  q_tokens = set(re.findall(r'\w+', query.lower()))
  t_tokens = set(re.findall(r'\w+', text.lower()))
  if not q_tokens:
    return 0.0
  return len(q_tokens.intersection(t_tokens)) / len(q_tokens)


def extract_features_for_reranking(
    max_queries=200,
    top_k_candidates=50,
):
  print('Loading datasets...')
  df_queries = pd.read_csv(query_csv).head(max_queries)
  df_items = pd.read_csv(items_csv).fillna('')

  # Create rich document string for items (incorporating name, category, restaurant)
  df_items['search_doc'] = build_item_search_text(df_items)

  item_lookup = df_items.set_index('item_id').to_dict(orient='index')
  all_item_ids = df_items['item_id'].tolist()

  print('Initializing embedding model for dense candidate scoring...')
  model = embedding_model.EmbeddingModel()
  item_embeddings = model.get_sentence_emdeddings(df_items['search_doc'].tolist())

  training_rows = []

  print(
      f'Extracting ranking features for {len(df_queries)} queries (top'
      f' {top_k_candidates} candidates per query)...'
  )
  for idx, row in df_queries.iterrows():
    q_id = row['query_id']
    query_str = row['query_string']
    relevant_ids = set(json.loads(row['response_item_ids']))

    q_emb = model.get_sentence_emdeddings([query_str])[0]
    cos_scores = item_embeddings @ q_emb

    # Get top K candidates based on dense score
    top_indices = np.argsort(cos_scores)[::-1][:top_k_candidates]

    for rank, item_idx in enumerate(top_indices):
      item_id = all_item_ids[item_idx]
      item_info = item_lookup[item_id]

      # Relevance Label: 1 if item is in ground-truth relevant set, else 0
      label = 1 if item_id in relevant_ids else 0

      # Feature Engineering
      dense_score = float(cos_scores[item_idx])
      name_overlap = compute_token_overlap(query_str, item_info['item_name'])
      cat_overlap = compute_token_overlap(query_str, item_info['menu_category'])
      desc_overlap = compute_token_overlap(query_str, item_info['item_description'])

      # Parse item rating safely
      try:
        rating = float(item_info['item_rating'])
      except ValueError:
        rating = 4.0  # Default fallback

      training_rows.append({
          'query_id': q_id,
          'item_id': item_id,
          'relevance_label': label,
          'candidate_rank': rank + 1,
          'dense_score': dense_score,
          'name_token_overlap': name_overlap,
          'category_token_overlap': cat_overlap,
          'desc_token_overlap': desc_overlap,
          'item_rating': rating,
      })

  df_train = pd.DataFrame(training_rows)
  df_train.to_csv(output_path, index=False)
  print(
      f'Feature extraction complete! Saved {len(df_train)} training samples to'
      f' {output_path}.'
  )
  return df_train


if __name__ == '__main__':
  df_features = extract_features_for_reranking()
  print('\nSample Extracted Features:')
  print(df_features.head(3))