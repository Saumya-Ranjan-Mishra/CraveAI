import pandas as pd
import xgboost as xgb
from pathlib import Path

feature_csv = Path(__file__).parent.parent / "data" / "xgboost_ranking_train_data.csv"

def train_reranker():
  print('Loading ranking feature dataset...')
  df = pd.read_csv(feature_csv)

  # Define features and target
  feature_cols = [
      'dense_score',
      'name_token_overlap',
      'category_token_overlap',
      'desc_token_overlap',
      'candidate_rank',
      'item_rating',
  ]
  target_col = 'relevance_label'
  group_col = 'query_id'

  # Split queries into train and validation sets to prevent data leakage across queries
  unique_queries = df['query_id'].unique()
  train_queries = set(unique_queries[: int(len(unique_queries) * 0.7)])
  val_queries = set(unique_queries[int(len(unique_queries) * 0.7) : int(len(unique_queries) * 0.9)])

  train_df = df[df['query_id'].isin(train_queries)]
  val_df = df[df['query_id'].isin(val_queries)]

  def prepare_dmatrix(data):
    data = data.sort_values('query_id')
    X = data[feature_cols]
    y = data[target_col]
    groups = data.groupby('query_id', sort=False).size().to_numpy()

    dmatrix = xgb.DMatrix(X, label=y)
    dmatrix.set_group(groups)
    return dmatrix

  dtrain = prepare_dmatrix(train_df)
  dval = prepare_dmatrix(val_df)

  params = {
      'eval_metric': 'ndcg@10',
      'objective': 'rank:ndcg',
      'learning_rate': 0.1,
      'max_depth': 5,
      'min_child_weight': 1,
      'subsample': 0.8,
      'colsample_bytree': 0.8,
  }

  print('Training XGBoost LambdaRank model...')
  evals = [(dtrain, 'train'), (dval, 'val')]
  model = xgb.train(
      params=params,
      dtrain=dtrain,
      num_boost_round=100,
      evals=evals,
      early_stopping_rounds=20,
  )

  # Save trained model
  model.save_model(Path(__file__).parent.parent / "indexes"/ "xgboost_reranker_model" / "crave_ai_xgb_reranker.json")
  print(
      'Model training complete! Saved trained reranker as'
      ' aurarank_xgb_reranker.json'
  )
  return model


if __name__ == '__main__':
  train_reranker()