def recall_at_k(ranked_ids: list[str], relevant_ids: list[str], k: int = 10) -> float:
  if not relevant_ids:
    return 0.0

  _top_k = set(ranked_ids[:k])
  _relevant = set(relevant_ids)
  hits = len(_top_k & _relevant)
  return float(hits / len(_relevant))