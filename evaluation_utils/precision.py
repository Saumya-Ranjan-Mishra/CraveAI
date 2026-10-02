def precision_at_k(ranked_items:list[str], relevant_items: list[str], k : int = 10) -> float:
  top_k_items = ranked_items[:k]
  if not relevant_items or not top_k_items:
    return 0.0

  relevant_items_set = set(relevant_items)
  total_matched_items = len(relevant_items_set & set(top_k_items))

  return float(total_matched_items / len(top_k_items))

  