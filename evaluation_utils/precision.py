def precision_at_k(ranked_items:list[str], relevant_items: list[str], k : int = 10) -> float:
  if not relevant_items:
    return 0.0

  relevant_items_set = set(relevant_items)
  ranked_items_set = set(ranked_items[:k])

  total_mathced_items = (relevant_items_set & ranked_items_set)

  return float(len(total_mathced_items)/k)

  