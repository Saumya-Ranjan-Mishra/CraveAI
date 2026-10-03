def reciprocal_rank_at_k(ranked_ids: list[str], relevant_ids: list[str], k: int = 10) -> float:
    if not relevant_ids:
        return 0.0

    relevant_id_set = set(relevant_ids)
    for rank, item_id in enumerate(ranked_ids[:k], start=1):
        if item_id in relevant_id_set:
            return 1.0 / rank
    return 0.0
  