def hit_rate_at_k(ranked_ids: list[str], relevant_ids: list[str], k: int = 10) -> float:
    if not relevant_ids:
        return 0.0

    relevant_id_set = set(relevant_ids)
    return float(any(item_id in relevant_id_set for item_id in ranked_ids[:k]))