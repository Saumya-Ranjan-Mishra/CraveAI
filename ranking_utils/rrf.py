def reciprocal_rank_fusion(result_lists: list[list[str]], rank_constant: int = 60, top_k: int = 10) -> list[str]:
    scores: dict[str, float] = {}

    for result_list in result_lists:
        seen: set[str] = set()
        for rank, item_id in enumerate(result_list, start=1):
            if item_id in seen:
                continue
            seen.add(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rank_constant + rank)

    return [
        item_id
        for item_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    ]