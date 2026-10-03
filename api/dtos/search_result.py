from dataclasses import dataclass

@dataclass
class SearchResult:
  item_id: str
  item_name: str
  item_desc: str
  score: float