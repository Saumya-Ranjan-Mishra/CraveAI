import pandas as pd


def build_item_search_text(corpus: pd.DataFrame) -> list[str]:
    fields = ["item_name", "menu_category", "restaurant_name", "item_description"]
    values = corpus.reindex(columns=fields).fillna("").astype(str)
    return (
        "Item: "
        + values["item_name"]
        + " | Category: "
        + values["menu_category"]
        + " | Restaurant: "
        + values["restaurant_name"]
        + " | Desc: "
        + values["item_description"]
    ).tolist()