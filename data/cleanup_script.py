import json
import re
import pandas as pd


def clean_text(text: str) -> str:
  if pd.isna(text):
    return ''
  text = str(text).replace('\\"', '"').replace('\\', '')
  text = re.sub(r'<[^>]*>', '', text)
  text = text.strip('"\' ')
  text = re.sub(r'\s+', ' ', text)
  return text


def build_rich_document(row) -> str:

  name = clean_text(row.get('item_name', ''))
  category = clean_text(row.get('menu_category', ''))
  restaurant = clean_text(row.get('restaurant_name', ''))
  desc = clean_text(row.get('item_description', ''))

  if not desc or len(desc) < 3:
    desc = name

  rich_doc = (
      f'Item: {name} | Category: {category} | Restaurant: {restaurant} | Details:'
      f' {desc}'
  )
  return rich_doc


def process_item_catalog(input_csv='items.csv', output_csv='items_cleaned.csv'):
  df = pd.read_csv(input_csv)

  print(f'Processing {len(df)} items from catalog...')

  # Apply cleaning across text columns
  df['item_name_clean'] = df['item_name'].apply(clean_text)
  df['item_description_clean'] = df['item_description'].apply(clean_text)

  # Create the rich unified search document
  df['search_document'] = df.apply(build_rich_document, axis=1)

  # Save cleaned output
  df.to_csv(output_csv, index=False)
  print(f'Cleaned catalog saved successfully to {output_csv}!')
  return df


if __name__ == '__main__':
  df_cleaned = process_item_catalog()
  print('\nSample Cleaned Document:')
  print(df_cleaned['search_document'].iloc[0])