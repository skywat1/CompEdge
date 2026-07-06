import pandas as pd
import numpy as np
import re
import json
from bs4 import BeautifulSoup

from fetch_missing import get_bbl, pluto

# Read Data
df = pd.read_csv('data/raw_sold.csv')

# zpid as primary key
df['zpid'] = df['listing-link-href'].str.extract(r'/(\d+)_zpid')

# Drop meta data columns
# Also drop price because we rederive from price history 
df = df.drop(columns=['web-scraper-start-url', 'listing-link', 'results-for',
                      'public_tax_history_property_taxes_percentage_change',
                      'public_tax_history_tax_assessment_percentage_change',
                      'price_history_percentage_change', 'price',
                      'price_history_merged'
                      ])

# Rename columns
df = df.rename(columns={'area': 'lot_area'})

# Clean formatting and change types
df['address'] = df['address'].str.split(',').str[0].str.strip()

df['area-sqft'] = pd.to_numeric(
    df['area-sqft'].str.replace(',', '', regex=False),
    errors='coerce'
)

df['lot_area'] = pd.to_numeric(
    df['lot_area']
        .str.replace(pat={',': '', 'Lot': ''}, regex=False)
        .str.strip(),
    errors='coerce'
)

df['built_in'] = pd.to_numeric(df['built_in'], errors='coerce')
df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce')


# Change facts_and_features html to JSON format
def parse_facts_html(html):
    """Zillow 'Facts & features' HTML -> {category: {subsection: {label: value}}}."""
    if not html or "<" not in str(html):
        return {}
    soup = BeautifulSoup(str(html), "html.parser")
    result = {}
    for cg in soup.select('[data-testid="category-group"]'):
        cat_el = cg.select_one("h3")
        category = cat_el.get_text(strip=True) if cat_el else "_uncategorised"
        result.setdefault(category, {})
        current_sub = None
        for el in cg.select("h6, li"):
            if el.name == "h6":
                current_sub = el.get_text(strip=True)
                result[category].setdefault(current_sub, {})
            else:  # <li> bullet
                if current_sub is None:
                    current_sub = "_general"
                    result[category].setdefault(current_sub, {})
                text = el.get_text(" ", strip=True)
                if ":" in text:
                    k, v = text.split(":", 1)
                    result[category][current_sub][k.strip()] = v.strip()
                elif text:
                    result[category][current_sub].setdefault("_bare", []).append(text)
    return result

df['facts_features_text'] = df['facts_features_text'].apply(parse_facts_html)


# Derive price history info
def parse_history_list(cell, key):
    """JSON string of [{key: value}, ...] -> [value, ...]."""
    if not isinstance(cell, str) or not cell.strip():
        return []
    try:
        return [d.get(key) for d in json.loads(cell)]
    except (json.JSONDecodeError, AttributeError):
        return []

def derive_sold(row):
    events = parse_history_list(row['price_history_event'], 'price_history_event')
    prices = parse_history_list(row['price_history_price'], 'price_history_price')
    dates = parse_history_list(row['price_history_date'], 'price_history_date')
    for i, event in enumerate(events):
        if event == 'Sold':
            price = prices[i] if i < len(prices) else None
            date = dates[i] if i < len(dates) else None
            if price and date:
                return pd.Series({'sold_price': price, 'sold_date': date})
            break
    return pd.Series({'sold_price': None, 'sold_date': None})

df[['sold_price', 'sold_date']] = df.apply(derive_sold, axis=1)

# Drop rows missing a valid Sold price/date and report them
print('Rows with missing sale price:')
missing_sold = df['sold_price'].isna() | df['sold_date'].isna()
for href in df.loc[missing_sold, 'listing-link-href']:
    print(href)
df = df[~missing_sold]

# Add days_old column
reference_date = pd.Timestamp('2026-07-02')
df['days_old'] = (reference_date - pd.to_datetime(df['sold_date'], format='%m/%d/%Y', errors='coerce')).dt.days

# Output cleaned data
df.to_csv('data/cleaned_sold.csv', index=False)