import pandas as pd
import numpy as np
import re
import json
from bs4 import BeautifulSoup

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

# If no borough, coerce to Brookyln
# ONLY BECAUSE DATA IS BROOKLYN ONLY. CHANGE WHEN INCOORPERATING
# OTHER BOROUGHS
_BOROUGHS = {'Manhattan', 'Bronx', 'Brooklyn', 'Queens', 'Staten Island'}
df['borough'] = df['borough'].where(df['borough'].isin(_BOROUGHS), 'Brooklyn')

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

# Drop non-arm's-length sales (deed/estate transfers) and price outliers
MIN_PRICE, MAX_PRICE = 100_000, 10_000_000
df['sold_price'] = pd.to_numeric(df['sold_price'], errors='coerce')
print('Rows with sale price outside range:')
out_of_range = (df['sold_price'] < MIN_PRICE) | (df['sold_price'] > MAX_PRICE)
for href in df.loc[out_of_range, 'listing-link-href']:
    print(href)
df = df[~out_of_range]

# Null out impossible sqft values (e.g. co-ops listing the whole building's
# footage) and lot_area for units that don't own their lot. Must happen before
# the $/sqft filter so junk sqft doesn't produce a false-low $/sqft.
_UNIT_TYPES = ['Condo', 'Cooperative']
bad_sqft = (df['area-sqft'] < 100) | (df['area-sqft'] > 20_000) | \
           (df['type'].isin(_UNIT_TYPES) & (df['area-sqft'] > 10_000))
df.loc[bad_sqft, 'area-sqft'] = np.nan
df.loc[df['type'].isin(_UNIT_TYPES), 'lot_area'] = np.nan

# Drop non-arm's-length sales the price range missed (rows with missing sqft
# pass through: NaN comparisons are False)
MIN_PRICE_PER_SQFT = 100
print('Rows with price per sqft below minimum:')
below_ppsf = (df['sold_price'] / df['area-sqft']) < MIN_PRICE_PER_SQFT
for href in df.loc[below_ppsf, 'listing-link-href']:
    print(href)
df = df[~below_ppsf]

# Add days_old column
reference_date = pd.Timestamp('2026-07-02')
df['days_old'] = (reference_date - pd.to_datetime(df['sold_date'], format='%m/%d/%Y', errors='coerce')).dt.days


#Dedup image links
def dedup_images(cell):
    if pd.isna(cell):
        return cell
    links = [link.strip() for link in cell.split(',') if link.strip()]
    seen = set()
    unique = []
    for link in links:
        if link not in seen:
            seen.add(link)
            unique.append(link)
    return ','.join(unique)

df['all_images'] = df['all_images'].apply(dedup_images)


# Output cleaned data
df.to_csv('data/cleaned_sold.csv', index=False)