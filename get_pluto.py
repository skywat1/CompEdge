import re
import requests
import numpy as np
import pandas as pd

from config import Config

# Unit/apartment designators Geosupport won't accept as part of the street.
# Cut the street at the first one of these (or at a '#').
_UNIT_RE = re.compile(
    r'\s+(#|apt\b|unit\b|ph\b|penthouse\b|fl\b|floor\b|ste\b|suite\b|rm\b|room\b)',
    re.IGNORECASE)


def clean_street(street: str) -> str:
    """Drop apartment/unit suffixes so Geoclient can match the street."""
    return _UNIT_RE.split(street, maxsplit=1)[0].strip()

# Columns PLUTO returns; used to build an all-NaN row on failure
PLUTO_FIELDS = [
    'latitude', 'longitude']

def get_bbl(address: str, borough: str):
    try:
        house_num, street = address.split(' ', 1)
    except Exception as e:
        print('get_bbl:', e)
        return ''
    
    params = {
        'houseNumber': house_num,
        'street': clean_street(street),
        'borough': borough,
        'zip': '',
        'key': Config.GEOCLIENT_V2_API_KEY,
    }
    
    try:
        r = requests.get(
            'https://api.nyc.gov/geoclient/v2/address',
            params=params, timeout=3)
        r.raise_for_status()
        data = r.json()['address']
    except Exception as e:
        print('get_bbl:', e)
        return ''
    
    try:
        res = data['bblBoroughCode'] + data['bblTaxBlock'] + data['bblTaxLot'] 
    except Exception as e:
        print('get_bbl:', e, address)
        return ''
    
    return res


def pluto(bbl: str):
    """One PLUTO lookup by BBL -> pd.Series of all PLUTO_FIELDS.

    Returns an all-NaN Series (indexed by PLUTO_FIELDS) on any failure or
    empty bbl, so callers can assign/fillna into a DataFrame safely.
    """
    def num(val):
        """PLUTO strings -> float, NaN if missing/unparseable."""
        try:
            return float(val)
        except (TypeError, ValueError):
            return np.nan

    if not bbl:
        return pd.Series(np.nan, index=PLUTO_FIELDS)

    try:
        r = requests.get(
            'https://data.cityofnewyork.us/resource/64uk-42ks.json',
            params={'bbl': bbl}, timeout=3)
        r.raise_for_status()
        data = r.json()[0]
    except Exception as e:
        print('pluto:', e)
        return pd.Series(np.nan, index=PLUTO_FIELDS)

    if not isinstance(data, dict):
        print('pluto: Invalid json format')
        return pd.Series(np.nan, index=PLUTO_FIELDS)

    return pd.Series({
        'latitude': num(data.get('latitude')),
        'longitude': num(data.get('longitude')),
    })
    
    
df = pd.read_csv('data/cleaned_sold.csv')
    
# Fill missing data with PLUTO
# One API call per row -> a DataFrame with all PLUTO_FIELDS as columns
pluto_data = df.apply(
    lambda row: pluto(get_bbl(row['address'], row['borough'])),
    axis=1
)

df[PLUTO_FIELDS] = pluto_data[PLUTO_FIELDS]

df.to_csv('data/pluto_data.csv')