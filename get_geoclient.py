import re
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.adapters import HTTPAdapter
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import Config

MAX_WORKERS = 12

_SESSION = requests.Session()
_SESSION.mount(
    'https://',
    HTTPAdapter(pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS))

def clean_street(street: str) -> str:
    # Unit/apartment designators Geosupport won't accept as part of the street.
    # Cut the street at the first one of these (or at a '#').
    _UNIT_RE = re.compile(
        r'\s+(#|apt\b|unit\b|ph\b|penthouse\b|fl\b|floor\b|ste\b|suite\b|rm\b|room\b)',
        re.IGNORECASE)
    
    """Drop apartment/unit suffixes so Geoclient can match the street."""
    return _UNIT_RE.split(street, maxsplit=1)[0].strip()

def _to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None

def get_geoclient(address: str, borough: str):
    EMPTY = {'bbl': None, 'latitude': None, 'longitude': None}
    
    try:
        house_num, street = address.split(' ', 1)
    except Exception as e:
        print('get_geoclient:', e, address)
        return EMPTY

    params = {
        'houseNumber': house_num,
        'street': clean_street(street),
        'borough': borough,
        'zip': '',
        'key': Config.GEOCLIENT_V2_API_KEY,
    }

    try:
        r = _SESSION.get(
            'https://api.nyc.gov/geoclient/v2/address',
            params=params, timeout=3)
        r.raise_for_status()
        data = r.json()['address']
    except Exception as e:
        print('get_geoclient:', e, address)
        return EMPTY

    bbl = None
    if data.get('bblBoroughCode') and data.get('bblTaxBlock') and data.get('bblTaxLot'):
        candidate = int(data['bblBoroughCode'] + data['bblTaxBlock'] + data['bblTaxLot'])
        if candidate:  # Geoclient returns 0000000000 for condos with no billing BBL yet
            bbl = candidate

    return {
        'bbl': bbl,
        'latitude': _to_float(data.get('latitude')),
        'longitude': _to_float(data.get('longitude')),
    }


if __name__ == '__main__':
    sold = pd.read_csv('data/cleaned_sold.csv')

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(
            executor.map(get_geoclient, sold['address'], sold['borough']),
            total=len(sold), desc='Geocoding'))

    geoclient_data = pd.DataFrame(results)
    geoclient_data.insert(0, 'zpid', sold['zpid'])
    geoclient_data['bbl'] = geoclient_data['bbl'].astype('Int64')

    geoclient_data.to_csv('data/geoclient.csv', index=False)
    print(f'Wrote data/geoclient.csv ({len(geoclient_data)} rows)')
