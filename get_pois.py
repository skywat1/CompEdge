from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

MAX_WORKERS = 4  # ohsome asks clients to keep parallelism low
CHUNK_SIZE = 500  # circles per request
RADIUS_M = 600
OHSOME_URL = 'https://api.ohsome.org/v1/elements/count/groupBy/boundary'

POI_CATEGORIES = {
    "dining":        ["amenity=restaurant", "amenity=cafe", "amenity=ice_cream",
                      "shop=ice_cream", "amenity=food_court", "shop=coffee", "shop=tea"],

    "fast_food":     ["amenity=fast_food"],

    "nightlife":     ["amenity=bar", "amenity=pub", "amenity=biergarten",
                      "amenity=nightclub"],

    "groceries":     ["shop=supermarket", "shop=convenience", "shop=deli",
                      "shop=greengrocer", "shop=butcher", "shop=seafood",
                      "shop=bakery", "shop=pastry", "shop=confectionery",
                      "shop=chocolate", "shop=cheese", "shop=health_food",
                      "shop=beverages", "shop=wine", "shop=alcohol",
                      "shop=grocery", "shop=farm", "amenity=marketplace"],

    "retail":        ["shop=clothes", "shop=shoes", "shop=jewelry", "shop=gift",
                      "shop=books", "shop=electronics", "shop=mobile_phone",
                      "shop=furniture", "shop=hardware", "shop=florist",
                      "shop=cosmetics", "shop=department_store", "shop=toys",
                      "shop=sports", "shop=bicycle", "shop=pet", "shop=stationery",
                      "shop=art", "shop=music", "shop=variety_store",
                      "shop=second_hand", "shop=optician"],

    "personal_svc":  ["shop=hairdresser", "shop=beauty", "shop=laundry",
                      "shop=dry_cleaning", "shop=massage", "shop=tattoo",
                      "shop=nail"],

    "health":        ["amenity=pharmacy", "healthcare=pharmacy", "shop=chemist",
                      "amenity=doctors", "healthcare=doctor", "amenity=clinic",
                      "healthcare=clinic", "amenity=dentist", "healthcare=dentist",
                      "amenity=hospital", "healthcare=hospital",
                      "healthcare=physiotherapist", "healthcare=psychotherapist",
                      "amenity=veterinary"],

    "education":     ["amenity=school", "amenity=kindergarten", "amenity=childcare",
                      "amenity=college", "amenity=university", "amenity=prep_school",
                      "amenity=music_school", "amenity=language_school"],

    "parks_rec":     ["leisure=park", "leisure=playground", "leisure=pitch",
                      "leisure=nature_reserve", "leisure=dog_park", "leisure=track",
                      "leisure=picnic_table", "leisure=recreation_centre",
                      "leisure=stadium", "tourism=picnic_site"],

    "fitness":       ["leisure=fitness_centre", "leisure=fitness_station",
                      "leisure=sports_centre", "leisure=sports_hall",
                      "amenity=dojo", "leisure=dance"],

    "culture":       ["tourism=museum", "tourism=gallery", "tourism=artwork",
                      "tourism=attraction", "amenity=theatre", "amenity=cinema",
                      "amenity=arts_centre", "amenity=concert_hall",
                      "amenity=music_venue", "amenity=library"],

    "civic_finance": ["amenity=bank", "amenity=atm", "amenity=post_office",
                      "amenity=townhall", "amenity=police", "amenity=fire_station",
                      "amenity=courthouse", "amenity=community_centre",
                      "office=government"],

    "mobility":      ["amenity=bicycle_rental", "amenity=ferry_terminal",
                      "amenity=bus_station", "amenity=taxi"],

    "transit":       ["railway=station", "railway=subway_entrance",
                      "public_transport=station"],

    "disamenity":    ["amenity=fuel", "amenity=car_wash", "shop=car_repair",
                      "shop=tyres", "shop=car_parts", "shop=car",
                      "amenity=waste_disposal", "amenity=waste_dump_site",
                      "amenity=waste_transfer_station", "man_made=wastewater_plant",
                      "man_made=storage_tank", "man_made=works", "man_made=chimney",
                      "amenity=prison", "amenity=crematorium",
                      "amenity=slaughterhouse"],
}

_SESSION = requests.Session()
_SESSION.mount(
    'https://',
    HTTPAdapter(
        pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS,
        max_retries=Retry(total=5, backoff_factor=2, allowed_methods=['POST'],
                          status_forcelist=[429, 500, 502, 503, 504])))


def get_poi_counts(chunk: pd.DataFrame, time: str, poi_filter: str) -> dict:
    """Count POIs matching poi_filter within RADIUS_M of each property,
    as of the OSM snapshot at `time`. Returns {zpid: count}."""
    bcircles = '|'.join(
        f'{row.zpid}:{row.longitude},{row.latitude},{RADIUS_M}'
        for row in chunk.itertuples())

    r = _SESSION.post(
        OHSOME_URL,
        data={'bcircles': bcircles, 'filter': poi_filter, 'time': time},
        timeout=300)
    r.raise_for_status()

    return {int(g['groupByObject']): int(g['result'][0]['value'])
            for g in r.json()['groupByResult']}


if __name__ == '__main__':
    geo = pd.read_csv('data/geoclient.csv')
    geo = geo.dropna(subset=['latitude', 'longitude'])

    sold = pd.read_csv('data/cleaned_sold.csv', usecols=['zpid', 'sold_date'])
    geo = geo.merge(sold.drop_duplicates('zpid'), on='zpid', how='left')
    geo['month'] = pd.to_datetime(geo['sold_date'], format='%m/%d/%Y').dt.to_period('M')

    # ohsome's OSM data lags the present; clamp snapshots to its latest timestamp
    meta = _SESSION.get('https://api.ohsome.org/v1/metadata', timeout=30).json()
    latest = meta['extractRegion']['temporalExtent']['toTimestamp'][:10]

    tasks = []
    for month, group in geo.groupby('month'):
        snapshot = min(f'{month}-01', latest)  # Period('M') formats as YYYY-MM
        for start in range(0, len(group), CHUNK_SIZE):
            chunk = group.iloc[start:start + CHUNK_SIZE]
            for category, tags in POI_CATEGORIES.items():
                tasks.append((chunk, snapshot, category, ' or '.join(tags)))

    def run(task):
        chunk, snapshot, category, poi_filter = task
        return category, get_poi_counts(chunk, snapshot, poi_filter)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(
            executor.map(run, tasks), total=len(tasks), desc='Counting POIs'))

    pois = pd.DataFrame(index=geo['zpid'], columns=list(POI_CATEGORIES), dtype='Int64')
    for category, counts in results:
        for zpid, count in counts.items():
            pois.at[zpid, category] = count

    pois.reset_index().to_csv('data/pois.csv', index=False)
    print(f'Wrote data/pois.csv ({len(pois)} rows)')
