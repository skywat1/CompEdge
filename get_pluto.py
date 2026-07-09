"""Attach PLUTO fields to sales, bucketed to the correct historical release.

Each sale is matched to the PLUTO version whose SOURCE DATA is entirely on or
before the sale date -- never a snapshot containing information from after the
sale. The boundary is a version's latest "date of data" (from page 2 of each
release's README), NOT its publication month: what can leak into a model is the
vintage of the values, and DCP publishes a version a few weeks after its data is
cut. See PLUTO_VERSIONS below.

Data flow:
    cleaned_sold.csv (zpid, sold_date) --join on zpid--> geoclient.csv (zpid, bbl)
    then each row's sold_date picks a PLUTO version, and we join on bbl to that
    version's archive CSV.

Run: `python get_pluto.py`. If an archive CSV is missing it prints which versions
the data needs and exits without writing.
"""
import bisect
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# --- config -----------------------------------------------------------------

SALES_CSV = Path('data/cleaned_sold.csv')     # has zpid, sold_date
GEO_CSV = Path('data/geoclient.csv')          # has zpid, bbl
OUT_CSV = Path('data/pluto.csv')

# PLUTO columns to pull (lowercase, exactly as in the archive CSV header). Output
# columns keep the same lowercase names. Look up meanings in pluto_datadictionary.pdf.
PLUTO_FIELDS = [
    # structural / size (numeric)
    'lotarea', 'bldgarea', 'resarea', 'officearea', 'retailarea', 'garagearea',
    'strgearea', 'factryarea', 'otherarea', 'lotfront', 'lotdepth', 'bldgfront',
    'bldgdepth', 'yearbuilt', 'yearalter1', 'yearalter2', 'builtfar', 'residfar',
    'commfar',
    # assessed value (numeric) -- DOF CAMA, leak-safe under the date bucketing
    'assesstot', 'assessland',
    # designation -> binarized to 1/0/NaN in _clean
    'histdist', 'landmark',
    # categorical codes / labels
    'schooldist', 'policeprct', 'sanitboro', 'sanitsub', 'sanitdistrict', 'landuse',
    'tract2010', 'areasource', 'bldgclass', 'overlay1', 'overlay2', 'spdist1',
    'spdist2', 'spdist3', 'ltdheight', 'ext',
]
FIELD_MAP = {c: c for c in PLUTO_FIELDS}

# Categorical fields -> written as clean strings so they round-trip from CSV as
# categories, not floats (numeric codes like 8.0 -> '8'; already-text ones pass through).
CATEGORICAL_FIELDS = [
    'schooldist', 'policeprct', 'sanitboro', 'sanitsub', 'sanitdistrict', 'landuse',
    'tract2010', 'areasource', 'bldgclass', 'overlay1', 'overlay2', 'spdist1',
    'spdist2', 'spdist3', 'ltdheight', 'ext',
]
# Designation fields collapsed to a present/absent flag (1/0/NaN).
BINARY_FIELDS = ['histdist', 'landmark']

# Every downloaded PLUTO version, as (boundary_date, version, csv_path).
# boundary_date = the version's LATEST source-data date, taken from the
# "DATES OF DATA" table on page 2 of that version's pluto_readme.pdf. A sale
# snaps to the newest version whose boundary_date <= sale date.
PLUTO_VERSIONS = [
    (date(2025, 6, 23), '25v2', 'data/pluto_archive/nyc_pluto_25v2_arc_csv/pluto_25v2.csv'),
    (date(2025, 9, 25), '25v3', 'data/pluto_archive/nyc_pluto_25v3_arc_csv/pluto_25v3.csv'),
    (date(2026, 1, 26), '25v4', 'data/pluto_archive/nyc_pluto_25v4_arc_csv/pluto_25v4.csv'),
    (date(2026, 4, 14), '26v1', 'data/pluto_archive/nyc_pluto_26v1_csv/pluto_26v1.csv'),
]

# ---------------------------------------------------------------------------


def _versions_sorted():
    return sorted(PLUTO_VERSIONS, key=lambda v: v[0])


def assign_version(sold_date, versions=None):
    """Version whose boundary date is the latest <= sold_date.

    Sales before the earliest boundary fall back to the earliest version
    (nothing older is available). None for NaT/missing dates.
    """
    if pd.isna(sold_date):
        return None
    versions = versions or _versions_sorted()
    rel = [v[0] for v in versions]
    d = sold_date.date() if hasattr(sold_date, 'date') else sold_date
    i = bisect.bisect_right(rel, d) - 1
    return versions[max(i, 0)][1]


def _bbl_key(series):
    """Normalize a bbl column to a nullable-int key for joining."""
    return pd.to_numeric(series, errors='coerce').astype('Int64')


def _clean(df):
    """Fix known PLUTO data issues in place so the output is model-ready.

    A lot is considered matched iff 'bldgclass' is populated (every real lot has
    one); unmatched lots keep NaN everywhere rather than being imputed.
    """
    matched = df['bldgclass'].notna() if 'bldgclass' in df else pd.Series(False, index=df.index)

    # HistDist bug: 25v2 mislabeled ~1.2k lots as "Individual Landmark" (fixed in
    # 25v3). That value belongs in the landmark field, not the historic-district
    # field, so blank it out before histdist is binarized below.
    if 'histdist' in df:
        bug = df['histdist'].astype('string').str.strip().str.lower() == 'individual landmark'
        df.loc[bug, 'histdist'] = pd.NA

    # Designation fields -> binary flags. A matched lot with a blank value is a
    # real 0 (known: not in a historic district / not landmarked); only unmatched
    # lots stay NaN (unknown). Binary keeps the in/out signal without re-encoding
    # neighborhood via the sparse district/landmark names.
    for col in BINARY_FIELDS:
        if col in df:
            present = df[col].notna().astype(float)
            df[col] = np.where(matched, present, np.nan)

    # Categorical codes are categories, not magnitudes (schooldist 8 isn't "more"
    # than 7). Write them as clean strings ('8', not 8.0) so they round-trip from
    # CSV as categoricals rather than floats; text codes (bldgclass 'A5') pass through.
    for col in CATEGORICAL_FIELDS:
        if col in df:
            df[col] = df[col].astype('string').str.replace(r'\.0$', '', regex=True)


def build():
    versions = _versions_sorted()
    path_for = {v[1]: Path(v[2]) for v in versions}

    # sales + bbl, joined on zpid
    sales = pd.read_csv(SALES_CSV, usecols=['zpid', 'sold_date'])
    geo = pd.read_csv(GEO_CSV, usecols=['zpid', 'bbl']).drop_duplicates('zpid')
    df = sales.merge(geo, on='zpid', how='left')
    df['sold_date'] = pd.to_datetime(df['sold_date'], errors='coerce')
    df['bbl'] = _bbl_key(df['bbl'])

    # bucket each sale to a version
    df['pluto_version'] = df['sold_date'].apply(lambda d: assign_version(d, versions))

    print('Sales per PLUTO version:')
    for ver, n in df['pluto_version'].value_counts(dropna=False).items():
        print(f'  {ver}: {n}')
    n_before = (df['sold_date'] < pd.Timestamp(versions[0][0])).sum()
    if n_before:
        print(f'  ({n_before} sales predate {versions[0][1]} -> bucketed to it)')

    # every needed version must have its archive CSV on disk
    needed = [v for v in df['pluto_version'].dropna().unique()]
    missing = [v for v in needed if not path_for[v].exists()]
    if missing:
        print('\nMissing archive CSVs for versions:', ', '.join(sorted(missing)))
        print('Fix the paths in PLUTO_VERSIONS. Nothing written.')
        return

    # join fields, loading each version CSV exactly once
    src_cols = list(FIELD_MAP)
    for out_col in FIELD_MAP.values():
        # object dtype holds both the string fields (landmark, histdist) and the
        # numeric ones without lossy float/str coercion on assignment.
        df[out_col] = pd.Series(pd.NA, index=df.index, dtype='object')

    for ver in needed:
        pv = pd.read_csv(path_for[ver], usecols=['bbl'] + src_cols, low_memory=False)
        pv['bbl'] = _bbl_key(pv['bbl'])
        pv = pv.dropna(subset=['bbl']).drop_duplicates('bbl').set_index('bbl')

        rows = df['pluto_version'] == ver
        for src, out_col in FIELD_MAP.items():
            df.loc[rows, out_col] = df.loc[rows, 'bbl'].map(pv[src]).values

    _clean(df)
    df.to_csv(OUT_CSV, index=False)
    # a lot "matched" if its bbl was found in the version file; bldgclass is
    # populated for every real lot, so use it as the presence probe.
    probe = 'bldgclass' if 'bldgclass' in df else FIELD_MAP[PLUTO_FIELDS[0]]
    matched = (df['bbl'].notna() & df[probe].notna()).sum()
    print(f'\nWrote {OUT_CSV} ({len(df)} rows, {matched} matched to a PLUTO lot).')


if __name__ == '__main__':
    build()
