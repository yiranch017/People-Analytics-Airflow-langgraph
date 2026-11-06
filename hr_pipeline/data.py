"""Download, validate, merge and store three independently supplied HR tables."""
import hashlib
import io
import json
import os
from pathlib import Path
import urllib.request
import zipfile

import pandas as pd
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
ARTIFACTS = ROOT / 'artifacts'
FILES = ['general_data.csv', 'employee_survey_data.csv', 'manager_survey_data.csv']
SOURCE = 'https://www.kaggle.com/api/v1/datasets/download/vjchoudhary7/hr-analytics-case-study'


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str))


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    if not all((RAW / f).exists() for f in FILES):
        request = urllib.request.Request(SOURCE, headers={'User-Agent': 'people-analytics-classroom/1.0'})
        with urllib.request.urlopen(request, timeout=90) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        for name in FILES:
            (RAW / name).write_bytes(archive.read(name))
    manifest = {name: {'sha256': hashlib.sha256((RAW / name).read_bytes()).hexdigest(),
                       'bytes': (RAW / name).stat().st_size} for name in FILES}
    save_json(ARTIFACTS / 'source_manifest.json', {'source': SOURCE, 'files': manifest})
    return str(RAW)


def clean_table(name, run_dir):
    df = pd.read_csv(RAW / name)
    if 'EmployeeID' not in df or df.EmployeeID.isna().any() or not df.EmployeeID.is_unique:
        raise ValueError(f'{name}: EmployeeID must be present, non-null and unique')
    for col in df.select_dtypes('object'):
        df[col] = df[col].str.strip()
    if name == 'general_data.csv':
        if not set(df.Attrition).issubset({'Yes', 'No'}):
            raise ValueError('Unexpected attrition label')
        # Missing HR fields vary across otherwise repeated profiles. Exclude these
        # fields from the grouping fingerprint; exclude the outcome as well.
        stable = [c for c in df if c not in {'EmployeeID', 'Attrition'} and not df[c].isna().any()]
        df['profile_group'] = pd.util.hash_pandas_object(df[stable], index=False).astype(str)
        df['attrition'] = df.Attrition.map({'Yes': 1, 'No': 0})
        df = df.drop(columns=['Attrition', 'EmployeeCount', 'Over18', 'StandardHours'])
    else:
        for col in df.columns.difference(['EmployeeID']):
            if not df[col].dropna().between(1, 4).all():
                raise ValueError(f'{name}: {col} outside 1–4 scale')
    # Preserve nulls here; imputation is fitted on training data only.
    destination = Path(run_dir) / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    return str(destination)


def merge_tables(paths, run_dir):
    frames = [pd.read_csv(p, dtype={'profile_group': str}) for p in paths]
    base = frames[0]
    for right in frames[1:]:
        if set(base.EmployeeID) != set(right.EmployeeID):
            raise ValueError('Unmatched employee IDs across HR sources')
        base = base.merge(right, on='EmployeeID', how='left', validate='one_to_one')
    path = Path(run_dir) / 'employees_merged.csv'
    base.to_csv(path, index=False)
    save_json(Path(run_dir) / 'quality.json', {
        'rows': len(base), 'distinct_employee_ids': int(base.EmployeeID.nunique()),
        'distinct_profile_groups': int(base.profile_group.nunique()),
        'missing_values': base.isna().sum().to_dict(),
        'join': 'EmployeeID, one-to-one, identical ID sets required',
    })
    return str(path)


def engine():
    url = os.environ.get('HR_DATABASE_URL')
    if not url:
        raise RuntimeError('HR_DATABASE_URL is required; run inside Docker Compose')
    return create_engine(url)


def load_database(path):
    df = pd.read_csv(path, dtype={'profile_group': str})
    with engine().begin() as conn:
        conn.execute(text('CREATE SCHEMA IF NOT EXISTS hr'))
        df.to_sql('employee_analytics', conn, schema='hr', if_exists='replace', index=False)
        conn.execute(text('ALTER TABLE hr.employee_analytics ADD PRIMARY KEY ("EmployeeID")'))
    return {'rows': len(df), 'table': 'hr.employee_analytics'}


def read_database():
    with engine().connect() as conn:
        return pd.read_sql(text('SELECT * FROM hr.employee_analytics ORDER BY "EmployeeID"'), conn)


def prepare(run_dir):
    download()
    paths = [clean_table(f, run_dir) for f in FILES]
    return load_database(merge_tables(paths, run_dir))
