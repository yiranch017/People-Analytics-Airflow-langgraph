"""Shared, reproducible group-isolated model evaluation."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Treat these as audit-only attributes, and never include IDs or group labels.
EXCLUDED = ['EmployeeID', 'profile_group', 'attrition', 'Age', 'Gender', 'MaritalStatus']


def split_data(df):
    splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
    develop, test = next(splitter.split(df, groups=df.profile_group))
    train_pos, val_pos = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=43)
                              .split(df.iloc[develop], groups=df.iloc[develop].profile_group))
    parts = [df.iloc[develop[train_pos]].copy(), df.iloc[develop[val_pos]].copy(), df.iloc[test].copy()]
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        assert set(parts[a].profile_group).isdisjoint(parts[b].profile_group)
    if any(p.attrition.nunique() != 2 for p in parts):
        raise ValueError('Each split must contain both classes')
    return parts


def make_model(train, spec):
    x = train.drop(columns=EXCLUDED, errors='ignore')
    categorical = list(x.select_dtypes('object').columns)
    numeric = list(x.columns.difference(categorical))
    transform = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)),
                              ('scale', StandardScaler())]), numeric),
        ('category', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                               ('encode', OneHotEncoder(handle_unknown='ignore'))]), categorical),
    ])
    family = spec['family']
    if family == 'dummy':
        estimator = DummyClassifier(strategy='prior')
    elif family == 'logistic':
        estimator = LogisticRegression(C=spec.get('C', 1.), class_weight='balanced', max_iter=1500, random_state=42)
    elif family == 'forest':
        estimator = RandomForestClassifier(n_estimators=150, max_depth=spec.get('depth', 5),
                   min_samples_leaf=10, class_weight='balanced', random_state=42, n_jobs=1)
    else:
        raise ValueError(f'Unsupported model: {family}')
    return Pipeline([('features', transform), ('model', estimator)])


def features(frame):
    return frame.drop(columns=EXCLUDED, errors='ignore')


def score(model, frame, threshold=.5):
    y = frame.attrition
    prob = model.predict_proba(features(frame))[:, 1]
    pred = (prob >= threshold).astype(int)
    return { 'average_precision': float(average_precision_score(y, prob)),
             'roc_auc': float(roc_auc_score(y, prob)),
             'f1': float(f1_score(y, pred, zero_division=0)),
             'precision': float(precision_score(y, pred, zero_division=0)),
             'recall': float(recall_score(y, pred, zero_division=0)),
             'accuracy': float(accuracy_score(y, pred)),
             'prevalence': float(y.mean()), 'rows': len(frame),
             'threshold': threshold }


def fit_trial(train, val, spec):
    model = make_model(train, spec)
    model.fit(features(train), train.attrition)
    return model, score(model, val, spec.get('threshold', .5))


def baseline(df, output):
    from .data import save_json
    train, val, test = split_data(df)
    results = {}
    for family in ['dummy', 'logistic']:
        _, results[family] = fit_trial(train, val, {'family': family})
    results['split'] = {name: {'rows': len(part), 'profile_groups': int(part.profile_group.nunique())}
                        for name, part in zip(['train', 'validation', 'sealed_test'], [train, val, test])}
    save_json(output, results)
    return results
