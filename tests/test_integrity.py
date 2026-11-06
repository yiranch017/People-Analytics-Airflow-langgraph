import pandas as pd
import pytest
from hr_pipeline.data import FILES, clean_table, merge_tables
from hr_pipeline.modeling import split_data, features
from hr_pipeline.agents import validation_judge_agent, route


def merged_data(tmp_path):
    paths = [clean_table(f, tmp_path) for f in FILES]
    return pd.read_csv(merge_tables(paths, tmp_path), dtype={'profile_group': str})


def test_real_source_joins_and_group_isolation(tmp_path):
    df = merged_data(tmp_path)
    assert len(df) == 4410
    assert df.EmployeeID.is_unique
    assert df.profile_group.nunique() == 1470
    train, val, test = split_data(df)
    assert len(train) + len(val) + len(test) == len(df)
    for a, b in [(train, val), (train, test), (val, test)]:
        assert set(a.profile_group).isdisjoint(b.profile_group)
    assert not {'EmployeeID', 'attrition', 'profile_group', 'Gender', 'Age', 'MaritalStatus'} & set(features(train))


def test_duplicate_join_key_is_rejected(tmp_path):
    paths = [clean_table(f, tmp_path) for f in FILES]
    survey = pd.read_csv(paths[1])
    pd.concat([survey, survey.iloc[:1]]).to_csv(paths[1], index=False)
    with pytest.raises(pd.errors.MergeError):
        merge_tables(paths, tmp_path)


def test_trial_budget_and_feedback_routing():
    trial = {'status': 'succeeded', 'metrics': {'average_precision': .25}}
    patch = validation_judge_agent({'current': trial, 'cycle': 1, 'max_cycles': 3})
    assert route(patch) == 'revise'
    patch = validation_judge_agent({'current': trial, 'cycle': 3, 'max_cycles': 3})
    assert route(patch) == 'finalize'
    assert patch['stop_reason'] == 'trial_budget'
