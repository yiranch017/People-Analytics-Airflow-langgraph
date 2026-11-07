"""Behavioral integration tests; these exercise real training and the graph."""
import numpy as np
import pandas as pd
import pytest
from hr_pipeline.data import FILES, clean_table, merge_tables
from hr_pipeline.modeling import split_data
from bonus.adaptive import run
from bonus.baselines import fixed, classroom
from bonus.engine import valid_spec
from bonus.evaluate import scenario, assess, CASES


@pytest.fixture(scope='module')
def parts(tmp_path_factory):
    folder = tmp_path_factory.mktemp('bonus')
    path = merge_tables([clean_table(f, folder) for f in FILES], folder)
    return split_data(pd.read_csv(path, dtype={'profile_group': str}))


@pytest.mark.parametrize('case', CASES, ids=lambda case: case['id'])
def test_adaptive_case_contracts(parts, case):
    fault = case['id'] if case['id'] in {'invalid_spec', 'transient_failure', 'always_fail', 'plateau', 'low_recall'} else None
    result = run(scenario(parts, case['id'], 100), fault=fault)
    contract = assess(case, result)
    assert contract['task_pass'], (result['stop_reason'], contract['task_checks'])
    assert contract['coordination_pass'], contract['coordination_checks']
    assert result['test_calls'] == 0
    for trial in result['trials']:
        assert valid_spec(trial['spec'])


def test_fixed_policy_cannot_repair_invalid_configuration(parts):
    result = fixed(parts, fault='invalid_spec')
    assert result['status'] == 'blocked'
    assert result['training_attempts'] == 3
    assert result['test_calls'] == 0


def test_no_holdout_scores_reach_routing_trace(parts):
    result = run(parts)
    assert all('test_metrics' not in event for event in result['trace'])
    assert result['test_metrics'] is None


def test_reject_nonfinite_and_out_of_bounds_configuration():
    for spec in [{'family': 'unknown'}, {'family': 'logistic', 'C': float('nan')},
                 {'family': 'forest', 'depth': 100}, {'family': 'logistic', 'threshold': .99}]:
        assert not valid_spec(spec)


def test_pinned_classroom_reference_executes_real_graph(parts):
    result = classroom(parts)
    assert result['status'] == 'succeeded', result['stop_reason']
    nodes = [t['node'] for t in result['trace']]
    assert {'data_scientist', 'hypothesis_generator', 'ml_engineer', 'judge', 'memory_writer',
            'context_builder', 'holdout_eval'} <= set(nodes)
    assert result['training_attempts'] == 3
    assert result['test_calls'] == 0
    assert result['reference_commit']
