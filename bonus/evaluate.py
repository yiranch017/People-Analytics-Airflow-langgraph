"""Run reproducible behavioral cases and a separate frozen clean-data comparison."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from hr_pipeline.data import ARTIFACTS, ROOT, read_database, save_json
from hr_pipeline.modeling import split_data
from .adaptive import run as adaptive
from .baselines import classroom, fixed, download_reference, REFERENCE

CASES = json.loads(Path(__file__).with_name('cases.json').read_text())
SYSTEMS = {'fixed_airflow': fixed, 'classroom_simple': classroom, 'adaptive': adaptive}


def scenario(parts, name, seed):
    parts = [p.copy(deep=True) for p in parts]
    if name == 'survey_missing':
        rng = np.random.default_rng(seed)
        train = parts[0]
        for c in ['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']:
            train.loc[rng.random(len(train)) < .2, c] = np.nan
    elif name == 'numeric_strings':
        for part in parts:
            part['MonthlyIncome'] = part.MonthlyIncome.map(lambda x: f'{x:,.0f}')
    elif name == 'missing_target':
        parts = [p.drop(columns='attrition') for p in parts]
    elif name == 'missing_labels':
        parts[0].loc[parts[0].index[:3], 'attrition'] = np.nan
    elif name == 'duplicate_id':
        parts[0] = pd.concat([parts[0], parts[0].iloc[:1]], ignore_index=True)
    elif name == 'profile_leakage':
        parts[1].loc[parts[1].index[0], 'profile_group'] = parts[0].profile_group.iloc[0]
    return parts


def assess(case, result):
    system = result['system']
    checks = {'expected_status': result['status'] == case['expected_status'],
              'attempt_budget': result['training_attempts'] <= (3 if case['id'] == 'always_fail' else 5),
              'trial_budget': len(result['trials']) <= 3,
              'no_test_calls': result['test_calls'] == 0,
              'finite_metrics': all(np.isfinite(list(t['metrics'].values())).all() for t in result['trials'])}
    if case['id'] in {'missing_target', 'missing_labels', 'duplicate_id', 'profile_leakage'}:
        checks['blocked_before_training'] = result['training_attempts'] == 0
    if case['id'] == 'numeric_strings':
        checks['correct_numeric_schema'] = bool(result['trials']) and all(t['numeric_income'] for t in result['trials'])
    # The common task contract applies to every system. The separate coordination
    # contract asks whether our adaptive trace used the specified specialist.
    coordination = {}
    if system == 'adaptive':
        visited = [e['node'] for e in result['trace']]
        for node in case.get('required_adaptive_nodes', []):
            coordination[f'visited_{node}'] = node in visited
        for node in case.get('forbidden_adaptive_nodes', []):
            coordination[f'skipped_{node}'] = node not in visited
        coordination['repair_budget'] = result.get('repairs', 0) <= 2
        coordination['retry_budget'] = result.get('retries', 0) <= 2
        if case['id'] == 'plateau':
            coordination['plateau_stop'] = result['stop_reason'] == 'validation_plateau'
        if case['id'] == 'transient_failure':
            coordination['no_quality_restart'] = visited.count('quality') == 1
    return {'task_checks': checks, 'task_pass': all(checks.values()),
            'coordination_checks': coordination, 'coordination_pass': all(coordination.values()) if coordination else None}


def suite(parts, repeats=3):
    if not 1 <= repeats <= 5:
        raise ValueError('repeats must be between 1 and 5')
    reference_manifest = download_reference()  # Exclude network setup from timed runs.
    runs = []
    for repeat in range(repeats):
        # Rotate ordering to reduce systematic first-run/cache timing effects.
        systems = list(SYSTEMS.items())
        systems = systems[repeat % 3:] + systems[:repeat % 3]
        for case in CASES:
            changed = scenario(parts, case['id'], 100 + repeat)
            fault = case['id'] if case['id'] in {'invalid_spec', 'transient_failure', 'always_fail', 'plateau', 'low_recall'} else None
            for name, fn in systems:
                result = fn(changed, fault=fault, holdout=False)
                runs.append({'case': case['id'], 'repeat': repeat, **result, **assess(case, result)})
    summary = []
    for name in SYSTEMS:
        subset = [r for r in runs if r['system'] == name]
        positive = [r for r in subset if next(c for c in CASES if c['id'] == r['case'])['expected_status'] == 'succeeded']
        summary.append({'system': name, 'cases_run': len(subset),
                        'task_contract_pass_rate': sum(r['task_pass'] for r in subset) / len(subset),
                        'completion_rate_on_recoverable_cases': sum(r['status'] == 'succeeded' for r in positive) / len(positive),
                        'coordination_pass_rate': sum(r['coordination_pass'] for r in subset) / len(subset) if name == 'adaptive' else None,
                        'mean_training_attempts': sum(r['training_attempts'] for r in subset) / len(subset),
                        'mean_elapsed_seconds': sum(r['elapsed_seconds'] for r in subset) / len(subset)})
    return {'created_utc': datetime.now(timezone.utc).isoformat(), 'mode': 'rules; no live LLM calls',
            'reference': REFERENCE, 'reference_manifest': reference_manifest,
            'repeats': repeats, 'cases': CASES, 'summary': summary, 'runs': runs,
            'interpretation': 'Fault cases evaluate behavior on development data only. Repeats are robustness checks, not independent samples of employees.'}


def final_comparison(parts):
    # All configurations and policies must be frozen before this command.
    # Each system gets exactly one final evaluation on the SAME holdout.
    results = [fn(parts, holdout=True) for fn in SYSTEMS.values()]
    return {'created_utc': datetime.now(timezone.utc).isoformat(), 'results': results,
            'policy': 'One clean held-out evaluation per system. Shared partitions and preprocessing. No test scores go back to agents.',
            'limitations': ['Fixed baseline has one successful trial; agent systems have up to three. Performance differences combine search budget and routing effects.',
                'AP selection replaces classroom F1 selection for a matched metric; original graph and fallback roles remain.',
                'Timed comparisons include Python orchestration and reference-module initialization, not Airflow scheduling/queue latency.',
                'The holdout was already used in the original assignment. This extension is exploratory, not a fresh independent confirmation.',
                'No live-LLM effectiveness or cost claim is established by rules-mode results.']}


def run_benchmark(repeats=3, final=False):
    df = read_database()
    parts = split_data(df)
    result = suite(parts, repeats)
    result['split_fingerprints'] = {name: hashlib.sha256('\n'.join(sorted(p.profile_group.astype(str))).encode()).hexdigest()
                                  for name, p in zip(['train', 'validation', 'test'], parts)}
    dest = ARTIFACTS / 'bonus'
    save_json(dest / 'suite.json', result)
    save_json(ROOT / 'evidence' / 'bonus-suite.json', result)
    if final:
        comparison = final_comparison(parts)
        save_json(dest / 'comparison.json', comparison)
        save_json(ROOT / 'evidence' / 'bonus-comparison.json', comparison)
    print(json.dumps(result['summary'], indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--final', action='store_true', help='Evaluate each frozen policy once on the previously used holdout')
    args = parser.parse_args()
    run_benchmark(args.repeats, args.final)
