"""Fixed policy and an adapter around the actual pinned classroom graph.

The reference is downloaded, not redistributed. The adapter changes the data,
training/scoring boundary, selection metric and common safety gate; graph edges,
role implementations and heuristic experiment proposals come from the reference.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import urllib.request
from unittest.mock import patch
from hr_pipeline.data import ARTIFACTS, save_json
from .engine import Experiment, input_error, TrainingFailure

REFERENCE = json.loads(Path(__file__).with_name('reference.json').read_text())
REFERENCE_DIR = ARTIFACTS / 'classroom_reference'
REFERENCE_LOCK = threading.Lock()


def download_reference():
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, source in REFERENCE['files'].items():
        url = f"https://raw.githubusercontent.com/Kedar-V/automl-research-lab/{REFERENCE['commit']}/{source}"
        path = REFERENCE_DIR / name
        # Cache pin is recorded separately; never mix source snapshots.
        marker = REFERENCE_DIR / 'commit.txt'
        if not path.exists() or not marker.exists() or marker.read_text().strip() != REFERENCE['commit']:
            with urllib.request.urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
        manifest[name] = {'source': url, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (REFERENCE_DIR / 'commit.txt').write_text(REFERENCE['commit'])
    save_json(REFERENCE_DIR / 'manifest.json', manifest)
    return manifest


def fixed(parts, fault=None, holdout=False):
    e = Experiment(parts, fault=fault, holdout=holdout)
    e.event('quality', 'shared_safety_gate')
    error = input_error(e.parts)
    if error:
        return e.finish('fixed_airflow', reason=error, status='blocked')
    spec = {'family': 'logistic', 'C': .25, 'threshold': .5}
    if fault == 'invalid_spec':
        spec['C'] = -1
    try:
        # Fixed Airflow task policy: retries repeat the same configuration.
        for attempt in range(3):
            e.event('fixed_training_task', 'predefined_configuration', retry=attempt)
            try:
                trial = e.train(spec)
                break
            except (TrainingFailure, ValueError):
                if attempt == 2:
                    raise
        return e.finish('fixed_airflow', trial, reason='fixed_steps_complete')
    except Exception as exc:
        return e.finish('fixed_airflow', reason=str(exc), status='blocked')


def _load_reference():
    download_reference()
    # graph.py imports loop_runtime by its classroom absolute name.
    # Isolate both modules to avoid changing the original Part 2 implementation.
    runtime_spec = importlib.util.spec_from_file_location('loop_runtime', REFERENCE_DIR / 'loop_runtime.py')
    runtime = importlib.util.module_from_spec(runtime_spec)
    runtime_spec.loader.exec_module(runtime)
    spec = importlib.util.spec_from_file_location('bonus_classroom_reference', REFERENCE_DIR / 'graph.py')
    module = importlib.util.module_from_spec(spec)
    # LangGraph resolves the reference TypedDict's postponed annotations using
    # its registered module namespace when compiling the graph.
    sys.modules[spec.name] = module
    with patch.dict(sys.modules, {'loop_runtime': runtime}):
        spec.loader.exec_module(module)
    return module


def classroom(parts, fault=None, holdout=False):
    e = Experiment(parts, fault=fault, holdout=holdout)
    e.event('quality', 'shared_safety_gate')
    error = input_error(e.parts)
    if error:
        return e.finish('classroom_simple', reason=error, status='blocked')
    best = None
    # Loading source and accessing the global import map are serialized.
    with REFERENCE_LOCK:
        try:
            ref = _load_reference()
            original_engineer = ref.node_ml_engineer

            def fit_boundary(df, target, spec, seed, holdout_frac, *, on_holdout):
                nonlocal best
                if on_holdout:
                    # Actual holdout scoring occurs once in Experiment.finish.
                    return {'evaluation': 'deferred_to_shared_holdout_boundary'}
                hp = spec.get('hyperparameters', {})
                mapped = {'family': {'logistic_regression': 'logistic', 'random_forest': 'forest'}[spec['model_family']],
                          'C': hp.get('C', .25), 'depth': hp.get('max_depth', 5), 'threshold': .5}
                if fault == 'invalid_spec' and e.attempts == 0:
                    mapped['C'] = -1
                row = e.train(mapped)
                if best is None or row['metrics']['average_precision'] > best['metrics']['average_precision']:
                    best = row
                return {'status': 'succeeded', 'validation_metrics': row['metrics'],
                        'experiment_id': spec.get('experiment_id'), 'duration_seconds': 0}

            def engineer(s):
                # Preserve the instructor role and repair code. Standardize its
                # best-spec selection to AP; historical best_val_f1 key now holds AP.
                result = original_engineer(s)
                if best:
                    result['best_val_f1'] = best['metrics']['average_precision']
                    result['best_spec'] = s.get('best_spec') or result.get('experiment_spec')
                return result

            def judge(result, target_score):
                metrics = result.get('validation_metrics', {})
                ap = metrics.get('average_precision', 0.)
                success = result.get('status') == 'succeeded'
                return {'action': 'accept' if success and ap >= target_score else 'revise',
                        'observation': f'validation AP {ap:.3f} ' + ('meets' if ap >= target_score else 'below') + ' target',
                        'next_action': 'finalize' if ap >= target_score else 'propose next experiment'}

            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'training_profile.csv'
                # The profiling role sees training only; fit_boundary supplies
                # explicit shared splits rather than the reference row split.
                e.parts[0].to_csv(path, index=False)
                ref._fit_score = fit_boundary
                ref.node_ml_engineer = engineer
                ref._heuristic_verdict = judge
                ref.llm_configured = lambda *args, **kwargs: False
                graph = ref.build_graph()
                for update in graph.stream({'dataset_path': str(path), 'target_column': 'attrition',
                         'max_cycles': 3, 'target_score': .4, 'cycle': 0},
                         {'recursion_limit': 40}, stream_mode='updates'):
                    for node, values in update.items():
                        e.event(node, 'instructor_graph_node',
                                action=(values.get('verdict') or {}).get('action'))
            if not best:
                return e.finish('classroom_simple', status='blocked', reason='no_successful_trial')
            return e.finish('classroom_simple', best, reason='instructor_graph_finalized',
                            reference_commit=REFERENCE['commit'])
        except Exception as exc:
            return e.finish('classroom_simple', best, status='blocked', reason=f'{type(exc).__name__}: {exc}')
