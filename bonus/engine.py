"""Shared evaluation boundary. No planner or judge can access holdout scores."""
import time
import numpy as np
import pandas as pd
from sklearn.base import clone
from hr_pipeline.modeling import features, fit_trial, make_model, score


class TrainingFailure(RuntimeError):
    pass


def input_error(parts):
    required = {'EmployeeID', 'profile_group', 'attrition'}
    for part in parts:
        if not required.issubset(part):
            return 'missing_required_column'
        if part.EmployeeID.isna().any() or not part.EmployeeID.is_unique:
            return 'duplicate_or_null_employee_id'
        if part.attrition.isna().any() or not set(part.attrition).issubset({0, 1}) or part.attrition.nunique() != 2:
            return 'invalid_or_missing_target'
        if part.profile_group.isna().any():
            return 'missing_profile_group'
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        if set(parts[i].profile_group) & set(parts[j].profile_group):
            return 'profile_leakage'
        if set(parts[i].EmployeeID) & set(parts[j].EmployeeID):
            return 'employee_leakage'
    predictors = [set(features(p).columns) for p in parts]
    if not predictors[0] or any(p != predictors[0] for p in predictors[1:]):
        return 'inconsistent_feature_schema'
    return None


def valid_spec(spec):
    try:
        if spec['family'] not in {'logistic', 'forest'}:
            return False
        return (.01 <= float(spec.get('C', 1)) <= 10 and
                3 <= int(spec.get('depth', 5)) <= 10 and
                .2 <= float(spec.get('threshold', .5)) <= .7)
    except (KeyError, ValueError, TypeError, OverflowError):
        return False


class Experiment:
    def __init__(self, parts, fault=None, holdout=False, max_attempts=5):
        self.parts = [p.copy(deep=True) for p in parts]
        self.fault = fault
        self.allow_holdout = holdout
        self.max_attempts = max_attempts
        self.attempts = 0
        self.test_calls = 0
        self.training_failures = 0
        self.trace = []
        self.trials = []
        self.started = time.perf_counter()

    def event(self, node, reason, **details):
        self.trace.append({'sequence': len(self.trace) + 1, 'node': node,
                           'reason': reason, **details})

    def train(self, spec):
        if self.attempts >= self.max_attempts:
            raise TrainingFailure('training_attempt_budget')
        self.attempts += 1
        if self.fault == 'always_fail' or (self.fault == 'transient_failure' and self.attempts == 1):
            self.training_failures += 1
            raise TrainingFailure('simulated_training_failure')
        train, val, _ = self.parts
        actual = dict(spec)
        if self.fault == 'plateau':
            # Deliberately constant predictions, explicitly labeled as injection.
            actual = {'family': 'dummy', 'threshold': .5}
        try:
            model, metrics = fit_trial(train, val, actual)
        except Exception:
            self.training_failures += 1
            raise
        if not np.isfinite(list(metrics.values())).all():
            raise TrainingFailure('nonfinite_metrics')
        row = {'spec': dict(spec), 'metrics': metrics, 'attempt': self.attempts,
               'numeric_income': pd.api.types.is_numeric_dtype(train.MonthlyIncome) if 'MonthlyIncome' in train else False,
               'injected_constant_predictor': self.fault == 'plateau'}
        self.trials.append(row)
        return row

    def finish(self, system, best=None, reason='finished', status='succeeded', **extra):
        test_metrics = None
        if best and self.allow_holdout and status == 'succeeded':
            # Only this boundary sees the test features and labels for scoring.
            train, val, test = self.parts
            develop = pd.concat([train, val], ignore_index=True)
            model = make_model(develop, best['spec'])
            model.fit(features(develop), develop.attrition)
            self.test_calls += 1
            test_metrics = score(model, test, best['spec'].get('threshold', .5))
        return {'system': system, 'status': status, 'stop_reason': reason,
                'best': best, 'trials': self.trials, 'trace': self.trace,
                'training_attempts': self.attempts, 'training_failures': self.training_failures,
                'test_calls': self.test_calls, 'test_metrics': test_metrics,
                'elapsed_seconds': round(time.perf_counter() - self.started, 4),
                'llm_calls': sum(t['node'] == 'llm_planner' and t['reason'] != 'rules' for t in self.trace),
                'mode': 'llm_requested' if any(t['node'] == 'llm_planner' for t in self.trace) else 'rules',
                'fault_injection': self.fault,
                **extra}
