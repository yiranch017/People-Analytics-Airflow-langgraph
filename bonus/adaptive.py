"""Evidence-based dispatch to specialist agents, with bounded repair/retry."""
from typing import TypedDict
from langgraph.graph import START, END, StateGraph
from hr_pipeline.agents import optional_llm_plan
from .engine import Experiment, input_error, valid_spec, TrainingFailure


class State(TypedDict, total=False):
    ready: bool
    fatal: str
    issue: str
    spec: dict
    error: str
    best: dict
    trial_count: int
    stale: int
    repairs: int
    retries: int
    threshold_done: bool
    next: str
    reason: str
    result: dict


def build_graph(experiment, live_llm=False):
    e = experiment

    def quality(s):
        error = input_error(e.parts)
        if error:
            e.event('quality', error)
            return {'fatal': error}
        # A schema correction is safe only for numeric fields in this dataset.
        surveys = ['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']
        numeric_fields = surveys + ['MonthlyIncome', 'TotalWorkingYears']
        bad_types = [c for c in numeric_fields if c in e.parts[0] and
                     any(p[c].dtype == object for p in e.parts)]
        issue = 'numeric_strings' if bad_types else ('survey_missingness' if
                any(p[[c for c in surveys if c in p]].isna().any().any() for p in e.parts) else '')
        e.event('quality', issue or 'valid_inputs', missingness_handled=not bool(issue))
        return {'issue': issue, 'ready': not bool(issue)}

    def supervisor(s):
        if s.get('fatal'):
            dest, reason = 'stop', s['fatal']
        elif s.get('issue') and not s.get('ready'):
            dest, reason = 'data_repair', s['issue']
        elif s.get('error'):
            if s['error'] == 'invalid_spec' and s.get('repairs', 0) < 2:
                dest, reason = 'config_repair', 'invalid_configuration'
            elif s['error'] != 'invalid_spec' and s.get('retries', 0) < 2 and e.attempts < e.max_attempts:
                dest, reason = 'retry', 'targeted_training_retry'
            else:
                dest, reason = 'stop', 'repair_or_retry_budget'
        elif s.get('stale', 0) >= 2:
            dest, reason = 'finalize', 'validation_plateau'
        elif s.get('trial_count', 0) >= 3:
            dest, reason = 'finalize', 'trial_budget'
        elif s.get('best') and s.get('trial_count', 0) >= 2 and s['best']['metrics']['average_precision'] >= .4:
            dest, reason = 'finalize', 'validation_goal'
        elif s.get('best') and s['best']['metrics']['recall'] < .5 and not s.get('threshold_done'):
            dest, reason = 'threshold', 'low_validation_recall'
        else:
            dest, reason = 'planner', 'next_bounded_experiment'
        e.event('supervisor', reason, destination=dest)
        return {'next': dest, 'reason': reason}

    def data_repair(s):
        if s['issue'] == 'numeric_strings':
            fields = ['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance', 'MonthlyIncome', 'TotalWorkingYears']
            for p in e.parts:
                for c in fields:
                    if c in p:
                        # Non-numeric tokens are not silently replaced with guesses.
                        import pandas as pd
                        values = p[c].str.replace(',', '', regex=False) if p[c].dtype == object else p[c]
                        p[c] = pd.to_numeric(values, errors='raise')
        # Nulls stay null here. The approved sklearn pipeline learns imputation
        # values on training data, never on validation or holdout data.
        e.event('data_repair', s['issue'], strategy='numeric_schema_or_training_only_imputation')
        return {'ready': True, 'issue': ''}

    def planner(s):
        if not s.get('trial_count'):
            spec = {'family': 'logistic', 'C': .25, 'threshold': .7 if e.fault == 'low_recall' else .5}
        elif s['best']['spec']['family'] == 'logistic':
            spec = {'family': 'forest', 'depth': 5, 'threshold': .5}
        else:
            spec = {'family': 'forest', 'depth': 8, 'threshold': .5}
        if live_llm:
            spec, mode, error = optional_llm_plan(e.trials, spec)
            e.event('llm_planner', mode, fallback_error=error)
        if e.fault == 'invalid_spec' and not s.get('trial_count') and not s.get('repairs'):
            spec = {'family': 'logistic', 'C': -1, 'threshold': .5}
        e.event('planner', 'validation_history_only', spec=spec)
        return {'spec': spec, 'error': ''}

    def threshold(s):
        spec = {**s['best']['spec'], 'threshold': .35}
        e.event('threshold', 'recall_below_0.5', spec=spec)
        return {'spec': spec, 'threshold_done': True, 'error': ''}

    def config_repair(s):
        spec = {'family': 'logistic', 'C': 1., 'threshold': .5}
        e.event('config_repair', 'replace_invalid_settings_with_safe_baseline', spec=spec)
        return {'spec': spec, 'error': '', 'repairs': s.get('repairs', 0) + 1}

    def retry(s):
        e.event('retry', 'retry_training_only', retry=s.get('retries', 0) + 1)
        return {'error': '', 'retries': s.get('retries', 0) + 1}

    def training(s):
        if not valid_spec(s['spec']):
            e.event('training', 'configuration_rejected_before_fit')
            return {'error': 'invalid_spec'}
        try:
            current = e.train(s['spec'])
            e.event('training', 'succeeded', attempt=e.attempts)
            return {'current': current, 'error': ''}
        except (TrainingFailure, ValueError) as exc:
            e.event('training', 'failed', error=str(exc), attempt=e.attempts)
            return {'error': str(exc)}

    def evaluate(s):
        current = e.trials[-1]
        prior = s.get('best')
        improved = not prior or current['metrics']['average_precision'] > prior['metrics']['average_precision'] + .001
        e.event('evaluate', 'improved' if improved else 'no_ranking_improvement', metrics=current['metrics'])
        # Equal ranking scores can improve threshold-level recall/F1. Retain
        # the better F1 on ties without calling it a ranking improvement.
        tie_better = prior and abs(current['metrics']['average_precision'] - prior['metrics']['average_precision']) <= .001 and current['metrics']['f1'] > prior['metrics']['f1']
        return {'best': current if improved or tie_better else prior,
                'stale': 0 if improved else s.get('stale', 0) + 1,
                'trial_count': s.get('trial_count', 0) + 1}

    def finalize(s):
        e.event('finalize', s['reason'])
        result = e.finish('adaptive', s.get('best'), reason=s['reason'],
                          repairs=s.get('repairs', 0), retries=s.get('retries', 0))
        return {'result': result}

    def stop(s):
        e.event('stop', s['reason'])
        return {'result': e.finish('adaptive', status='blocked', reason=s['reason'],
                                  repairs=s.get('repairs', 0), retries=s.get('retries', 0))}

    g = StateGraph(State)
    for name, fn in [('quality', quality), ('supervisor', supervisor), ('data_repair', data_repair),
                     ('planner', planner), ('threshold', threshold), ('config_repair', config_repair),
                     ('retry', retry), ('training', training), ('evaluate', evaluate), ('finalize', finalize), ('stop', stop)]:
        g.add_node(name, fn)
    g.add_edge(START, 'quality')
    g.add_edge('quality', 'supervisor')
    g.add_conditional_edges('supervisor', lambda s: s['next'],
        {n: n for n in ['data_repair', 'planner', 'threshold', 'config_repair', 'retry', 'finalize', 'stop']})
    g.add_edge('data_repair', 'supervisor')
    for n in ['planner', 'threshold', 'config_repair', 'retry']:
        g.add_edge(n, 'training')
    g.add_conditional_edges('training', lambda s: 'supervisor' if s.get('error') else 'evaluate',
                            {'supervisor': 'supervisor', 'evaluate': 'evaluate'})
    g.add_edge('evaluate', 'supervisor')
    g.add_edge('finalize', END)
    g.add_edge('stop', END)
    return g.compile()


def run(parts, fault=None, holdout=False, live_llm=False):
    e = Experiment(parts, fault=fault, holdout=holdout)
    try:
        result = build_graph(e, live_llm).invoke({}, {'recursion_limit': 70})['result']
    except Exception as exc:
        e.event('stop', 'unhandled_failure', error=f'{type(exc).__name__}: {exc}')
        result = e.finish('adaptive', status='blocked', reason='unhandled_failure')
    return result
