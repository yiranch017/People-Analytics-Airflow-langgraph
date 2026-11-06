"""Original HR-specific LangGraph: quality gate, feedback, bounded trials, report.

Rules mode is explicit. Optional LLM planning uses only aggregate validation
results, with constrained settings. Training and routing stay ordinary code.
"""
import json
import os
import urllib.request
from typing import TypedDict
from langgraph.graph import END, START, StateGraph
from .data import ARTIFACTS, read_database, save_json
from .modeling import EXCLUDED, features, fit_trial, score, split_data, make_model


class HRState(TypedDict, total=False):
    train: object
    validation: object
    test: object
    quality: dict
    trials: list
    spec: dict
    current: dict
    best: dict
    cycle: int
    max_cycles: int
    verdict: dict
    stop_reason: str
    result: dict


def load_people_data(state):
    df = read_database()
    train, validation, test = split_data(df)
    return {'train': train, 'validation': validation, 'test': test, 'trials': [], 'cycle': 0,
            'max_cycles': int(state.get('max_cycles', 3))}


def hr_quality_agent(state):
    train, val, test = state['train'], state['validation'], state['test']
    if any(not set(a.profile_group).isdisjoint(b.profile_group)
           for a, b in [(train, val), (train, test), (val, test)]):
        raise ValueError('Repeated profiles crossed evaluation splits')
    return {'quality': {'group_isolation': True, 'excluded_predictors': EXCLUDED,
            'split_rows': {'train': len(train), 'validation': len(val), 'test': len(test)},
            'training_attrition_share': float(train.attrition.mean()),
            'framing': 'Retrospective classification on teaching data; no prospective horizon is established.'}}


def optional_llm_plan(history, fallback):
    key, model = os.getenv('OPENROUTER_API_KEY'), os.getenv('OPENROUTER_MODEL')
    if not key or not model:
        return fallback, 'rules', None
    prompt = ('You are an HR analytics experiment planner. Using validation results only, '
              'choose the next experiment. Return JSON with family (logistic or forest), '
              'C (0.01 to 10), depth (3 to 10), threshold (0.2 to 0.7), rationale. '
              'Optimize average precision. Do not claim causality or deployment readiness.')
    payload = {'model': model, 'messages': [{'role': 'system', 'content': prompt},
                {'role': 'user', 'content': json.dumps({'validation_history': history, 'fallback': fallback})}],
               'temperature': .2, 'max_tokens': 350, 'response_format': {'type': 'json_object'}}
    try:
        request = urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',
                    data=json.dumps(payload).encode(), headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=45) as response:
            result = json.load(response)
        proposal = json.loads(result['choices'][0]['message']['content'])
        if proposal['family'] not in ['logistic', 'forest']:
            raise ValueError('Unsupported model family')
        spec = {'family': proposal['family'], 'C': float(proposal.get('C', 1)),
                'depth': int(proposal.get('depth', 5)), 'threshold': float(proposal.get('threshold', .5)),
                'rationale': str(proposal.get('rationale', 'LLM proposal'))[:600]}
        if not (.01 <= spec['C'] <= 10 and 3 <= spec['depth'] <= 10 and .2 <= spec['threshold'] <= .7):
            raise ValueError('Proposal outside allowed bounds')
        return spec, 'llm', None
    except Exception as exc:
        return fallback, 'rules_fallback', type(exc).__name__


def retention_hypothesis_agent(state):
    cycle = state['cycle']
    if cycle == 0:
        spec = {'family': 'logistic', 'C': .1, 'threshold': .5,
                'rationale': 'Start with a regularized, class-balanced linear benchmark.'}
    elif cycle == 1:
        spec = {'family': 'forest', 'depth': 4, 'threshold': .5,
                'rationale': 'Compare nonlinear survey and work-history associations against the linear trial.'}
    else:
        low_recall = state.get('best', {}).get('metrics', {}).get('recall', 0) < .5
        spec = {'family': 'forest', 'depth': 8, 'threshold': .35 if low_recall else .5,
                'rationale': 'Increase tree capacity; lower threshold only if best validation recall is below 0.5.'}
    spec, mode, error = optional_llm_plan(state['trials'], spec)
    spec.update(planner_mode=mode, planner_error=error)
    return {'spec': spec}


def model_training_agent(state):
    try:
        _, metrics = fit_trial(state['train'], state['validation'], state['spec'])
        current = {'cycle': state['cycle'] + 1, 'spec': state['spec'], 'metrics': metrics, 'status': 'succeeded'}
    except Exception as exc:
        current = {'cycle': state['cycle'] + 1, 'spec': state['spec'], 'status': 'failed', 'error': str(exc)}
    return {'current': current, 'cycle': state['cycle'] + 1}


def validation_judge_agent(state):
    current, best = state['current'], state.get('best')
    if current['status'] == 'succeeded' and (not best or current['metrics']['average_precision'] > best['metrics']['average_precision']):
        best = current
    # Require at least two trials to demonstrate feedback; never exceed budget.
    good_enough = bool(best and best['metrics']['average_precision'] >= .4 and state['cycle'] >= 2)
    stop = good_enough or state['cycle'] >= state['max_cycles']
    reason = 'validation_goal' if good_enough else ('trial_budget' if stop else 'revise')
    return {'best': best, 'stop_reason': reason,
            'verdict': {'action': 'finalize' if stop else 'revise', 'reason': reason,
                        'metric': 'validation average precision', 'target': .4}}


def experiment_memory_agent(state):
    return {'trials': state['trials'] + [{**state['current'], 'verdict': state['verdict']}]}


def route(state):
    return state['verdict']['action']


def sealed_test_agent(state):
    import pandas as pd
    best = state.get('best')
    if not best:
        raise RuntimeError('All experiments failed; test evaluation was not attempted')
    # Hyperparameters and threshold are now frozen. Refit on development data.
    development = pd.concat([state['train'], state['validation']], ignore_index=True)
    model = make_model(development, best['spec'])
    model.fit(features(development), development.attrition)
    metrics = score(model, state['test'], best['spec'].get('threshold', .5))
    return {'result': {'quality': state['quality'], 'trials': state['trials'], 'best_validation_trial': best,
                       'test_metrics': metrics, 'stop_reason': state['stop_reason'],
                       'test_policy': 'One final evaluation per run; never supplied to planner/judge.'}}


def people_partner_report_agent(state):
    result = state['result']
    result['interpretation'] = [
        'Patterns describe this teaching dataset; they do not establish causes of resignation.',
        'The source has no reliable prediction snapshot or future outcome window; this is retrospective classification.',
        'Repeated HR profiles stay in the same split. Small, repeated teaching data limits generalization.',
        'Age, gender and marital status are audit-only; excluding them does not establish fairness.',
        'Use aggregate findings to form follow-up questions, not individual employment decisions.',
        'Attrition share in this snapshot is not an annual attrition rate; no currency or observation dates are assumed.',
    ]
    save_json(ARTIFACTS / 'agent_results.json', result)
    return {'result': result}


def build_graph():
    g = StateGraph(HRState)
    nodes = {'load_people_data': load_people_data, 'hr_quality_agent': hr_quality_agent,
             'retention_hypothesis_agent': retention_hypothesis_agent, 'model_training_agent': model_training_agent,
             'validation_judge_agent': validation_judge_agent, 'experiment_memory_agent': experiment_memory_agent,
             'sealed_test_agent': sealed_test_agent, 'people_partner_report_agent': people_partner_report_agent}
    for name, function in nodes.items():
        g.add_node(name, function)
    chain = [START, 'load_people_data', 'hr_quality_agent', 'retention_hypothesis_agent',
             'model_training_agent', 'validation_judge_agent', 'experiment_memory_agent']
    for source, target in zip(chain, chain[1:]):
        g.add_edge(source, target)
    g.add_conditional_edges('experiment_memory_agent', route,
           {'revise': 'retention_hypothesis_agent', 'finalize': 'sealed_test_agent'})
    g.add_edge('sealed_test_agent', 'people_partner_report_agent')
    g.add_edge('people_partner_report_agent', END)
    return g.compile()


def run():
    graph = build_graph()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / 'agent_graph.mmd').write_text(graph.get_graph().draw_mermaid())
    return graph.invoke({'max_cycles': 3}, config={'recursion_limit': 40})['result']


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
