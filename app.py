"""Local classroom dashboard with a live view of the actual compiled agent graph."""
import json
import os
from pathlib import Path
import streamlit as st
import pandas as pd
from hr_pipeline.data import ARTIFACTS, read_database
from hr_pipeline.agents import build_graph

st.set_page_config(page_title='People Analytics Lab', page_icon='🌱', layout='wide')
st.title('People Analytics Lab')
st.caption('HR records + employee listening + manager surveys · Week 11 orchestration project')
st.info('Teaching-data case study: retrospective attrition classification. Results are associations, not causes or employment recommendations.')


def graph_dot(graph, active=None, direction='LR'):
    raw = graph.get_graph()
    lines = [f'digraph G {{ rankdir={direction}; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor="#eaf2f8", fontname="Arial"];']
    for name in raw.nodes:
        color = '#99e2b4' if name == active else '#eaf2f8'
        lines.append(f'"{name}" [label="{name.replace("_", " ")}", fillcolor="{color}"];')
    for edge in raw.edges:
        label = str(edge.data or '')
        lines.append(f'"{edge.source}" -> "{edge.target}" [label="{label}"];')
    return '\n'.join(lines) + '\n}'


overview, agents, bonus, learning = st.tabs(['People insights', 'Live agent workflow', 'Bonus: adaptive evaluation', 'Explain this project'])
with overview:
    try:
        df = read_database()
    except Exception:
        st.warning('Run the people_analytics_daily DAG in Airflow first, then refresh this page.')
        df = None
    if df is not None:
        a, b, c = st.columns(3)
        a.metric('Employee records', f'{len(df):,}')
        b.metric('Distinct HR profile groups', f'{df.profile_group.nunique():,}')
        c.metric('Attrition share in source snapshot', f'{df.attrition.mean():.1%}')
        st.subheader('Where are the patterns worth investigating?')
        summary = df.groupby('Department').agg(employees=('attrition', 'size'), attrition_share=('attrition', 'mean'))
        st.bar_chart(summary['attrition_share'])
        st.dataframe(summary.style.format({'attrition_share': '{:.1%}'}), use_container_width=True)
        st.caption('These are snapshot label shares, not annual attrition rates. Repeated profiles mean row counts overstate independent information.')
        st.subheader('Employee listening and attrition')
        survey = df.groupby('JobSatisfaction').agg(employees=('attrition', 'size'), attrition_share=('attrition', 'mean'))
        st.bar_chart(survey['attrition_share'])
        st.caption('Survey timing and nonresponse can affect interpretation. Satisfaction associations do not establish an intervention effect.')
        baseline_path = ARTIFACTS / 'baseline_results.json'
        if baseline_path.exists():
            st.subheader('Part 1: validation benchmarks')
            baseline = json.loads(baseline_path.read_text())
            st.dataframe(pd.DataFrame({k: v for k, v in baseline.items() if k != 'split'}).T, use_container_width=True)

with agents:
    graph = build_graph()
    st.subheader('The graph defined in your code')
    canvas = st.empty()
    canvas.graphviz_chart(graph_dot(graph), use_container_width=True)
    mode = 'Optional LLM planner configured' if os.getenv('OPENROUTER_API_KEY') and os.getenv('OPENROUTER_MODEL') else 'Rules mode · no API key required'
    st.caption(mode + ' · training, scoring and routing run in Python')
    st.write('Selection metric: validation average precision. Stop after reaching 0.40 with at least two trials, or at the three-trial limit.')
    if st.button('Run HR research loop', type='primary', disabled=df is None):
        with st.status('Running experiments on database data…', expanded=True) as status:
            try:
                for update in graph.stream({'max_cycles': 3}, config={'recursion_limit': 40}, stream_mode='updates'):
                    for name, patch in update.items():
                        canvas.graphviz_chart(graph_dot(graph, name), use_container_width=True)
                        st.write('✓ ' + name.replace('_', ' '))
                        if 'current' in patch:
                            st.json(patch['current'])
                status.update(label='Research loop complete', state='complete')
            except Exception as exc:
                status.update(label='Run failed — see error below', state='error')
                st.exception(exc)
    result_path = ARTIFACTS / 'agent_results.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        st.subheader('Validation experiments')
        st.dataframe(pd.DataFrame([{'trial': t['cycle'], 'family': t['spec']['family'],
                                   'planner': t['spec']['planner_mode'], **t.get('metrics', {})}
                                  for t in result['trials']]), use_container_width=True)
        st.subheader('Final reserved test set')
        a, b, c = st.columns(3)
        a.metric('Average precision', f"{result['test_metrics']['average_precision']:.3f}")
        b.metric('Recall', f"{result['test_metrics']['recall']:.3f}")
        c.metric('F1', f"{result['test_metrics']['f1']:.3f}")
        st.json(result['test_metrics'])
        st.caption('The final test set is evaluated once per run and does not feed the experiment loop. Re-running this page is not fresh independent test evidence.')
        for item in result['interpretation']:
            st.write('• ' + item)
        st.download_button('Download evaluation results', result_path.read_text(), 'agent_results.json', 'application/json')

with bonus:
    from bonus.adaptive import build_graph as build_bonus_graph
    from bonus.engine import Experiment
    from bonus.evaluate import CASES, scenario, assess, run_benchmark
    from hr_pipeline.modeling import split_data
    st.subheader('Part 3: adaptive coordination')
    st.write('The supervisor selects data repair, experiment planning, threshold adjustment, configuration repair, retry, or termination from intermediate evidence.')
    st.caption('Default: rules mode. Fault scenarios are controlled simulations on development data, not new observations about employees. This tab never scores the reserved test set.')
    selected_case = st.selectbox('Choose a behavior to demonstrate', [c['id'] for c in CASES])
    rubric = next(c for c in CASES if c['id'] == selected_case)
    st.write('**Input:** ' + rubric['input'])
    st.write('**Expected:** ' + rubric['expected_status'] + ' · ' + rubric['scoring'])
    st.write('**Pass threshold:** ' + rubric['threshold'])
    live_llm = st.checkbox('Use optional LLM experiment planner', value=False,
                disabled=not (os.getenv('OPENROUTER_API_KEY') and os.getenv('OPENROUTER_MODEL')))
    parts = scenario(split_data(df), selected_case, 100) if df is not None else None
    fault = selected_case if selected_case in {'invalid_spec', 'transient_failure', 'always_fail', 'plateau', 'low_recall'} else None
    experiment = Experiment(parts, fault=fault) if parts is not None else None
    graph = build_bonus_graph(experiment, live_llm=live_llm)
    canvas = st.empty()
    canvas.graphviz_chart(graph_dot(graph, direction='TB'), use_container_width=True)
    if st.button('Run adaptive scenario', disabled=parts is None):
        with st.status('Following the adaptive graph…', expanded=True) as status:
            try:
                result = None
                for update in graph.stream({}, {'recursion_limit': 70}, stream_mode='updates'):
                    for node, values in update.items():
                        canvas.graphviz_chart(graph_dot(graph, node, direction='TB'), use_container_width=True)
                        st.write(node + ': ' + str(values.get('reason', values.get('error', 'completed'))))
                        result = values.get('result', result)
                st.session_state['bonus_demo'] = {'case': selected_case, 'result': result, 'contract': assess(rubric, result)}
                status.update(label='Adaptive scenario complete', state='complete')
            except Exception as exc:
                status.update(label='Scenario failed', state='error')
                st.exception(exc)
    if 'bonus_demo' in st.session_state:
        demo = st.session_state['bonus_demo']
        st.subheader('Observed execution trace: ' + demo['case'])
        st.dataframe(pd.DataFrame(demo['result']['trace']), use_container_width=True)
        st.json({'status': demo['result']['status'], 'stop_reason': demo['result']['stop_reason'],
                 'training_attempts': demo['result']['training_attempts'], 'test_calls': demo['result']['test_calls'],
                 'contract': demo['contract']})
    st.subheader('Three-system comparison')
    st.caption('Fixed Airflow policy versus the pinned instructor graph adapted to HR splits and AP selection versus the adaptive supervisor. Baselines use the same safety gate and preprocessing. Fixed policy trains one configuration; agent systems can train up to three.')
    if st.button('Run development-data evaluation', disabled=df is None):
        with st.spinner('Evaluating 12 cases across three systems; no holdout scoring…'):
            run_benchmark(repeats=1, final=False)
        st.success('Evaluation complete')
    suite_path = ARTIFACTS / 'bonus' / 'suite.json'
    if suite_path.exists():
        suite = json.loads(suite_path.read_text())
        st.caption(f"{len(suite['cases'])} cases × {suite['repeats']} repeats × 3 systems · {suite['mode']}")
        st.dataframe(pd.DataFrame(suite['summary']), use_container_width=True)
        st.caption('Task contracts are common to all systems. Specialist-routing contracts apply only to the adaptive design; baseline coordination scores are not assigned.')
        detail = pd.DataFrame([{'case': r['case'], 'repeat': r['repeat'], 'system': r['system'],
                              'status': r['status'], 'task_pass': r['task_pass'],
                              'coordination_pass': r['coordination_pass'], 'attempts': r['training_attempts'],
                              'stop_reason': r['stop_reason']} for r in suite['runs']])
        st.dataframe(detail, use_container_width=True, height=380)
        st.download_button('Download behavioral evaluation', suite_path.read_text(), 'bonus-suite.json', 'application/json')
    comparison_path = ARTIFACTS / 'bonus' / 'comparison.json'
    if comparison_path.exists():
        comparison = json.loads(comparison_path.read_text())
        st.subheader('Frozen clean-data model comparison')
        st.dataframe(pd.DataFrame([{'system': r['system'], 'trials': len(r['trials']),
                                  **(r['test_metrics'] or {})} for r in comparison['results']]), use_container_width=True)
        for limitation in comparison['limitations']:
            st.caption(limitation)

with learning:
    st.subheader('Airflow coordinates recurring work')
    st.write('A DAG is a task dependency diagram. The scheduler starts runs; the executor runs eligible tasks; the UI shows status and logs. Independent cleaning tasks can overlap. A merge waits for all three. Failed tasks retry twice.')
    st.code('ingest → [clean HR, clean employee survey, clean manager survey]\n       → join → PostgreSQL → read back + baseline → HR agent loop')
    st.write('The schedule is 6:00 a.m. America/Toronto, daily. The source is a static teaching dataset: scheduling demonstrates reliable orchestration, not newly arriving HR records.')
    st.subheader('LangGraph coordinates the experiment feedback')
    st.write('The quality agent validates split isolation; the hypothesis agent proposes a bounded experiment; training computes real metrics; the judge chooses revise or finalize; memory stores the trial; the People Partner report explains limitations. Rules mode is the default, and it is explicitly labeled.')
    st.subheader('Practice explaining these choices')
    st.write('Why is EmployeeID a one-to-one key? Why does accuracy alone hide poor attrition detection? Why must repeated profiles stay together? Why does a satisfaction association not prove a retention intervention will work? What would a prospective employee-month dataset need?')
