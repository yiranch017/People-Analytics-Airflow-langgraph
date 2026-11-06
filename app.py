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


def graph_dot(graph, active=None):
    raw = graph.get_graph()
    lines = ['digraph G { rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor="#eaf2f8", fontname="Arial"];']
    for name in raw.nodes:
        color = '#99e2b4' if name == active else '#eaf2f8'
        lines.append(f'"{name}" [label="{name.replace("_", " ")}", fillcolor="{color}"];')
    for edge in raw.edges:
        label = str(edge.data or '')
        lines.append(f'"{edge.source}" -> "{edge.target}" [label="{label}"];')
    return '\n'.join(lines) + '\n}'


overview, agents, learning = st.tabs(['People insights', 'Live agent workflow', 'Explain this project'])
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

with learning:
    st.subheader('Airflow coordinates recurring work')
    st.write('A DAG is a task dependency diagram. The scheduler starts runs; the executor runs eligible tasks; the UI shows status and logs. Independent cleaning tasks can overlap. A merge waits for all three. Failed tasks retry twice.')
    st.code('ingest → [clean HR, clean employee survey, clean manager survey]\n       → join → PostgreSQL → read back + baseline → HR agent loop')
    st.write('The schedule is 6:00 a.m. America/Toronto, daily. The source is a static teaching dataset: scheduling demonstrates reliable orchestration, not newly arriving HR records.')
    st.subheader('LangGraph coordinates the experiment feedback')
    st.write('The quality agent validates split isolation; the hypothesis agent proposes a bounded experiment; training computes real metrics; the judge chooses revise or finalize; memory stores the trial; the People Partner report explains limitations. Rules mode is the default, and it is explicitly labeled.')
    st.subheader('Practice explaining these choices')
    st.write('Why is EmployeeID a one-to-one key? Why does accuracy alone hide poor attrition detection? Why must repeated profiles stay together? Why does a satisfaction association not prove a retention intervention will work? What would a prospective employee-month dataset need?')
