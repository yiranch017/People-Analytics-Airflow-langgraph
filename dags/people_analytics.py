"""Scheduled, parallel HR ETL → PostgreSQL → analysis → agent research."""
from datetime import timedelta
import pendulum
from airflow.decorators import dag, task


@dag(dag_id='people_analytics_daily', schedule='0 6 * * *',
     start_date=pendulum.datetime(2026, 11, 1, tz='America/Toronto'),
     catchup=False, max_active_runs=1,
     default_args={'retries': 2, 'retry_delay': timedelta(seconds=20)},
     tags=['week11', 'people-analytics'],
     description='Three HR sources → clean in parallel → join → PostgreSQL → baseline → research loop')
def people_analytics_daily():
    @task
    def ingest_sources():
        from hr_pipeline.data import download
        return download()

    @task
    def clean_source(name, source_dir, **context):
        from hr_pipeline.data import ARTIFACTS, clean_table
        # Each run owns its staging directory. XCom carries paths, not employee rows.
        return clean_table(name, ARTIFACTS / 'runs' / context['ts_nodash'])

    @task
    def join_employee_tables(paths, **context):
        from hr_pipeline.data import ARTIFACTS, merge_tables
        return merge_tables(paths, ARTIFACTS / 'runs' / context['ts_nodash'])

    @task
    def load_postgres(path):
        from hr_pipeline.data import load_database
        return load_database(path)

    @task
    def analyze_from_database(loaded):
        from hr_pipeline.data import ARTIFACTS, read_database, save_json
        from hr_pipeline.modeling import baseline
        df = read_database()
        if len(df) != loaded['rows']:
            raise ValueError('Database row count does not reconcile')
        segments = df.groupby('Department').agg(employees=('attrition', 'size'), attrition_share=('attrition', 'mean')).reset_index()
        save_json(ARTIFACTS / 'department_summary.json', segments.to_dict(orient='records'))
        return baseline(df, ARTIFACTS / 'baseline_results.json')

    @task
    def run_hr_agent_loop(baseline_result):
        from hr_pipeline.agents import run
        result = run()
        return {'test_metrics': result['test_metrics'], 'trials': len(result['trials'])}

    source = ingest_sources()
    general = clean_source.override(task_id='clean_hr_records')('general_data.csv', source)
    employee = clean_source.override(task_id='clean_employee_survey')('employee_survey_data.csv', source)
    manager = clean_source.override(task_id='clean_manager_survey')('manager_survey_data.csv', source)
    merged = join_employee_tables([general, employee, manager])
    loaded = load_postgres(merged)
    analyzed = analyze_from_database(loaded)
    run_hr_agent_loop(analyzed)


people_analytics_daily()
