"""Manual Part 3 verification: three actual Airflow tasks, then robustness suite.

No held-out scores are computed by this DAG. The explicit --final CLI command
produces the separately documented, frozen clean-data model comparison.
"""
import pendulum
from airflow.decorators import dag, task


@dag(dag_id='people_analytics_bonus', schedule=None, catchup=False,
     start_date=pendulum.datetime(2026, 11, 1, tz='America/Toronto'),
     max_active_runs=1, tags=['week11', 'bonus', 'evaluation'],
     description='Run fixed, classroom and adaptive policies, then evaluate coordination')
def people_analytics_bonus():
    @task
    def run_policy(system, **context):
        from hr_pipeline.data import ARTIFACTS, read_database, save_json
        from hr_pipeline.modeling import split_data
        from bonus.evaluate import SYSTEMS
        result = SYSTEMS[system](split_data(read_database()), holdout=False)
        save_json(ARTIFACTS / 'bonus' / 'dag' / context['ts_nodash'] / f'{system}.json', result)
        if result['status'] != 'succeeded':
            raise RuntimeError(f"{system}: {result['stop_reason']}")
        return {'system': system, 'status': result['status'],
                'validation_ap': result['best']['metrics']['average_precision'],
                'test_calls': result['test_calls']}

    @task
    def evaluate_coordination(fixed, classroom, adaptive):
        from bonus.evaluate import run_benchmark
        result = run_benchmark(repeats=3, final=False)
        failures = [r for r in result['runs'] if r['system'] == 'adaptive' and
                    (not r['task_pass'] or not r['coordination_pass'])]
        if failures:
            raise RuntimeError(f'{len(failures)} adaptive contract failures; see artifacts/bonus/suite.json')
        return result['summary']

    fixed = run_policy.override(task_id='fixed_airflow_baseline')('fixed_airflow')
    classroom = run_policy.override(task_id='classroom_simple_baseline')('classroom_simple')
    adaptive = run_policy.override(task_id='adaptive_agent_system')('adaptive')
    evaluate_coordination(fixed, classroom, adaptive)


people_analytics_bonus()
