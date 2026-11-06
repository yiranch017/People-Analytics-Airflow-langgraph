FROM apache/airflow:2.10.4-python3.11
COPY requirements.txt /tmp/hr-requirements.txt
RUN pip install --no-cache-dir "apache-airflow==2.10.4" -r /tmp/hr-requirements.txt
# The upstream classroom image includes Snowpark, whose old cloudpickle cap
# conflicts with LangGraph. This project uses PostgreSQL, not Snowflake.
RUN pip uninstall -y apache-airflow-providers-snowflake snowflake-snowpark-python && pip check
ENV PYTHONPATH=/project MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /project
