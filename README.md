# People Analytics Lab — Week 11

**Question:** How do employee survey responses and work-history characteristics relate to observed attrition, and can a model distinguish attrition labels better than a naive benchmark?

This project implements an original scheduled Airflow pipeline and HR-specific LangGraph loop. It practices skills relevant to people analytics: joining HR systems, employee listening, data quality, model evaluation and careful interpretation. It is a teaching case, not an analysis of any actual employer or a validated prospective attrition model.

## Run the project

Start Docker Desktop, then run from this directory:

```bash
docker compose up --build -d
```

1. Open **Airflow:** http://localhost:8090. Login: `student` / `classroom`.
2. Unpause `people_analytics_daily`, trigger a run, and watch its Graph view.
3. Once successful, open **People Analytics Lab:** http://localhost:8502.
4. View HR charts and model results. In **Live agent workflow**, use **Run HR research loop** to watch the graph execute.

```bash
docker compose exec scheduler python -m pytest -q
docker compose exec scheduler python -m hr_pipeline.agents
docker compose down
```

The second command runs Part 2 independently after Part 1 populates the database. Stopping retains the database and logs. Ports differ from the classroom demo. Airflow 2.10.4 and Python 3.11 match that demo.

## Part 1: data pipeline

Source: [Vijay Choudhary's HR Analytics Case Study](https://www.kaggle.com/datasets/vjchoudhary7/hr-analytics-case-study). Three separate source files each contain 4,410 rows:

- `general_data.csv`: HR records and attrition labels.
- `employee_survey_data.csv`: satisfaction and work-life balance.
- `manager_survey_data.csv`: involvement and performance ratings.

The pipeline downloads the files, cleans them in parallel, validates unique/non-null IDs and survey ranges, trims strings, recodes the label, removes constant fields, and merges on `EmployeeID` with one-to-one validation. It loads `hr.employee_analytics` in PostgreSQL. Analysis explicitly reads that database table for department summaries and a logistic-regression baseline.

```text
                    ┌→ Clean HR records ────────┐
Download sources ───┼→ Clean employee survey ──┼→ Join → PostgreSQL
                    └→ Clean manager survey ───┘             │
                                      Database analysis ←───┘
                                             ↓
                                       HR agent loop
```

Schedule: **06:00 America/Toronto daily**, cron `0 6 * * *`. Catchup is disabled; failed tasks retry twice. LocalExecutor permits the three independent cleaning tasks to overlap; the successful run confirms that they did. This is a static source, so scheduling demonstrates orchestration rather than newly arriving daily HR data. Snapshot loads replace the table transactionally to avoid duplicate appends.

Raw files are downloaded from the publisher and excluded from Git. The publisher API reports the license as **Unknown**; do not redistribute raw data as though a reuse license were established.

## Part 2: agent modifications

```text
Database → HR quality agent → Retention hypothesis → Training → Validation judge
                                    ↑                              ↓
                                    └──── revise ← Experiment memory
                                                         │ finalize
                                                         ↓
                                            Reserved test → People Partner report
```

This original graph adds an HR-specific quality gate and People Partner report, uses validation **average precision** for model selection, preserves experiment history, and enforces a three-trial limit. It finalizes after at least two trials when validation average precision reaches 0.40, or when the budget expires. The UI renders the actual compiled graph and highlights execution.

Default **rules mode** runs without an API key and is labeled explicitly. Optional LLM experiment planning is implemented: copy `.env.example` to `.env`, set `OPENROUTER_API_KEY` and `OPENROUTER_MODEL`, then recreate services. Only aggregate validation history is sent; proposed settings are bounded and validated. Live LLM mode requires your key and has not been tested by the rules-mode verification.

## Findings and validation

The 4,410 records contain only **1,470 distinct profiles across non-missing general HR predictors**. Matching profiles are grouped before the train/validation/test split to avoid inflated scores from repeated records appearing on both sides. Missing-value imputation is fitted inside each training pipeline. IDs, outcome, profile fingerprint, age, gender and marital status are excluded as model inputs; this alone does not establish fairness.

All eight Airflow tasks succeeded, all three integrity tests passed, and the live dashboard run completed. The selected forest achieved **test average precision 0.271, recall 0.327 and F1 0.297**. Results are modest. The naive validation baseline achieved about 84% accuracy with zero recall, demonstrating why accuracy alone is insufficient.

See [verified results and screenshots](RESULTS.md) and the [detailed walkthrough](WALKTHROUGH.md). Findings describe associations in a teaching snapshot. They do not establish causal retention drivers, an annual attrition rate, a future prediction window, or readiness for individual employment decisions. Optional Part 3 bonus work is not claimed.

Review the code and results before submission, and follow your course's AI-use attribution requirements.
