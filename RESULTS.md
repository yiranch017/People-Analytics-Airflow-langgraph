# Verified results

The scheduled Airflow run completed all eight tasks successfully. The three cleaning tasks overlapped in execution, as intended. The dashboard also successfully ran the live agent graph with no UI exceptions.

## Data and split checks

- Three independently supplied tables, 4,410 rows each, merged one-to-one on EmployeeID.
- 1,470 repeated-profile groups; group-disjoint split: 2,646 training rows, 882 validation rows, 882 reserved test rows.
- 3 integrity tests passed; no DAG import errors; no broken Python requirements.

## Validation experiments

| Model | Average precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|
| Dummy baseline | 0.160 | 0.000 | 0.000 | 0.840 |
| Part 1 logistic baseline | 0.368 | 0.645 | 0.394 | 0.683 |
| Agent trial 1: logistic | 0.370 | 0.645 | 0.386 | 0.671 |
| Agent trial 2: forest | 0.404 | 0.426 | 0.360 | 0.759 |

The loop stopped after two trials because the forest reached the exercise validation average-precision goal of 0.40. This is a stopping rule, not evidence of production quality.

## Final reserved test set

| Average precision | ROC AUC | Precision | Recall | F1 | Accuracy |
|---:|---:|---:|---:|---:|---:|
| 0.271 | 0.613 | 0.273 | 0.327 | 0.297 | 0.743 |

Test attrition prevalence is 0.167. The selected model ranks cases above the constant-score baseline, but performance is modest and validation-to-test degradation is substantial. The dummy model illustrates the accuracy trap: approximately 84% validation accuracy with zero recall.

The evidence supports successful orchestration and a reproducible modeling exercise. It does not support deployment, individual risk decisions, causal explanations, or claims that a retention intervention will work. Better temporally defined data and out-of-time validation would be needed for a prospective use case.

Rules mode was tested. Optional LLM planning was not tested because no user API key was provided. Repeated UI executions reproduce the same held-out test evaluation; they are not independent tests.

## Evidence

- [Airflow successful graph](evidence/airflow-dag-graph.png)
- [Agent graph and model results](evidence/agent-graph-and-results.png)
- [Live dashboard execution](evidence/live-agent-execution.png)
- [People insights](evidence/people-insights.png)
- [Machine-readable evaluation](evidence/evaluation-results.json)
