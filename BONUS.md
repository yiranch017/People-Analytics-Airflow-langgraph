# Part 3 — Adaptive multi-agent coordination and evaluation

The bonus extension adds evidence-based specialist selection, targeted recovery and a reproducible three-system comparison to the original HR assignment. All original Part 1 and Part 2 workflows remain available.

## Run and inspect

With the original Docker environment running and the HR database populated:

```bash
# First run downloads two source files from the pinned instructor repository.
docker compose exec scheduler python -m bonus.evaluate --repeats 3

# Run all integrity and behavioral tests.
docker compose exec scheduler python -m pytest -q
```

Open http://localhost:8502 → **Bonus: adaptive evaluation**. Select a scenario, read its expected behavior, then click **Run adaptive scenario**. The graph highlights execution and shows routing reasons, training attempts and pass/fail checks. A blocked result is correct for unsafe inputs such as missing labels or overlapping profile groups.

For Airflow evidence, unpause and trigger the manual **people_analytics_bonus** DAG at http://localhost:8090. Three independent tasks run the fixed, classroom and adaptive policies on the same clean partitions. A downstream task evaluates all 12 behavioral cases with three repetitions. This DAG never evaluates the test set. Its tasks do not retry the entire benchmark, so a broken evaluator is visible rather than hidden by reruns.

The optional `--final` flag adds one clean held-out model evaluation per system after the policies have been frozen:

```bash
docker compose exec scheduler python -m bonus.evaluate --repeats 3 --final
```

Do not use repeated held-out evaluations to tune policies. The final comparison is stored separately from behavioral testing. The holdout was already examined in the original assignment, so this extension is exploratory rather than a fresh independent confirmation.

## What adapts?

The graph topology is defined in code; the executed path changes according to intermediate results. The supervisor chooses among data repair, planning, threshold adjustment, configuration repair, local retry, finalization and safe termination.

| Evidence | Selected action | Boundary |
|---|---|---|
| Survey values missing | Acknowledge approved imputation through data specialist | Imputation statistics are fitted only on training data |
| Known numeric column arrives as comma-formatted strings | Normalize numeric schema | Invalid non-numeric tokens fail; outcomes are never invented |
| Labels missing, duplicate IDs, overlapping groups | Stop with explicit reason | No training or test evaluation |
| Invalid model settings | Configuration specialist selects safe settings | At most two repairs |
| Simulated temporary training failure | Retry training only | At most two retries; never rerun ingestion |
| Validation recall below 0.5 | Threshold specialist proposes 0.35 | Validation only; AP ranking need not improve |
| AP goal reached after at least two trials | Finalize | Goal 0.40 is an exercise rule, not production acceptance |
| Two trials without meaningful ranking improvement | Stop for plateau | AP improvement must exceed 0.001 |
| Budget exhausted | Finalize best valid trial or block | At most three successful trials / five fit attempts |

Role outputs and routing reasons are recorded in a chronological execution trace. Trial history is the shared memory. Threshold-level F1 can break an AP tie without claiming better ranking. This distinction matters: changing a decision threshold does not improve average precision.

Default execution is **rules-based multi-agent coordination**. Optional LLM planning uses the existing bounded OpenRouter adapter, sends only aggregate development history, and is explicitly labeled. It requires your key and model in `.env`; live LLM results are not established by the deterministic benchmark. The supervisor and graph construction remain code-defined. No claim is made that an LLM dynamically generates the graph or arbitrary code.

## The three comparison systems

1. **Fixed Airflow policy:** one predefined class-balanced logistic configuration (`C=0.25`), with up to two retries of that same task. It cannot repair an invalid configuration or reinterpret a changed input schema. The same function runs as a real task in the bonus DAG and in the evaluation harness. Harness timing excludes Airflow queue/scheduling latency.
2. **Classroom simple graph:** executes the instructor's actual `solution/langgraph/graph.py`, downloaded with `loop_runtime.py` at the commit in [bonus/reference.json](bonus/reference.json). This preserves the instructor's role functions, memory/context loop, conditional edges and fallback experiment proposals. It is not the project's custom Part 2 graph relabeled as the class baseline.
3. **Adaptive supervisor:** original graph in [bonus/adaptive.py](bonus/adaptive.py), with specialized recovery paths, explicit failure taxonomy, plateau detection and a shared attempt budget.

### Documented classroom adapters

- Profile training data only; read the already prepared HR data rather than the demo fraud/Titanic CSV.
- Use the same group-disjoint partitions, excluded fields and training-fitted preprocessing as the other systems.
- Replace the training/scoring boundary, preserving actual sklearn training. Keep instructor experiment settings and native repair proposals.
- Standardize selection and the stopping target to AP rather than classroom F1. The historical `best_val_f1` state key consequently stores AP within this adapter; genuine F1 is still reported separately.
- Defer the instructor holdout node's scoring to the shared final evaluation boundary, so scenarios do not inspect test scores.
- Disable all LLM calls for reproducible rules-mode comparison.
- Apply the same prerequisite data safety gate to all three systems.

The downloaded reference stays under ignored `artifacts/classroom_reference/`; it is not redistributed. Its exact pinned URLs and observed SHA-256 hashes are recorded with evaluation results. Reference setup/network failure is an evaluator failure, not evidence that the classroom method is ineffective. A test verifies that the actual instructor graph executes successfully.

## Evaluation design

[bonus/cases.json](bonus/cases.json) specifies the input, expected outcome, scoring criteria and pass/fail threshold for every case. These are controlled scenarios on copies of the partitions; the PostgreSQL source is never corrupted. Survey masks use recorded seeds. Fault injection is explicitly recorded rather than presented as a naturally occurring failure.

The 12 cases are: clean inputs, additional survey missingness, numeric strings, missing target column, missing labels, duplicate IDs, profile leakage, invalid configuration, temporary training failure, persistent failure, constant-prediction plateau, and low validation recall.

There are two distinct contracts:

- **Common task contract:** correct successful/blocked outcome, finite metrics, correct numeric schema when applicable, bounded training, and zero test calls. Missing labels, duplicate IDs and leakage must block before fitting. A model that silently treats income strings as categories may complete, but fails the schema requirement.
- **Adaptive coordination contract:** visits the required specialist, skips unnecessary repair on clean inputs, keeps repairs/retries bounded, avoids restarting quality checks during a training retry, and records the appropriate plateau termination. These architecture-specific assertions apply only to the adaptive graph; baseline coordination scores are not assigned artificial zeros.

Intermediate quality includes schema handling, configuration validity and route correctness. Final quality includes common behavioral outcomes plus a separately frozen clean-data model evaluation (AP, ROC AUC, precision, recall, F1 and accuracy). Efficiency reports fit attempts and elapsed wall time. Rules-mode calls/cost are zero; no live LLM cost or effectiveness claim is made.

Three repetitions rotate system ordering and vary the synthetic missingness mask. These are robustness checks, not independent employee samples or a basis for statistical significance. Every system receives the same changed input within a case. Expected outcomes are written before the final benchmark; failed baseline contracts remain in the results.

## Interpretation limits

These tests were designed around this project's supported recovery paths. A 100% pass rate means the stated contracts were met; it does not imply general autonomous competence or coverage of all HR data problems. Some safeguards are shared, so safe stopping on missing labels is not credited exclusively to the supervisor.

The fixed baseline gets one successful configuration, while agent systems may try three. Model performance differences therefore combine search budget, proposals and coordination; they cannot be attributed to routing alone. Timing includes Python work and instructor-module initialization but excludes initial reference downloads and Airflow scheduling overhead. The original dataset has repeated profiles and no prospective prediction window; the original HR interpretation limitations still apply.

See [BONUS_RESULTS.md](BONUS_RESULTS.md) for measured results and real execution screenshots. No improvement claim should be inferred merely from adding agents.
