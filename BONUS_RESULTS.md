# Part 3 — Measured results

## Execution evidence

The manual `people_analytics_bonus` Airflow DAG completed successfully. Its three independent policy tasks feed the behavioral evaluation task. The full project test suite passed **19 tests**, including execution of the pinned instructor graph. No DAG import errors or broken Python requirements were found.

## Behavioral comparison

**12 cases × 3 repetitions × 3 systems = 108 executions.** Every scenario uses development data only. No behavioral run called the held-out evaluator.

| System | Passed common contracts | Recoverable-case completion | Adaptive routing contracts | Mean fit attempts | Mean elapsed seconds |
|---|---:|---:|---:|---:|---:|
| fixed_airflow | 83.3% | 85.7% | N/A | 1.08 | 0.281 |
| classroom_simple | 83.3% | 100.0% | N/A | 2.33 | 1.191 |
| adaptive | 100.0% | 100.0% | 100% | 1.75 | 0.602 |

Both baselines pass 30/36 common contracts; the adaptive system passes 36/36 common contracts and 36/36 of its own coordination contracts. The additional adaptive checks are architecture-specific and are not used to assign zero scores to baselines.

### What failed, and why?

- **Fixed workflow:** comma-formatted income was treated as categorical instead of restoring the numeric schema; retries could not correct an invalid C parameter.
- **Classroom graph:** the same numeric-schema issue remained; persistent training failure consumed more than the scenario's three-attempt termination threshold, despite eventually stopping.
- **Adaptive graph:** satisfied every declared case, including configuration repair, local recovery, threshold dispatch and plateau termination. This only establishes coverage of these specified scenarios.

The classroom graph already recovers from the temporary training error and invalid parameter through its native repair logic. These capabilities are not claimed as unique to the adaptive extension. All systems share the basic safety gate and block missing labels, duplicate employee IDs and profile leakage before training.

Timings include Python work and reference-module initialization, but exclude first-time network download and Airflow scheduling overhead. The fixed workflow is cheapest on straightforward inputs. These synthetic, repeated cases do not establish population-level statistical significance.

## Frozen clean-data model comparison

All three frozen policies use the same 2,646 training rows, 882 validation rows and 882 held-out rows, with profile groups isolated across partitions. Each system was scored once on the holdout for this comparison.

| System | Successful validation trials | Test AP | ROC AUC | Precision | Recall | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed_airflow | 1 | 0.307 | 0.691 | 0.303 | 0.701 | 0.423 | 0.681 |
| classroom_simple | 3 | 0.307 | 0.690 | 0.299 | 0.680 | 0.416 | 0.681 |
| adaptive | 2 | 0.284 | 0.627 | 0.268 | 0.306 | 0.286 | 0.745 |

**The adaptive system did not improve clean-data predictive performance.** The fixed and classroom systems achieved higher held-out AP, recall and F1. The adaptive forest met the development stopping rule but generalized less well. We retain this result rather than retune against the test set.

The supported conclusion is that explicit specialist routing improved satisfaction of these behavioral contracts, not that more agents produce a better attrition model. The fixed baseline gets one configuration and agent workflows can try up to three, so this is not an equal-search-budget causal experiment about routing. The holdout had already been examined in the original assignment; these are exploratory results.

## Reproducibility and evidence

- [Declared test inputs and thresholds](bonus/cases.json)
- [Full behavioral results and traces](evidence/bonus-suite.json)
- [Frozen model comparison](evidence/bonus-comparison.json)
- [Successful bonus Airflow graph](evidence/bonus-airflow-graph.png)
- [Adaptive retry trace](evidence/bonus-adaptive-retry.png)
- [Complete adaptive graph](evidence/bonus-adaptive-graph.png)
- [Three-system dashboard](evidence/bonus-comparison-dashboard.png)
- [Held-out model comparison](evidence/bonus-model-comparison.png)
- [Design, commands and baseline adaptations](BONUS.md)

Rules mode was evaluated; no live LLM calls were made. The instructor source snapshot, source URLs and observed SHA-256 hashes are included in the machine-readable behavioral results. Simulated training faults and constant-prediction plateau inputs are labeled explicitly.
