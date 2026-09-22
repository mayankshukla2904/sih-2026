Note: The model size for pruned + fined tuned model here is not actual size as the pruned weights are masked, not actually removed, for quick experimental purposes.

The fine tuning happened for 6 epochs only as Early stopping was triggered.


### FP32 — Pruned vs Baseline

| Metric                     | Pruned + Fine-tuned |     Baseline |
| -------------------------- | ------------------: | -----------: |
| Accuracy (%)               |             99.0684 |      98.7370 |
| TPR (%)                    |               75.38 |        84.10 |
| FAR (%)                    |                0.20 |         0.63 |
| False accepts              |                  22 |           69 |
| Threshold                  |                0.52 |         0.52 |
| Model size (KB)            |              <95.91 |        95.91 |
| Parameters                 |              23,747 |       23,747 |
| Host tensor buffers (KiB)  |            1,715.37 |       902.62 |
| ESP32 arena used (KiB)     |                   — |            — |
| ESP32 arena budget (KiB)   |                   — |         65.0 |
| ESP32 arena headroom (KiB) |                   — |            — |
| ESP32 arena status         |                   — | Not measured |
| Mean latency (ms)          |              0.1844 |       0.1681 |
| Median latency (ms)        |              0.1708 |       0.1435 |
| P95 latency (ms)           |              0.2395 |       0.2627 |
| P99 latency (ms)           |              0.2890 |       0.4105 |
| Min latency (ms)           |              0.1679 |       0.1415 |
| Max latency (ms)           |              0.3041 |       0.6506 |

### FP16 — Pruned vs Baseline

| Metric                     | Pruned + Fine-tuned |     Baseline |
| -------------------------- | ------------------: | -----------: |
| Accuracy (%)               |             99.0774 |      98.7370 |
| TPR (%)                    |               75.38 |        84.10 |
| FAR (%)                    |                0.20 |         0.63 |
| False accepts              |                  22 |           69 |
| Threshold                  |                0.52 |         0.52 |
| Model size (KB)            |              <57.69 |        57.69 |
| Parameters                 |              23,747 |       23,747 |
| Host tensor buffers (KiB)  |            1,758.50 |       945.63 |
| ESP32 arena used (KiB)     |                   — |            — |
| ESP32 arena budget (KiB)   |                   — |         65.0 |
| ESP32 arena headroom (KiB) |                   — |            — |
| ESP32 arena status         |                   — | Not measured |
| Mean latency (ms)          |              0.1881 |       0.1593 |
| Median latency (ms)        |              0.1677 |       0.1456 |
| P95 latency (ms)           |              0.2537 |       0.2113 |
| P99 latency (ms)           |              0.5258 |       0.2299 |
| Min latency (ms)           |              0.1664 |       0.1421 |
| Max latency (ms)           |              0.5834 |       0.2461 |

### INT8 — Pruned vs Baseline

| Metric                     | Pruned + Fine-tuned |      Baseline |
| -------------------------- | ------------------: | ------------: |
| Accuracy (%)               |             99.0684 |       98.7549 |
| TPR (%)                    |               74.87 |         84.10 |
| FAR (%)                    |                0.21 |          0.66 |
| False accepts              |                  23 |            72 |
| Threshold                  |                0.52 |          0.52 |
| Model size (KB)            |              <47.74 |         47.74 |
| Parameters                 |              23,747 |        23,747 |
| Host tensor buffers (KiB)  |              430.54 |        227.36 |
| ESP32 arena used (KiB)     |               55.81 |         55.81 |
| ESP32 arena budget (KiB)   |                64.0 |          64.0 |
| ESP32 arena headroom (KiB) |                8.19 |          8.19 |
| ESP32 arena status         |                   — | Within budget |
| Mean latency (ms)          |              0.4668 |        0.4078 |
| Median latency (ms)        |              0.4232 |        0.3790 |
| P95 latency (ms)           |              0.6906 |        0.5337 |
| P99 latency (ms)           |              1.0254 |        0.7098 |
| Min latency (ms)           |              0.4030 |        0.3335 |
| Max latency (ms)           |              1.0361 |        0.8023 |

 
