# Release validation

| Check | Result |
|---|---|
| Automated tests | 71 passed |
| CLI and web predictions | 14 candidates across seven designs; matching predictions |
| Five-split replay | 35 models and 105 tau-b values reproduced |
| Five-split retraining | Model selection and all predictions reproduced |
| Installation package | Wheel build, separate installation and CLI prediction passed |
| CPU placement build | DREAMPlace compiled from the bundled source |

The new-design calibration orchestrator is tested for training/prediction isolation with mocked EDA and an actual XGBoost fit. Machine-independent records are in [validation/summary.json](validation/summary.json).
