# Runtime measurements

The model evaluation page includes a compact per-design timing table and a CSV export. Values are bundled in `src/paris_tool/resources/benchmark.json` under `runtime` and do not depend on the selected evaluation split.

| Column | Measurement |
| --- | --- |
| CTS | Historical wall time per candidate from `4_1_cts.log`; excludes placement and filler insertion. |
| GR | Historical wall time per candidate from `5_1_grt.log`; excludes separate feature exports and detailed routing. |
| GR feature export | Sum of recorded GR design-information and timing exports. This covers part of feature preparation; a complete checkpoint-to-1,002-column extraction timer was not recorded. |
| Feature aggregation | Reading, validating and aggregating the existing three GR feature tables, once per design on the fixed 200-candidate evaluation pool. Reported per batch. |
| Model inference | Loaded model and in-memory feature matrix, including the GR anchor, inverse TNS transform and ranking. Five warmups followed by 100 measurements, reported as mean milliseconds per batch. Excludes model loading and feature processing. |

Historical CTS and GR times cover all 1,204 archived candidates. The two counts in the table are the physical sample count and prediction batch size. Historical stage times reflect the original server load and are not a controlled speedup comparison.

Aggregation and inference were measured on 2026-11-26 using two CPU threads on an AMD EPYC 9374F, without a GPU. All seven frozen models reproduced their archived predictions; no training, physical-design run or DR labeling was performed for this measurement. Full definitions and hardware details remain available under the page's expandable timing notes.
