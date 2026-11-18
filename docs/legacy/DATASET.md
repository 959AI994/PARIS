# Dataset and EDA adapter contract

Schema ID: `paris.dataset.v1`. Each design/candidate pair is unique; identifiers
are strings. Candidate rows, including all associated wire rows, are indivisible
training/evaluation groups.

| File | Grain / purpose |
|---|---|
| `features.parquet` | one CTS/GR-only candidate row; keys, `fast_l1_tns`, `grt_l2_tns`, endpoint summaries and optional pre-DR descriptors |
| `source_sink.parquet` | GR driver/sink/RF/corner rows; geometry, blockage, congestion, delay/slew/slack features |
| `net_load.parquet` | GR net/driver/corner rows with pi R/C1/C2 |
| `labels.parquet` | measured final setup TNS per candidate, separate from features |
| `wire_labels.parquet` | measured DR wire-delay labels for matched historical driver/sink/RF/corner rows |
| `DATASET.json` | schema, settings and content hashes |

Native GR extraction emits 104 endpoint/path summary fields plus two keys and
four CTS/GR TNS/WNS fields (110 columns). It uses all finite timing endpoints;
path descriptors are a bounded GR-selected sample. No full critical-path
coverage or reconstructed DR timing graph is implied. Net selection uses GR
only. Historical DR matching can discard changed/unmatched arcs after selection;
the receipt reports counts, and no synthetic labels are substituted.

Units: time ns, distance um, resistance kOhm, capacitance fF. Tcl reads SDC and
setRC in their original library context, then explicitly changes reporting units.
Captured layer RC is converted from its original command units before geometric
aggregation. Each stage uses the configured executable, exact libraries, SDC,
RC settings and original checkpoint. No new detailed route occurs during
`collect`; `label` can invoke an explicit user routing hook.

Each completed candidate stores hashes of all relevant input files and scripts,
output hashes and STA/endpoint-sum closure results. Cached results are reused
only when hashes match. Partial outputs are preserved and rejected. Configure a
new workspace when changing tool versions, extraction settings or input files.

GR and DR must be part of the user's intended physical sequence; the tool checks
numerical closure but cannot infer whether an arbitrary supplied checkpoint was
produced by the correct optimization history. Imported feature tables likewise
require their owner's availability/label provenance audit.

Default native training validates a held-out candidate subset and chooses its
recipe inside history only. Final refit absorbs all labeled candidates. A frozen
evaluation model/score file must be used for budget replay; predicting training
candidate IDs with the final refit is rejected.

OpenROAD adapter notes: internal timing APIs `sta::endpoints`,
`sta::find_pi_elmore` and `sta::find_elmore`, and GR segment reading must exist in
the selected version. Incompatible versions fail with a retained Tcl log. The
public package does not provide tool binaries, PDKs, process models or extraction
rules. The validated binary versions and real checkpoint test are recorded in
the release's acceptance report, not used as hardcoded defaults.
