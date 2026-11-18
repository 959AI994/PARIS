# Changelog

## 0.8.0 - 2026-11-18

* Package the current PARIS web workspace and single-XGBoost CLI with design-specific model banks.
* Bundle the five fixed evaluation groups, training IDs, features, labels, weights and replay scripts.
* Include the used Open3DBench Place-MoL/DREAMPlace and OpenROAD-3D source, licenses and CPU build scripts.
* Add raw-design CLI preparation, budgeted DR calibration, isolated training and GR-only prediction.
* Verify a fresh CPU placement build, real RTL-to-prediction run, CLI/web parity and frozen-model replay.

## 0.5.1 - 2026-11-17

* Reject positive candidate GR/CTS TNS and nonfinite/nonphysical training GR arcs, including arcs without matched DR labels. Preserve the valid signed driver-minus-sink slew difference.

## 0.5.0 - 2026-11-17

* Per-design PARIS iteration workspace: cumulative sample inspection, fixed development holdout and isolated training jobs.
* Full original architecture refit through reference-bundle implementations; native backend is not substituted.
* Comparable GR/current/new validation, conservative manual activation, model export and reversible version pointers.
* Dataset/model signatures, immutable validation labels/GR inputs, recipe/provenance separation and deployment context checks.
* Explicit engineering-smoke records cannot be activated or exported as deployment weights.

## 0.4.0 - 2026-11-17

* INPUT modes for prepared features and raw netlist/project ZIP import.
* Required netlist/LEF/Liberty/SDC inspection with explicit missing-flow blockers.
* Trusted server-side preparation recipes, fixed stage order, isolated artifacts/hashes/logs and original-model compatibility gate.
* Searchable actual Parquet schema/field-family/value catalog, including partial table availability.
* Raw end-to-end execution requires a verified PDK-specific external flow; no universal raw converter is included.
* Original prediction implementation and weights remain unchanged.

## 0.3.0 - 2026-11-17

* Local graphical input/process/output workspace via `paris ui`.
* Server directory selection, full original-model/schema inspection and actual CLI job execution.
* Per-design full ranking with pagination, search, exact tie display, GR rank changes and CSV/Parquet/log downloads.
* Isolated run directories, persistent receipts, loopback binding, scoped roots and same-origin/session checks.
* Original prediction modules and model weights remain unchanged.

## 0.2.0 - 2026-11-17

* Full original seven-design seed-17 PARIS4 runtime, original model restoration and per-design frozen decisions.
* Original-model Parquet output and lossless score readers preserve float64 ties through tool integration.
* Explicit `paris4-native` self-training name distinguishes the portable profile from original weights.
* Regression acceptance covers all 485 real target candidates, with target labels read after prediction freeze.

## 0.1.0 - 2026-11-17

* Unified `paris` CLI and Python backend contract.
* User-configured OpenROAD paths, stage binaries and exact design libraries.
* Independent GR-only feature collection and measured DR label collection.
* Portable PARIS4 self-training, candidate holdout, model sealing and export.
* Original frozen PARIS4 model compatibility adapter.
* Persistent detailed-route execution budgets and explicit user flow hooks.
* Deterministic screening and frozen-prediction offline budget replay.
* Synthetic demonstration, real-checkpoint acceptance and source-only packaging.

Scope: single-corner setup/max timing; no bundled PDK or external EDA binary.
Cross-design models require their own validated backend/schema. A new portable
training recipe does not inherit old experimental accuracy claims. Full RTL-to-GR
flow generation, small-data transfer calibration and commercial-flow adapters
remain future extensions.
