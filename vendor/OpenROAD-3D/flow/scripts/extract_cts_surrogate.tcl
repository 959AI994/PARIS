utl::set_metrics_stage "cts_surrogate__{}"
source $::env(SCRIPTS_DIR)/load.tcl

set cts_surrogate_odb 4_cts.odb
if {[info exists ::env(CTS_SURROGATE_ODB)]} {
  set cts_surrogate_odb $::env(CTS_SURROGATE_ODB)
}
set cts_surrogate_sdc 4_cts.sdc
if {[info exists ::env(CTS_SURROGATE_SDC)]} {
  set cts_surrogate_sdc $::env(CTS_SURROGATE_SDC)
}
load_design $cts_surrogate_odb $cts_surrogate_sdc \
  "Extracting post-CTS surrogate features"

if {[info exist env(IDEAL_CLOCK)]} {
  set_ideal_network [all_clocks]
} else {
  set_propagated_clock [all_clocks]
}

# Keep this feature boundary identical to L1: post-CTS geometry and placement
# parasitics only.  No global/detailed routing or OpenRCX data may enter the
# surrogate feature file.
estimate_parasitics -placement

set endpoint_count 4096
if {[info exists ::env(CTS_SURROGATE_ENDPOINT_COUNT)]} {
  set endpoint_count $::env(CTS_SURROGATE_ENDPOINT_COUNT)
}
set output_path $::env(LOG_DIR)/4_cts_surrogate_endpoints.rpt
if {[info exists ::env(CTS_SURROGATE_OUTPUT)]} {
  set output_path $::env(CTS_SURROGATE_OUTPUT)
}
set detail_count 64
if {[info exists ::env(CTS_SURROGATE_DETAIL_COUNT)]} {
  set detail_count $::env(CTS_SURROGATE_DETAIL_COUNT)
}
set detail_path $::env(LOG_DIR)/4_cts_surrogate_detail.json
if {[info exists ::env(CTS_SURROGATE_DETAIL_OUTPUT)]} {
  set detail_path $::env(CTS_SURROGATE_DETAIL_OUTPUT)
}

# One worst setup path per endpoint provides a bounded view of the complete
# tail.  The compact `end` format carries endpoint and timing summaries.  A
# much smaller JSON sample retains per-point delay/slew/capacitance fields for
# path-shape and spatial aggregates without producing multi-gigabyte datasets.
report_checks \
  -path_delay max \
  -group_count $endpoint_count \
  -endpoint_count 1 \
  -unique_paths_to_endpoint \
  -sort_by_slack \
  -format end \
  > $output_path

report_checks \
  -path_delay max \
  -group_count $detail_count \
  -endpoint_count 1 \
  -unique_paths_to_endpoint \
  -sort_by_slack \
  -format json \
  -fields {capacitance slew input_pin net} \
  > $detail_path

report_tns_metric
report_worst_slack_metric
utl::metric_integer "timing__drv__setup_violation_count" \
  [sta::endpoint_violation_count max]
puts "CTS_SURROGATE_PATHS $output_path"
puts "CTS_SURROGATE_DETAIL $detail_path"
