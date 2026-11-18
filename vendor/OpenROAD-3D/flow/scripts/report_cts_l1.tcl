utl::set_metrics_stage "cts_l1__{}"
source $::env(SCRIPTS_DIR)/load.tcl
set cts_l1_odb 4_cts.odb
if {[info exists ::env(CTS_L1_ODB)]} {
  set cts_l1_odb $::env(CTS_L1_ODB)
}
set cts_l1_sdc 4_cts.sdc
if {[info exists ::env(CTS_L1_SDC)]} {
  set cts_l1_sdc $::env(CTS_L1_SDC)
}
load_design $cts_l1_odb $cts_l1_sdc "Reporting post-CTS placement-RC L1"

if {[info exist env(IDEAL_CLOCK)]} {
  set_ideal_network [all_clocks]
} else {
  set_propagated_clock [all_clocks]
}

# Official L1 is post-CTS geometry + placement parasitics. GRT is not required:
# 4/4 design probes matched 5_1_grt.odb + estimate_parasitics -placement exactly.
estimate_parasitics -placement
if {[info exists ::env(CTS_L1_TNS_ONLY)] && $::env(CTS_L1_TNS_ONLY)} {
  # Problem 1 ranks candidates by TNS.  Avoid the generic QoR report here:
  # report_power and the remaining diagnostics can take longer than STA itself
  # on the largest designs and do not affect the ranking score.
  report_tns
  report_tns_metric
  report_worst_slack
  report_worst_slack_metric
  utl::metric_integer "timing__drv__setup_violation_count" [sta::endpoint_violation_count max]
} else {
  source $env(SCRIPTS_DIR)/report_metrics.tcl
  report_metrics "cts L1 placement-RC" false false
  utl::metric_integer "timing__drv__setup_violation_count" [sta::endpoint_violation_count max]
}
