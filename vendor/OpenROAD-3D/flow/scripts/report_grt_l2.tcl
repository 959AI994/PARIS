utl::set_metrics_stage "grt_l2__{}"
source $::env(SCRIPTS_DIR)/load.tcl
load_design 5_1_grt.odb 4_cts.sdc "Reporting GRT-checkpoint global-routing-RC L2"

if {[info exist env(IDEAL_CLOCK)]} {
  set_ideal_network [all_clocks]
} else {
  set_propagated_clock [all_clocks]
}

# True L2: GRT checkpoint + global-routing parasitics.  Not placement-RC
# (report_grt_checkpoint.tcl) and not the skip-rcx finish model.
estimate_parasitics -global_routing
report_tns
report_tns_metric
report_worst_slack
report_worst_slack_metric
utl::metric_integer "timing__drv__setup_violation_count" [sta::endpoint_violation_count max]
