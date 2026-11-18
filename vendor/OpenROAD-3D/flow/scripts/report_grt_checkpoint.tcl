utl::set_metrics_stage "globalroute__{}"
source $::env(SCRIPTS_DIR)/load.tcl
load_design 5_1_grt.odb 4_cts.sdc "Reporting saved global-route checkpoint"

if {[info exist env(IDEAL_CLOCK)]} {
  set_ideal_network [all_clocks]
} else {
  set_propagated_clock [all_clocks]
}

# Coarse STA must not reuse the skip-rcx finish parasitic model.
# final_report.tcl calls `estimate_parasitics -global_routing` when OpenRCX is
# skipped, so GRT-RC TNS is the official label. Repeating that here copies the
# label and makes Kendall tau-b identically 1. Placement parasitics are a
# strictly cheaper RC model on the same checkpoint netlist.
estimate_parasitics -placement
source $env(SCRIPTS_DIR)/report_metrics.tcl
report_metrics "global route checkpoint" false false
utl::metric_integer "timing__drv__setup_violation_count" [sta::endpoint_violation_count max]
