utl::set_metrics_stage "finish__{}"
source $::env(SCRIPTS_DIR)/load.tcl
load_design 6_1_fill.odb 6_1_fill.sdc "Starting final report"

if {[info exist env(IDEAL_CLOCK)]} {
  set_ideal_network [all_clocks]
  set_false_path -through rst_ni
  set_false_path -through reset_i
  set_false_path -through reset_l
  set_false_path -through p_clk_async_reset_i
} else {
  set_propagated_clock [all_clocks]
}

# Ensure all OR created (rsz/cts) instances are connected
global_connect

# Delete routing obstructions for final DEF
source $::env(SCRIPTS_DIR)/deleteRoutingObstructions.tcl
deleteRoutingObstructions

write_db $::env(RESULTS_DIR)/6_final.odb
write_def $::env(RESULTS_DIR)/6_final.def
write_verilog $::env(RESULTS_DIR)/6_final.v

# Run extraction and STA
set skip_rcx 0
if {[info exist ::env(SKIP_RCX)] && $::env(SKIP_RCX) == "1"} {
  set skip_rcx 1
}
if {[info exist ::env(RCX_RULES)] == 0 || $::env(RCX_RULES) eq ""} {
  set skip_rcx 1
}

if {!$skip_rcx} {

  # Set RC corner for RCX
  # Set in config.mk
  if {[info exist ::env(RCX_RC_CORNER)]} {
    set rc_corner $::env(RCX_RC_CORNER)
  }

  # RCX section
  define_process_corner -ext_model_index 0 X
  extract_parasitics -ext_model_file $::env(RCX_RULES)

  # Write Spef
  write_spef $::env(RESULTS_DIR)/6_final.spef
  file delete $::env(DESIGN_NAME).totCap

  # Read Spef for OpenSTA
  read_spef $::env(RESULTS_DIR)/6_final.spef

} else {
  puts "OpenRCX disabled; using estimated parasitics for final STA."
  # This is the official skip-rcx label. Coarse checkpoint reporting must NOT
  # repeat -global_routing; see scripts/report_grt_checkpoint.tcl.
  estimate_parasitics -global_routing
}

source $::env(SCRIPTS_DIR)/report_metrics.tcl
report_metrics "finish"

# Save a final image if openroad is compiled with the gui
if {[expr [llength [info procs save_image]] > 0]} {
    gui::show "source $::env(SCRIPTS_DIR)/save_images.tcl" false
}
