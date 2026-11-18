set all_lefs [concat $env(TECH_LEF) $env(SC_LEF) $env(ADDITIONAL_LEFS)]
foreach lef_file $all_lefs {
    read_lef $lef_file
}
read_def $env(RESULTS_DIR)/bottom.def
if {[info exists ::env(LAYER_DP_SAVE_IMAGES)] && $::env(LAYER_DP_SAVE_IMAGES)} {
  save_image -resolution 1 $::env(RESULTS_DIR)/bottom_berfore_legalized.webp
}
set layer_dp_max_displacement 300
if {[info exists ::env(LAYER_DP_MAX_DISPLACEMENT)]} {
  set layer_dp_max_displacement $::env(LAYER_DP_MAX_DISPLACEMENT)
}
detailed_placement -max_displacement $layer_dp_max_displacement
if {[info exists ::env(LAYER_DP_SAVE_IMAGES)] && $::env(LAYER_DP_SAVE_IMAGES)} {
  save_image -resolution 1 $::env(RESULTS_DIR)/bottom_legalized.webp
}
write_def $env(RESULTS_DIR)/bottom_legalized.def
