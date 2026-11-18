proc require_env {name} {
  if {![info exists ::env($name)] || $::env($name) eq ""} {
    error "missing required environment variable $name"
  }
  return $::env($name)
}

proc clean_tsv {value} {
  if {[string first "\t" $value] >= 0 || [string first "\n" $value] >= 0 || [string first "\r" $value] >= 0} {
    error "value cannot be represented in TSV: $value"
  }
  return $value
}

proc write_row {channel values} {
  set cleaned {}
  foreach value $values {
    lappend cleaned [clean_tsv $value]
  }
  puts $channel [join $cleaned "\t"]
}

proc read_names {path} {
  set channel [open $path r]
  set names {}
  while {[gets $channel line] >= 0} {
    set line [string trim $line]
    if {$line ne ""} {
      lappend names $line
    }
  }
  close $channel
  return $names
}

proc sta_pin {name kind} {
  return [sta::get_port_pin_error "pin" $name]
}

proc pin_slew {pin transition} {
  set value [get_property $pin "slew_max_${transition}"]
  if {$value eq ""} {
    return ""
  }
  return $value
}

proc pin_slack {pin transition} {
  set value [get_property $pin "slack_max_${transition}"]
  if {$value eq ""} {
    return ""
  }
  return $value
}

proc net_arc_delay {driver sink transition} {
  return [sta::time_sta_ui [sta::find_elmore $driver $sink $transition max]]
}

set odb_path [require_env ODB_PATH]
set sdc_path [require_env SDC_PATH]
set setrc_path [require_env SETRC_PATH]
set output_path [require_env TIMING_OUTPUT]
set stage [require_env EXTRACTION_STAGE]

if {[info exists ::env(LIB_FILES)] && $::env(LIB_FILES) ne ""} {
  foreach liberty [split $::env(LIB_FILES) "|"] {
    read_liberty $liberty
  }
}
read_db $odb_path
read_sdc $sdc_path
set_propagated_clock [all_clocks]
set layer_rc_output [require_env LAYER_RC_OUTPUT]
set layer_rc_channel [open $layer_rc_output w]
write_row $layer_rc_channel {layer resistance_kohm_per_um capacitance_ff_per_um}
rename set_layer_rc original_set_layer_rc
proc set_layer_rc {args} {
  global layer_rc_channel
  array set options {-layer "" -resistance "" -capacitance ""}
  array set options $args
  if {$options(-layer) ne "" && $options(-resistance) ne "" && $options(-capacitance) ne ""} {
    write_row $layer_rc_channel [list $options(-layer) [expr {[sta::resistance_ui_sta $options(-resistance)]/1e3}] [expr {[sta::capacitance_ui_sta $options(-capacitance)]/1e-15}]]
  }
  original_set_layer_rc {*}$args
}
source $setrc_path
set_cmd_units -time ns -capacitance fF -resistance kOhm -distance um
close $layer_rc_channel
rename set_layer_rc {}
rename original_set_layer_rc set_layer_rc

if {$stage eq "gr"} {
  if {[info exists ::env(RERUN_GLOBAL_ROUTE)] && $::env(RERUN_GLOBAL_ROUTE) eq "1"} {
    set routing_range "$::env(ROUTING_MIN_LAYER)-$::env(ROUTING_MAX_LAYER)"
    set_routing_layers -signal $routing_range
    if {$::env(GLOBAL_ROUTE_ADJUSTMENT) > 0.0} {
      set_global_routing_layer_adjustment $routing_range $::env(GLOBAL_ROUTE_ADJUSTMENT)
    }
    global_route -allow_congestion -congestion_iterations $::env(GLOBAL_ROUTE_ITERATIONS)
  } else {
    read_global_route_segments [require_env GR_SEGMENTS_PATH]
  }
  estimate_parasitics -global_routing
} elseif {$stage eq "dr"} {
  if {[info exists ::env(DR_SPEF_INPUT)] && $::env(DR_SPEF_INPUT) ne ""} {
    read_spef $::env(DR_SPEF_INPUT)
  } else {
  set rcx_rules [require_env RCX_RULES]
  define_process_corner -ext_model_index 0 X
  extract_parasitics -ext_model_file $rcx_rules
  set spef_path [require_env DR_SPEF_PATH]
  write_spef $spef_path
  read_spef $spef_path
  }
} else {
  error "EXTRACTION_STAGE must be gr or dr"
}

# 触发 OpenSTA 完整 graph/delay 计算，随后才能查询 pin slew 和 reduced π/Elmore。
report_checks -path_delay max -group_count 1 -endpoint_count 1 > /dev/null

set requested_names [read_names [require_env NET_LIST_PATH]]
set requested {}
foreach name $requested_names {
  dict set requested $name 1
}

set output [open $output_path w]
write_row $output {net_name driver_pin sink_pin transition driver_slew sink_slew slack wire_delay pi_r pi_c1 pi_c2 status}
set block [ord::get_db_block]
foreach net [$block getNets] {
  set net_name [$net getName]
  if {![dict exists $requested $net_name]} {
    continue
  }
  set drivers {}
  set sinks {}
  foreach iterm [$net getITerms] {
    set direction [[$iterm getMTerm] getIoType]
    set record [list "[[$iterm getInst] getName]/[[$iterm getMTerm] getName]" iterm]
    if {$direction eq "OUTPUT" || $direction eq "INOUT"} {
      lappend drivers $record
    }
    if {$direction eq "INPUT" || $direction eq "INOUT"} {
      lappend sinks $record
    }
  }
  foreach bterm [$net getBTerms] {
    set direction [$bterm getIoType]
    set record [list [$bterm getName] bterm]
    if {$direction eq "INPUT" || $direction eq "INOUT"} {
      lappend drivers $record
    }
    if {$direction eq "OUTPUT" || $direction eq "INOUT"} {
      lappend sinks $record
    }
  }
  if {[llength $drivers] != 1 || [llength $sinks] == 0} {
    write_row $output [list $net_name "" "" "" "" "" "" "" "" "" "" unsupported_connectivity]
    continue
  }
  lassign [lindex $drivers 0] driver_name driver_kind
  set driver [sta_pin $driver_name $driver_kind]
  foreach transition {rise fall} {
    set pi [sta::find_pi_elmore $driver $transition max]
    if {[llength $pi] == 3} {
      set pi_c2 [sta::capacitance_sta_ui [lindex $pi 0]]
      set pi_r [sta::resistance_sta_ui [lindex $pi 1]]
      set pi_c1 [sta::capacitance_sta_ui [lindex $pi 2]]
    } else {
      set pi_r ""
      set pi_c1 ""
      set pi_c2 ""
    }
    set driver_slew [pin_slew $driver $transition]
    foreach sink_record $sinks {
      lassign $sink_record sink_name sink_kind
      set sink [sta_pin $sink_name $sink_kind]
      set sink_slew [pin_slew $sink $transition]
      set slack [pin_slack $sink $transition]
      set wire_delay [net_arc_delay $driver $sink $transition]
      set status ok
      if {$pi_r eq "" || $driver_slew eq "" || $sink_slew eq "" || $slack eq "" || $wire_delay eq ""} {
        set status missing_timing
      }
      write_row $output [list $net_name $driver_name $sink_name $transition $driver_slew $sink_slew $slack $wire_delay $pi_r $pi_c1 $pi_c2 $status]
    }
  }
}
close $output
