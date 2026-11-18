# Export true STA endpoints, with GR-only path summaries on a bounded sample.
proc needed {name} {
  if {![info exists ::env($name)] || $::env($name) eq ""} { error "missing $name" }
  return $::env($name)
}
proc number {value} {
  if {![string is double -strict $value]} { return "" }
  if {[catch {expr {abs($value) < 1e20}} good] || !$good} { return "" }
  return $value
}
proc property_number {object name} {
  if {[catch {get_property $object $name} value]} { return "" }
  return [number $value]
}
proc field {value} { return [string map [list "\t" " " "\n" " " "\r" " "] $value] }
proc row {channel values} { puts $channel [join $values "\t"] }
proc name_of {object} {
  if {$object eq "" || $object eq "NULL"} { return "" }
  if {[catch {get_property $object full_name} value]} {
    if {[catch {get_property $object name} value]} { return "" }
  }
  return [field $value]
}
proc pin_inst {name} {
  set slash [string last "/" $name]
  if {$slash < 0} { return "" }
  return [string range $name 0 [expr {$slash-1}]]
}

set stage [needed EXTRACTION_STAGE]
foreach liberty [split [needed LIB_FILES] "|"] { read_liberty $liberty }
read_db [needed ODB_PATH]
read_sdc [needed SDC_PATH]
set_propagated_clock [all_clocks]
source [needed SETRC_PATH]
set_cmd_units -time ns -capacitance fF -resistance kOhm -distance um
if {$stage eq "cts"} {
  estimate_parasitics -placement
} elseif {$stage eq "gr"} {
  read_global_route_segments [needed GR_SEGMENTS_PATH]
  estimate_parasitics -global_routing
} elseif {$stage eq "dr"} {
  define_process_corner -ext_model_index 0 X
  extract_parasitics -ext_model_file [needed RCX_RULES]
  write_spef [needed DR_SPEF_PATH]
  read_spef [needed DR_SPEF_PATH]
} else { error "invalid stage $stage" }
report_checks -path_delay max -group_count 1 -endpoint_count 1 > /dev/null
set api_tns [total_negative_slack -max]
set api_wns [worst_slack -max]
set output [open [needed ENDPOINT_OUTPUT] w]
row $output {endpoint slack slack_rise slack_fall arrival_rise arrival_fall slew_rise slew_fall}
set endpoint_sum 0.0
set endpoint_count 0
set finite_count 0
set ordered {}
set endpoint_pins [dict create]
set endpoint_group_api [expr {[llength [info commands sta::path_group_names]] && [llength [info commands sta::endpoint_slack]]}]
puts "ENDPOINT_SLACK_API grouped=$endpoint_group_api; endpoint sum must close to TNS"
foreach pin [sta::endpoints] {
  incr endpoint_count
  set name [name_of $pin]
  # An endpoint with fanout also propagates required times from downstream
  # checks. Its generic pin slack can therefore count a downstream violation
  # again. Match OpenSTA's TNS definition by visiting constrained path groups.
  set slack ""
  if {$endpoint_group_api} {
  foreach group [sta::path_group_names] {
    if {$group eq "unconstrained"} { continue }
    set end_slack [number [sta::endpoint_slack $pin $group max]]
    if {$end_slack ne "" && ($slack eq "" || $end_slack < $slack)} {
      set slack $end_slack
    }
  }
  } else {
    # Older builds lack the endpoint-specific API. Accept pin slack only if
    # the independent endpoint-sum/TNS check below closes; never force it.
    set slack [property_number $pin slack_max]
  }
  if {$slack eq ""} { continue }
  incr finite_count
  set endpoint_sum [expr {$endpoint_sum + min($slack,0.0)}]
  set values [list $name $slack]
  foreach prop {slack_max_rise slack_max_fall arrival_max_rise arrival_max_fall slew_max_rise slew_max_fall} {
    lappend values [property_number $pin $prop]
  }
  row $output $values
  dict set endpoint_pins $name $pin
  lappend ordered [list $slack $name]
}
close $output
set metrics [open [needed METRICS_OUTPUT] w]
row $metrics {metric value}
foreach {key val} [list api_tns $api_tns api_wns $api_wns endpoint_sum $endpoint_sum endpoint_count $endpoint_count finite_endpoints $finite_count] { row $metrics [list $key $val] }
close $metrics
puts "ENDPOINT_AUDIT stage=$stage count=$finite_count api_tns=$api_tns sum=$endpoint_sum"
if {[expr {abs($endpoint_sum-$api_tns)}] > [expr {max(0.002,abs($api_tns)*2e-5)}]} {
  error "true endpoint sum does not match STA TNS"
}

set paths_output [open [needed PATH_OUTPUT] w]
row $paths_output {endpoint path_slack launch_clock capture_clock startpoint start_arrival end_arrival end_required path_span cell_delay wire_delay cell_arcs wire_arcs slew_mean slew_max point_count}
# Only GR selects endpoints and constructs model input. DR exports scalar labels.
if {$stage eq "gr" && [llength $ordered] > 0} {
  set limit 1024
  if {[info exists ::env(PATH_ENDPOINT_LIMIT)]} { set limit $::env(PATH_ENDPOINT_LIMIT) }
  set sorted [lsort -real -index 0 $ordered]
  set selected [dict create]
  # 50% worst slack, 25% nearest zero, 25% spread across the remaining slack range.
  foreach item [lrange $sorted 0 [expr {$limit/2-1}]] { dict set selected [lindex $item 1] 1 }
  set near {}
  foreach item $ordered { lappend near [list [expr {abs([lindex $item 0])}] [lindex $item 1]] }
  set added 0
  foreach item [lsort -real -index 0 $near] {
    set name [lindex $item 1]
    if {![dict exists $selected $name]} { dict set selected $name 1; incr added }
    if {$added >= $limit/4} { break }
  }
  set remaining {}
  foreach item $sorted { if {![dict exists $selected [lindex $item 1]]} { lappend remaining [lindex $item 1] } }
  set need [expr {$limit-[dict size $selected]}]
  if {$need > 0 && [llength $remaining] > 0} {
    for {set i 0} {$i < $need} {incr i} {
      set index [expr {int(($i+0.5)*[llength $remaining]/double($need))}]
      if {$index < [llength $remaining]} { dict set selected [lindex $remaining $index] 1 }
    }
  }
  set to_pins {}
  foreach name [dict keys $selected] { lappend to_pins [dict get $endpoint_pins $name] }
  set count [expr {max(1,3*[llength $to_pins])}]
  set paths [find_timing_paths -path_delay max -to $to_pins -group_count $count -endpoint_count 3 -sort_by_slack]
  foreach path $paths {
    set endpoint [name_of [get_property $path endpoint]]
    set launch [name_of [get_property $path startpoint_clock]]
    set capture [name_of [get_property $path endpoint_clock]]
    set startpoint [name_of [get_property $path startpoint]]
    set slack [property_number $path slack]
    set points [get_property $path points]
    set first_arrival ""
    set last_arrival ""
    set last_required ""
    set previous_name ""
    set previous_arrival ""
    set cell_delay 0.0
    set wire_delay 0.0
    set cell_arcs 0
    set wire_arcs 0
    set slew_sum 0.0
    set slew_max 0.0
    set slew_count 0
    foreach point $points {
      set pin [get_property $point pin]
      set pin_name [name_of $pin]
      set arrival [property_number $point arrival]
      if {$first_arrival eq ""} { set first_arrival $arrival }
      if {$previous_arrival ne "" && $arrival ne ""} {
        set delta [expr {$arrival-$previous_arrival}]
        if {[pin_inst $pin_name] ne "" && [pin_inst $pin_name] eq [pin_inst $previous_name]} {
          set cell_delay [expr {$cell_delay+$delta}]; incr cell_arcs
        } else { set wire_delay [expr {$wire_delay+$delta}]; incr wire_arcs }
      }
      set slew [property_number $pin slew_max]
      if {$slew ne ""} { set slew_sum [expr {$slew_sum+$slew}]; set slew_max [expr {max($slew_max,$slew)}]; incr slew_count }
      set previous_name $pin_name
      set previous_arrival $arrival
      set last_arrival $arrival
    }
    set span ""
    if {$first_arrival ne "" && $last_arrival ne ""} { set span [expr {$last_arrival-$first_arrival}] }
    # Enumerated alternatives do not carry a queryable Path::required in the
    # frozen OpenSTA build. The path-end constraint is arrival + path slack.
    if {$last_arrival ne "" && $slack ne ""} { set last_required [expr {$last_arrival+$slack}] }
    set slew_mean ""
    if {$slew_count > 0} { set slew_mean [expr {$slew_sum/double($slew_count)}] }
    row $paths_output [list $endpoint $slack $launch $capture $startpoint $first_arrival $last_arrival $last_required $span $cell_delay $wire_delay $cell_arcs $wire_arcs $slew_mean $slew_max [llength $points]]
  }
  puts "PATH_AUDIT endpoints=[dict size $selected] paths=[llength $paths]"
}
close $paths_output
