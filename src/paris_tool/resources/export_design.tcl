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

proc pin_center {bterm} {
  set x_sum 0.0
  set y_sum 0.0
  set count 0
  foreach bpin [$bterm getBPins] {
    foreach box [$bpin getBoxes] {
      set x_sum [expr {$x_sum + ([$box xMin] + [$box xMax]) / 2.0}]
      set y_sum [expr {$y_sum + ([$box yMin] + [$box yMax]) / 2.0}]
      incr count
    }
  }
  if {$count == 0} {
    error "BTerm has no physical box: [$bterm getName]"
  }
  return [list [expr {round($x_sum / $count)}] [expr {round($y_sum / $count)}]]
}

proc write_db_global_segments {block path} {
  set output [open $path w]
  foreach net [$block getNets] {
    set wire [$net getGlobalWire]
    if {$wire eq "NULL"} {
      continue
    }
    puts $output [$net getName]
    puts $output "("
    set decoder [odb::dbWireDecoder]
    $decoder begin $wire
    set current_point {}
    set current_layer ""
    array unset junctions
    while {1} {
      set op [$decoder next]
      if {$op == $odb::dbWireDecoder_END_DECODE} {
        break
      }
      if {$op == $odb::dbWireDecoder_PATH} {
        set current_point {}
        set current_layer [[$decoder getLayer] getName]
      } elseif {$op == $odb::dbWireDecoder_JUNCTION || $op == $odb::dbWireDecoder_SHORT || $op == $odb::dbWireDecoder_VWIRE} {
        set junction_value [$decoder getJunctionValue]
        if {![info exists junctions($junction_value)]} {
          error "wire references unknown junction $junction_value on net [$net getName]"
        }
        lassign $junctions($junction_value) x y current_layer
        set current_point [list $x $y]
      } elseif {$op == $odb::dbWireDecoder_POINT || $op == $odb::dbWireDecoder_POINT_EXT} {
        if {$op == $odb::dbWireDecoder_POINT} {
          lassign [$decoder getPoint] x y
        } else {
          lassign [$decoder getPoint_ext] x y extension
        }
        set layer [[$decoder getLayer] getName]
        if {[llength $current_point] == 2} {
          lassign $current_point previous_x previous_y
          puts $output "$previous_x $previous_y $current_layer $x $y $layer"
        }
        set current_point [list $x $y]
        set current_layer $layer
        set junctions([$decoder getJunctionId]) [list $x $y $layer]
      } elseif {$op == $odb::dbWireDecoder_VIA || $op == $odb::dbWireDecoder_TECH_VIA} {
        if {[llength $current_point] != 2} {
          error "via without current point on net [$net getName]"
        }
        set layer [[$decoder getLayer] getName]
        lassign $current_point x y
        puts $output "$x $y $current_layer $x $y $layer"
        set current_layer $layer
        set junctions([$decoder getJunctionId]) [list $x $y $layer]
      }
    }
    puts $output ")"
  }
  close $output
}

set odb_path [require_env ODB_PATH]
set output_dir [require_env OUTPUT_DIR]
file mkdir $output_dir
read_db $odb_path

set block [ord::get_db_block]
if {$block eq "NULL"} {
  error "ODB does not contain a block: $odb_path"
}
set dbu_per_micron [$block getDbUnitsPerMicron]
set die [$block getDieArea]

set metadata [open [file join $output_dir metadata.tsv] w]
write_row $metadata {dbu_per_micron die_x_min die_y_min die_x_max die_y_max block_name}
write_row $metadata [list $dbu_per_micron [$die xMin] [$die yMin] [$die xMax] [$die yMax] [$block getName]]
close $metadata

set nets [open [file join $output_dir nets.tsv] w]
write_row $nets {net_name signal_type}
foreach net [$block getNets] {
  write_row $nets [list [$net getName] [$net getSigType]]
}
close $nets

set pins [open [file join $output_dir pins.tsv] w]
write_row $pins {net_name pin_name pin_kind direction x_dbu y_dbu}
foreach net [$block getNets] {
  set net_name [$net getName]
  foreach iterm [$net getITerms] {
    lassign [$iterm getAvgXY] placed x y
    if {!$placed} {
      error "ITerm has no physical location: [[ $iterm getInst] getName]/[[$iterm getMTerm] getName]"
    }
    set pin_name "[[$iterm getInst] getName]/[[$iterm getMTerm] getName]"
    write_row $pins [list $net_name $pin_name iterm [[$iterm getMTerm] getIoType] $x $y]
  }
  foreach bterm [$net getBTerms] {
    lassign [pin_center $bterm] x y
    write_row $pins [list $net_name [$bterm getName] bterm [$bterm getIoType] $x $y]
  }
}
close $pins

set macros [open [file join $output_dir macro_obstructions.tsv] w]
write_row $macros {x_min y_min x_max y_max instance_name}
foreach inst [$block getInsts] {
  set master [$inst getMaster]
  set master_type [$master getType]
  if {$master_type eq "BLOCK" || $master_type eq "PAD" || $master_type eq "RING"} {
    set rect [$inst getBBox]
    write_row $macros [list [$rect xMin] [$rect yMin] [$rect xMax] [$rect yMax] [$inst getName]]
  }
}
foreach obstruction [$block getObstructions] {
  set rect [$obstruction getBBox]
  write_row $macros [list [$rect xMin] [$rect yMin] [$rect xMax] [$rect yMax] __block_obstruction__]
}
close $macros

set grid_file [open [file join $output_dir gcell_grid.tsv] w]
write_row $grid_file {axis coordinate_dbu}
set grid [$block getGCellGrid]
if {$grid ne "NULL"} {
  foreach coordinate [$grid getGridX] {
    write_row $grid_file [list x $coordinate]
  }
  foreach coordinate [$grid getGridY] {
    write_row $grid_file [list y $coordinate]
  }
}
close $grid_file

if {[info exists ::env(SEGMENTS_PATH)] && $::env(SEGMENTS_PATH) ne ""} {
  set routing_range "$::env(ROUTING_MIN_LAYER)-$::env(ROUTING_MAX_LAYER)"
  set_routing_layers -signal $routing_range
  if {$::env(GLOBAL_ROUTE_ADJUSTMENT) > 0.0} {
    set_global_routing_layer_adjustment $routing_range $::env(GLOBAL_ROUTE_ADJUSTMENT)
  }
  global_route -allow_congestion -congestion_iterations $::env(GLOBAL_ROUTE_ITERATIONS)
  write_global_route_segments $::env(SEGMENTS_PATH)
}
