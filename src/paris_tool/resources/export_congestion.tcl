proc require_env {name} {
  if {![info exists ::env($name)] || $::env($name) eq ""} {
    error "missing required environment variable $name"
  }
  return $::env($name)
}

proc collection_database {block} {
  if {![catch {$block getDb} db]} {
    return $db
  }
  return [$block getDataBase]
}

proc collection_number {value description} {
  if {![string is double -strict $value]} {
    error "$description is not numeric: $value"
  }
  return $value
}

read_db [require_env GR_ODB]
set block [ord::get_db_block]
set db [collection_database $block]
set tech [$db getTech]
set grid [$block getGCellGrid]
if {$grid eq "NULL"} {
  error "GR checkpoint does not contain a GCell grid"
}

set min_layer [require_env ROUTING_MIN_LAYER]
set max_layer [require_env ROUTING_MAX_LAYER]
set min_tech_layer [$tech findLayer $min_layer]
set max_tech_layer [$tech findLayer $max_layer]
if {$min_tech_layer eq "NULL" || $max_tech_layer eq "NULL"} {
  error "routing layer range does not exist: $min_layer-$max_layer"
}
set min_level [$min_tech_layer getRoutingLevel]
set max_level [$max_tech_layer getRoutingLevel]
set routing_layers {}
foreach layer [$tech getLayers] {
  set level [$layer getRoutingLevel]
  if {$level >= $min_level && $level <= $max_level} {
    lappend routing_layers $layer
  }
}

set x_grid [$grid getGridX]
set y_grid [$grid getGridY]
set output [open [require_env CONGESTION_PATH] w]
puts $output "x,y,capacity,usage"
for {set x 0} {$x < [llength $x_grid]} {incr x} {
  for {set y 0} {$y < [llength $y_grid]} {incr y} {
    set capacity 0.0
    set usage 0.0
    foreach layer $routing_layers {
      set layer_name [$layer getName]
      set layer_capacity [collection_number \
        [$grid getCapacity $layer $x $y] \
        "capacity for $layer_name at ($x, $y)"]
      set layer_usage [collection_number \
        [$grid getUsage $layer $x $y] \
        "usage for $layer_name at ($x, $y)"]
      set capacity [expr {$capacity + $layer_capacity}]
      set usage [expr {$usage + $layer_usage}]
    }
    puts $output "$x,$y,$capacity,$usage"
  }
}
close $output
