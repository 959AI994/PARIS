# Pre-partition OpenSTA dump from the 2D floorplan DEF + evaluation SDC.
# Required env: LEF_FILES, LIB_FILES, DEF_FILE, SDC_FILE, OUT_DIR
# Optional: PATH_COUNT (default 2000), THREADS (default 8)

set_thread_count [expr {[info exists ::env(THREADS)] ? $::env(THREADS) : 8}]

if {![info exists ::env(LEF_FILES)] || ![info exists ::env(LIB_FILES)] \
    || ![info exists ::env(DEF_FILE)] || ![info exists ::env(SDC_FILE)] \
    || ![info exists ::env(OUT_DIR)]} {
  puts "ERROR: LEF_FILES LIB_FILES DEF_FILE SDC_FILE OUT_DIR are required"
  exit 1
}

set out_dir $::env(OUT_DIR)
file mkdir $out_dir

foreach lef $::env(LEF_FILES) {
  puts "read_lef $lef"
  read_lef $lef
}
foreach lib $::env(LIB_FILES) {
  puts "read_liberty $lib"
  read_liberty $lib
}

puts "read_def $::env(DEF_FILE)"
read_def $::env(DEF_FILE)
puts "read_sdc $::env(SDC_FILE)"
if {[catch { read_sdc $::env(SDC_FILE) } sdc_err]} {
  puts "WARN read_sdc: $sdc_err"
}

catch { set_ideal_network [all_clocks] }
# Force a timing update so pin slack properties are populated.
catch { report_checks -group_count 1 > /dev/null }

set period 1.0
set clocks [all_clocks]
set valid_periods {}
foreach clk $clocks {
  set candidate ""
  catch { set candidate [get_property $clk period] }
  if {[string is double -strict $candidate] && $candidate > 0} {
    lappend valid_periods $candidate
  }
}
if {[llength $valid_periods] > 0} {
  set period [lindex [lsort -real $valid_periods] 0]
}
if {![string is double -strict $period] || $period <= 0} {
  set period 1.0
}
puts "clock_period $period"
puts "clock_count [llength $clocks] normalization minimum_clock_period"

proc csv_quote {value} {
  set text [string map {\" \"\"} $value]
  return "\"$text\""
}

proc worst_pin_slack {obj} {
  set worst ""
  foreach pin [get_pins -quiet -of_objects $obj] {
    foreach prop {slack_max_rise slack_max_fall} {
      if {[catch { set s [get_property $pin $prop] }]} {
        continue
      }
      if {$s eq "" || $s eq "INFINITY" || $s eq "N/A"} {
        continue
      }
      if {![string is double -strict $s]} {
        continue
      }
      if {$worst eq "" || $s < $worst} {
        set worst $s
      }
    }
  }
  foreach port [get_ports -quiet -of_objects $obj] {
    foreach prop {slack_max_rise slack_max_fall} {
      if {[catch { set s [get_property $port $prop] }]} {
        continue
      }
      if {$s eq "" || $s eq "INFINITY" || $s eq "N/A"} {
        continue
      }
      if {![string is double -strict $s]} {
        continue
      }
      if {$worst eq "" || $s < $worst} {
        set worst $s
      }
    }
  }
  return $worst
}

set net_csv [file join $out_dir net_slack.csv]
set fp [open $net_csv w]
puts $fp "name,slack,period,object_type"
set net_count 0
set slack_count 0
foreach net [get_nets -quiet *] {
  incr net_count
  set name [get_full_name $net]
  set slack [worst_pin_slack $net]
  if {$slack eq ""} {
    continue
  }
  incr slack_count
  puts $fp "[csv_quote $name],$slack,$period,net"
}
close $fp
puts "wrote $net_csv nets=$net_count with_slack=$slack_count"

set path_count 2000
if {[info exists ::env(PATH_COUNT)]} {
  set path_count $::env(PATH_COUNT)
}
set path_rpt [file join $out_dir paths.rpt]
report_checks -path_delay max -group_count $path_count > $path_rpt
puts "wrote $path_rpt"

set meta [file join $out_dir sta_meta.json]
set mf [open $meta w]
puts $mf "{ \"clock_period\": $period, \"clock_count\": [llength $clocks], \"period_normalization\": \"minimum_clock_period\", \"net_count\": $net_count, \"net_with_slack\": $slack_count, \"path_count\": $path_count }"
close $mf
puts "STA_DUMP_DONE"
exit
